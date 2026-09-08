"""
The acquisition plan.

A :class:`useq.MDASequence` carries the declarative schedule (time plan,
positions, per-position channels and exposures, PFS offsets) and is written
next to the data as ``plan.useq.yaml``. :class:`AcquisitionPlan` wraps it with
the rules useq cannot express and PyCLM needs:

- per-channel cadence counted from the experiment's ``t_delay`` (``every_t``),
  and ``t_delay`` / ``t_stop`` themselves,
- the per-position time offset (``time_between_positions``),
- the stimulation channel and the SLM-update / pattern-request events that
  surround it,
- the pattern method's requirements, which set the pattern cadence (the
  Router turns them into subscriptions; nothing about routing is on an event).

Everything PyCLM-specific lives under ``metadata["pyclm"]`` in the sequence,
and this wrapper, not useq's iterator, is the authority for *when* things
happen. useq's iteration is used only for the structure *within* one
timepoint at one position (channels now; z and grid later).

See docs/stage1-plan-design.md.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from math import lcm
from pathlib import Path

import useq

from .events import storage_group
from .experiments import Experiment, ExperimentSchedule, ImagingConfig
from .kinds import seg_kind
from .patterns import AcquiredImageRequest

logger = logging.getLogger(__name__)

PLAN_FORMAT = 1
PLAN_FILENAME = "plan.useq.yaml"
PFS_DEVICE_NAME = "PFS"
DEFAULT_STIM_CHANNEL_NAME = "stimulation"


@dataclass(frozen=True)
class PlannedEvent:
    """One step the Manager carries out at timepoint ``t`` for one experiment.

    ``kind`` is one of ``"request_pattern"``, ``"position"``,
    ``"update_pattern"``, ``"acquire"``. ``index`` identifies the frame
    (``{"t", "p", "c"}``; ``c`` absent for non-acquisition events). Who
    consumes the frame is the Router's decision, not the event's.
    """

    kind: str
    t: int
    experiment: str
    index: dict
    scheduled_offset_s: float
    channel: str | None = None
    is_stim: bool = False
    save: bool = True


@dataclass(frozen=True)
class ExpectedDataset:
    """A frame the plan will produce, described the way the writer needs it."""

    experiment: str
    t: int
    channel: str
    is_stim: bool
    group: str
    config: ImagingConfig
    save: bool


def _needs(req) -> dict:
    out = {
        "raw": bool(req.needs_raw),
        "seg": bool(req.needs_seg),
        "tracks": bool(getattr(req, "needs_tracks", False)),
    }
    named = [str(n) for n in getattr(req, "segmentations", ())]
    if named:
        out["segmentations"] = named
    return out


def requirement_kinds(needs: dict) -> list[str]:
    """The routing kinds a ``pattern_requirements`` entry asks for, in delivery order."""
    kinds = [k for k in ("raw", "seg") if needs.get(k)]
    kinds += [
        seg_kind(n) for n in needs.get("segmentations", ()) if seg_kind(n) != "seg"
    ]
    if needs.get("tracks"):
        kinds.append("tracks")
    return kinds


def _channel_group(exp: Experiment) -> str:
    """The MicroManager config group used to switch imaging channels."""
    for name, cfg in exp.channels.items():
        for cg in cfg.get_config_groups():
            if cg.config == name:
                return cg.group
    return "Channel"


def _stim_channel_name(exp: Experiment, group: str) -> str:
    """The stimulation channel's preset in the channel group, or a fallback."""
    for cg in exp.stimulation.get_config_groups():
        if cg.group == group:
            return cg.config
    return DEFAULT_STIM_CHANNEL_NAME


class AcquisitionPlan:
    """
    The single place scheduling logic lives.

    Build with :meth:`from_schedule` (from TOMLs and the position list) or
    :meth:`from_yaml` (from a previously written ``plan.useq.yaml`` plus the
    schedule that supplies hardware detail). Query with :meth:`events_at`,
    :meth:`is_scheduled`, :meth:`expected_datasets` and
    :meth:`estimate_timepoint_s`.
    """

    def __init__(
        self,
        sequence: useq.MDASequence,
        schedule: ExperimentSchedule,
        requirements: dict[str, list[AcquiredImageRequest]] | None = None,
    ):
        self.sequence = sequence
        self.schedule = schedule
        self.requirements: dict[str, list[AcquiredImageRequest]] = dict(
            requirements or {}
        )

        self._positions = list(sequence.stage_positions)
        self._names = [p.name for p in self._positions]
        self._validate_structure()

        self._meta = {
            p.name: dict(p.sequence.metadata["pyclm"]) for p in self._positions
        }
        # intra-timepoint structure from useq's own iteration (channels; later z/grid)
        self._template = {
            p.name: [
                e
                for e in p.sequence
                if isinstance(e.action, useq.AcquireImage) and e.channel is not None
            ]
            for p in self._positions
        }
        self._validate_metadata()

        # useq yields the start of each timepoint in seconds (older versions: timedelta)
        self._t_offsets = [
            float(v.total_seconds()) if hasattr(v, "total_seconds") else float(v)
            for v in sequence.time_plan
        ]
        self._required = {
            name: self._resolve_requirements(name) for name in self._names
        }
        self._pattern_lcm = {name: self._compute_lcm(name) for name in self._names}

    # ------------------------------------------------------------------ build
    @classmethod
    def from_schedule(
        cls,
        schedule: ExperimentSchedule,
        requirements: dict[str, list[AcquiredImageRequest]] | None = None,
    ) -> AcquisitionPlan:
        times = schedule.times
        requirements = dict(requirements or {})
        positions = []

        for name, exp in schedule.experiments.items():
            pos = schedule.positions[name]
            group = _channel_group(exp)
            stim = exp.stimulation
            stim_name = _stim_channel_name(exp, group) if stim.exposure > 0 else None

            channels = []
            every_t = {}
            binning = {}
            if stim_name is not None:
                channels.append(
                    useq.Channel(
                        config=stim_name,
                        group=group,
                        exposure=float(stim.exposure),
                        do_stack=False,
                    )
                )
                every_t[stim_name] = int(stim.every_t)
                binning[stim_name] = int(stim.binning)
            for cname, cfg in exp.channels.items():
                channels.append(
                    useq.Channel(
                        config=cname, group=group, exposure=float(cfg.exposure)
                    )
                )
                every_t[cname] = int(cfg.every_t)
                binning[cname] = int(cfg.binning)

            meta = {
                "experiment": name,
                "t_delay": int(exp.t_delay),
                "t_stop": int(exp.t_stop),
                "stim_channel": stim_name,
                "every_t": every_t,
                "pattern_every_t": int(exp.pattern.every_t),
                "binning": binning,
                "pattern_requires": cls._requirements_by_name(
                    exp, stim_name, requirements.get(name)
                ),
            }

            sub = {"channels": channels, "metadata": {"pyclm": meta}}
            pfs = pos.extras.get("PFSOffset") if hasattr(pos, "extras") else None
            if pfs is not None:
                # recorded for provenance and tooling; PositionMover still owns focus
                sub["autofocus_plan"] = useq.AxesBasedAF(
                    autofocus_device_name=PFS_DEVICE_NAME,
                    autofocus_motor_offset=float(pfs),
                    axes=("p",),
                )

            positions.append(
                useq.Position(
                    x=pos.x,
                    y=pos.y,
                    z=pos.z,
                    name=name,
                    sequence=useq.MDASequence(**sub),
                )
            )

        sequence = useq.MDASequence(
            time_plan=useq.TIntervalLoops(interval=times.interval, loops=times.count),
            stage_positions=positions,
            metadata={
                "pyclm": {
                    "plan_format": PLAN_FORMAT,
                    "useq_version": _pkg_version("useq-schema"),
                    "setup_time_seconds": float(times.setup),
                    "time_between_positions": float(times.between),
                }
            },
        )
        return cls(sequence, schedule, requirements)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        schedule: ExperimentSchedule,
        requirements: dict[str, list[AcquiredImageRequest]] | None = None,
    ) -> AcquisitionPlan:
        return cls(useq.MDASequence.from_file(str(path)), schedule, requirements)

    def yaml_str(self) -> str:
        return self.sequence.yaml()

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.yaml_str(), encoding="utf-8")
        return path

    @staticmethod
    def _requirements_by_name(exp, stim_name, reqs) -> dict[str, dict[str, bool]]:
        out = {}
        for req in reqs or ():
            if stim_name is not None and req.id == exp.stimulation.channel_id:
                out[stim_name] = _needs(req)
                continue
            for cname, cfg in exp.channels.items():
                if cfg.channel_id == req.id:
                    out[cname] = _needs(req)
        return out

    # --------------------------------------------------------------- validate
    def _validate_structure(self):
        if not self._positions:
            raise ValueError("plan has no positions")
        if len(set(self._names)) != len(self._names):
            raise ValueError(f"duplicate position names in plan: {self._names}")
        if set(self._names) != set(self.schedule.experiment_names):
            raise ValueError(
                f"plan positions {sorted(self._names)} do not match schedule "
                f"experiments {sorted(self.schedule.experiment_names)}"
            )
        for p in self._positions:
            if p.sequence is None or "pyclm" not in p.sequence.metadata:
                raise ValueError(f"position {p.name!r} has no pyclm metadata")
        if self.sequence.time_plan is None:
            raise ValueError("plan has no time plan")
        if "pyclm" not in self.sequence.metadata:
            raise ValueError("plan has no top-level pyclm metadata")

    def _validate_metadata(self):
        needed = {
            "experiment",
            "t_delay",
            "t_stop",
            "stim_channel",
            "every_t",
            "pattern_every_t",
        }
        for name, meta in self._meta.items():
            missing = needed - set(meta)
            if missing:
                raise ValueError(
                    f"position {name!r} metadata missing {sorted(missing)}"
                )
            channels = {e.channel.config for e in self._template[name]}
            unknown = set(meta["every_t"]) - channels
            if unknown:
                raise ValueError(
                    f"position {name!r}: every_t names channels not in the sequence: {sorted(unknown)}"
                )
            if (
                meta["stim_channel"] is not None
                and meta["stim_channel"] not in channels
            ):
                raise ValueError(
                    f"position {name!r}: stimulation channel {meta['stim_channel']!r} not in the sequence"
                )

    # ------------------------------------------------------------ requirements
    def _resolve_requirements(self, name: str) -> dict[str, dict[str, bool]]:
        """{channel name: {"raw", "seg", "tracks"}} for the pattern method of one experiment."""
        exp = self.schedule.experiments[name]
        stim_name = self._meta[name]["stim_channel"]
        reqs = self.requirements.get(name)
        if reqs:
            by_name = self._requirements_by_name(exp, stim_name, reqs)
        else:
            by_name = self._meta[name].get("pattern_requires") or {}
        out = {}
        for c, v in by_name.items():
            needs = {k: bool(v.get(k, False)) for k in ("raw", "seg", "tracks")}
            named = [str(n) for n in v.get("segmentations", ())]
            if named:
                needs["segmentations"] = named
            out[c] = needs
        return out

    def _compute_lcm(self, name: str) -> int:
        meta = self._meta[name]
        values = [int(meta["pattern_every_t"])]
        values += [
            int(meta["every_t"][c])
            for c in self._required[name]
            if c in meta["every_t"]
        ]
        return lcm(*values)

    # --------------------------------------------------------------- queries
    @property
    def experiments(self) -> list[str]:
        """Experiment (position) names in acquisition order."""
        return list(self._names)

    @property
    def timepoints(self) -> int:
        return len(self._t_offsets)

    @property
    def setup_s(self) -> float:
        return float(self.sequence.metadata["pyclm"]["setup_time_seconds"])

    @property
    def between_s(self) -> float:
        return float(self.sequence.metadata["pyclm"]["time_between_positions"])

    @property
    def interval_s(self) -> float:
        if len(self._t_offsets) > 1:
            return self._t_offsets[1] - self._t_offsets[0]
        return float(self.schedule.times.interval)

    def time_offset_s(self, t: int) -> float:
        """Seconds after the start at which timepoint ``t`` begins."""
        return self._t_offsets[t]

    def structure(self, experiment: str) -> list[useq.MDAEvent]:
        """useq's events for one timepoint at one position (channels now; z/grid later)."""
        return list(self._template[experiment])

    def channels(self, experiment: str) -> list[str]:
        """Channel names at a position, stimulation first, in acquisition order."""
        return [e.channel.config for e in self._template[experiment]]

    def stim_channel(self, experiment: str) -> str | None:
        return self._meta[experiment]["stim_channel"]

    def imaging_config(self, experiment: str, channel: str) -> ImagingConfig:
        exp = self.schedule.experiments[experiment]
        if channel == self._meta[experiment]["stim_channel"]:
            return exp.stimulation
        return exp.channels[channel]

    def every_t(self, experiment: str, channel: str) -> int:
        return int(self._meta[experiment]["every_t"][channel])

    def pattern_lcm(self, experiment: str) -> int:
        return self._pattern_lcm[experiment]

    def _relative_t(self, experiment: str, t: int) -> int | None:
        """Experiment-relative timepoint, or None when the experiment is inactive at ``t``."""
        meta = self._meta[experiment]
        this_t = t - int(meta["t_delay"])
        if this_t < 0:
            return None
        if int(meta["t_stop"]) > 0 and this_t >= int(meta["t_stop"]):
            return None
        return this_t

    def is_active(self, experiment: str, t: int) -> bool:
        return self._relative_t(experiment, t) is not None

    def is_scheduled(self, experiment: str, channel: str, t: int) -> bool:
        """Whether ``channel`` (a channel or the stimulation channel) is acquired at ``t``."""
        every = self._meta[experiment]["every_t"].get(channel)
        if every is None:
            return False
        this_t = self._relative_t(experiment, t)
        if this_t is None:
            return False
        return this_t % int(every) == 0

    def pattern_due(self, experiment: str, t: int) -> bool:
        """Whether the experiment's pattern method runs at ``t`` (so its inputs are routed to it)."""
        this_t = self._relative_t(experiment, t)
        return this_t is not None and this_t % self._pattern_lcm[experiment] == 0

    def pattern_requirements(self, experiment: str) -> dict[str, dict]:
        """
        ``{channel: {"raw", "seg", "tracks"[, "segmentations"]}}`` the pattern
        method needs when it runs (``segmentations`` lists the named
        segmentation tables; see :func:`requirement_kinds`).
        """
        return {c: dict(v) for c, v in self._required[experiment].items()}

    def events_at(self, t: int) -> list[PlannedEvent]:
        """Everything the Manager does at timepoint ``t``, in order."""
        out: list[PlannedEvent] = []
        t_offset = self.time_offset_s(t)

        for p_index, name in enumerate(self._names):
            this_t = self._relative_t(name, t)
            if this_t is None:
                continue

            acquisitions = [
                e
                for e in self._template[name]
                if self.is_scheduled(name, e.channel.config, t)
            ]
            if not acquisitions:
                continue

            offset = t_offset + p_index * self.between_s
            make_pattern = this_t % self._pattern_lcm[name] == 0
            base_index = {"t": t, "p": name}

            if make_pattern:
                out.append(PlannedEvent("request_pattern", t, name, base_index, offset))
            out.append(PlannedEvent("position", t, name, base_index, offset))

            for e in acquisitions:
                channel = e.channel.config
                is_stim = channel == self._meta[name]["stim_channel"]
                index = {**base_index, "c": channel}
                config = self.imaging_config(name, channel)
                if is_stim:
                    out.append(
                        PlannedEvent(
                            "update_pattern", t, name, index, offset, channel, True
                        )
                    )
                out.append(
                    PlannedEvent(
                        "acquire",
                        t,
                        name,
                        index,
                        offset,
                        channel,
                        is_stim,
                        save=bool(config.save),
                    )
                )
        return out

    def datasets_at(self, experiment: str, t: int) -> list[ExpectedDataset]:
        out = []
        for e in self._template[experiment]:
            channel = e.channel.config
            if not self.is_scheduled(experiment, channel, t):
                continue
            is_stim = channel == self._meta[experiment]["stim_channel"]
            config = self.imaging_config(experiment, channel)
            out.append(
                ExpectedDataset(
                    experiment,
                    t,
                    channel,
                    is_stim,
                    storage_group(channel, is_stim),
                    config,
                    bool(config.save),
                )
            )
        return out

    def expected_datasets(self) -> Iterator[ExpectedDataset]:
        """Every frame the plan will produce, in acquisition order."""
        for t in range(self.timepoints):
            for name in self._names:
                yield from self.datasets_at(name, t)

    # ---------------------------------------------------------------- timing
    def estimate_timepoint_s(
        self, t: int, settle_s: float, move_s: float = 0.0
    ) -> float:
        """
        Rough duration of timepoint ``t``: per acquisition the exposure plus the
        settle time, plus ``move_s`` per position visited. Stage moves and SLM
        uploads are not modelled beyond ``move_s``.
        """
        total = 0.0
        for ev in self.events_at(t):
            if ev.kind == "position":
                total += move_s
            elif ev.kind == "acquire":
                total += (
                    settle_s
                    + self.imaging_config(ev.experiment, ev.channel).exposure / 1000.0
                )
        return total

    def over_budget(
        self, settle_s: float, move_s: float = 0.0
    ) -> list[tuple[int, float]]:
        """Timepoints whose estimated duration exceeds the interval, as (t, seconds)."""
        interval = self.interval_s
        return [
            (t, est)
            for t in range(self.timepoints)
            if (est := self.estimate_timepoint_s(t, settle_s, move_s)) > interval
        ]

    def __repr__(self):
        return (
            f"AcquisitionPlan({len(self._names)} experiments, {self.timepoints} timepoints, "
            f"interval {self.interval_s}s)"
        )
