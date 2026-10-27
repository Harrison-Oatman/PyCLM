from typing import ClassVar

import numpy as np
import tifffile
from skimage.measure import regionprops

from ..measure import PerTrack, Regions
from .dose import check_dose_mode, deliver
from .pattern import AcquiredImageRequest, DataDock, PatternMethod
from .zoo import ZooMeta

CONTROLLERS = ("bang_bang", "p", "pi")


def check_controller(name: str, controller: str, kp: float, ki: float) -> None:
    if controller not in CONTROLLERS:
        raise ValueError(
            f"{name}: controller must be one of "
            f"{', '.join(map(repr, CONTROLLERS))}, got {controller!r}"
        )
    if kp < 0 or ki < 0:
        raise ValueError(f"{name}: kp and ki must not be negative")


def control(controller, error, kp, ki, integral, dt_min):
    """
    One controller step on the relative error ``(target - value) / target``
    (array or scalar). Returns the light dose in 0-1 and the new integral
    term; the integral is held in 0-1, so it cannot wind up while the light
    is saturated.
    """
    error = np.asarray(error, dtype=float)
    if controller == "bang_bang":
        return (error > 0).astype(float), integral
    if controller == "p":
        return np.clip(kp * error, 0.0, 1.0), integral
    integral = np.clip(integral + ki * error * dt_min, 0.0, 1.0)
    return np.clip(kp * error + integral, 0.0, 1.0), integral


class NucleusControlMethod(PatternMethod):
    name = "nucleus_control_base"
    # early TOMLs named the channel nuc_channel (silently ignored until 2.0)
    renamed_kwargs: ClassVar[dict[str, str]] = {"nuc_channel": "channel"}

    def __init__(self, channel=None, tracks=False, **kwargs):
        super().__init__(**kwargs)

        if channel is None:
            raise AttributeError(
                "NucleusControlMethod must be provided a "
                'segmentation channel in the .toml: e.g. channel = "545"'
            )

        self.channel = channel
        # with tracks = true the per-cell loop runs on tracked labels, so
        # prop.label is a track id that persists across timepoints
        self.use_tracks = bool(tracks)
        self.add_requirement(
            channel, raw=True, seg=not self.use_tracks, tracks=self.use_tracks
        )

    def process_prop(self, prop):
        return prop.image

    def cell_labels(self, context) -> np.ndarray:
        """The label image the per-cell loop runs on: tracked labels with ``tracks = true``, else the segmentation."""
        if self.use_tracks:
            return context.tracks(self.channel).labels
        return context.segmentation(self.channel)

    def generate(self, context) -> np.ndarray:
        seg = self.cell_labels(context)
        raw = context.raw(self.channel)

        h, w = self.pattern_shape

        new_img = np.zeros((int(h), int(w)), dtype=np.float16)

        for prop in regionprops(seg, intensity_image=raw):
            cell_stim = self.process_prop(prop)

            new_img[prop.bbox[0] : prop.bbox[2], prop.bbox[1] : prop.bbox[3]] += (
                cell_stim
            )

        new_img_clamped = np.clip(new_img, 0, 1).astype(np.float16)

        return new_img_clamped


class BinaryNucleusClampModel(NucleusControlMethod):
    """
    Holds each nucleus of ``channel`` at ``target`` (mean intensity, camera
    counts). ``controller`` decides each nucleus's light from its relative
    error ``e = (target - mean) / target``:

    - ``"bang_bang"`` (default): fully lit while dimmer than the target;
    - ``"p"``: a dose of ``kp * e`` (0-1);
    - ``"pi"``: ``kp * e`` plus ``ki`` x the error summed over minutes, per
      tracked nucleus (so it needs a ``[tracking]`` table).

    A dose between 0 and 1 is delivered by ``dose_mode`` (``"gray"``,
    ``"dither"``, or ``"duty"`` over ``period_min``).
    """

    name = "nucleus_clamp"
    renamed_kwargs: ClassVar[dict[str, str]] = {"clamp_target": "target"}

    def __init__(
        self,
        channel=None,
        target=10000,
        controller="bang_bang",
        kp=1.0,
        ki=0.1,
        dose_mode="gray",
        period_min=10.0,
        **kwargs,
    ):
        """
        :param channel: the channel whose nuclei are measured and lit
        :param target: mean nuclear intensity (camera counts) to hold each
            nucleus at: a nucleus dimmer than this is lit
        :param controller: "bang_bang", "p" or "pi"
        :param kp: dose per unit of relative error ("p", "pi")
        :param ki: dose per unit of relative error per minute, summed ("pi")
        :param dose_mode: "gray", "dither" or "duty": how a dose reaches the sample
        :param period_min: the period of a "duty" dose, in minutes
        """
        check_controller(self.name, controller, kp, ki)
        if controller == "pi":
            # the summed error belongs to a nucleus, so nuclei must be tracked
            kwargs["tracks"] = True
        super().__init__(channel=channel, **kwargs)
        check_dose_mode(self.name, dose_mode, period_min)
        if controller != "bang_bang" and target <= 0:
            raise ValueError(f"{self.name}: target must be positive, got {target}")

        self.target = target
        self.controller = controller
        self.kp = float(kp)
        self.ki = float(ki)
        self.dose_mode = dose_mode
        self.period_min = float(period_min)
        self._integral = PerTrack()
        self._last_time = None

    zoo_meta = ZooMeta(
        source="mcf10a",
        title="Nucleus Clamp",
        kwargs={"channel": "545", "target": 65},
        description="Holds each nucleus at a target intensity: bang-bang (lit while dimmer), or a proportional (p) or PI dose.",
        group="per_nucleus",
    )

    def process_prop(self, prop):
        if prop.intensity_mean > self.target:
            return prop.image * 0

        return prop.image

    def doses(self, context, regions: Regions, raw) -> np.ndarray:
        """Each nucleus's dose (aligned with ``regions.ids``) from its controller."""
        ids = regions.ids
        error = (self.target - regions.measure(raw)) / self.target
        dt_min = 0.0
        if self._last_time is not None:
            dt_min = max(0.0, (context.time - self._last_time) / 60)
        self._last_time = context.time
        integral = self._integral.get(ids, default=0.0)
        dose, integral = control(
            self.controller, error, self.kp, self.ki, integral, dt_min
        )
        if self.controller == "pi":
            self._integral.update(ids, integral)
        return dose

    def generate(self, context) -> np.ndarray:
        if self.controller == "bang_bang":
            return super().generate(context)
        regions = Regions(self.cell_labels(context))
        dose = regions.paint(self.doses(context, regions, context.raw(self.channel)))
        return deliver(dose, self.dose_mode, context.time / 60, self.period_min)


class CenteredImageModel(NucleusControlMethod):
    name = "target_image"
    renamed_kwargs: ClassVar[dict[str, str]] = {
        "tif_path": "path",
        "min_intensity": "target_min",
        "max_intensity": "target_max",
    }
    zoo_meta = ZooMeta(
        source="mcf10a",
        title="Target Image",
        kwargs={
            "channel": "545",
            "path": "target_shield.tif",
            "target_min": 0,
            "target_max": 100,
        },
        description="Bang-bang control towards a target image centered on the field: each nucleus is lit while it is dimmer than the target at its centroid, rescaled to target_min..target_max.",
        group="per_nucleus",
    )

    def __init__(
        self,
        channel=None,
        path=None,
        target_min=2000,
        target_max=5000,
        **kwargs,
    ):
        """
        :param channel: the channel whose nuclei are measured and lit
        :param path: the target image (a TIF), centered on the field
        :param target_min: the target intensity (camera counts) of the
            image's darkest pixel
        :param target_max: the target intensity of its brightest pixel
        """
        super().__init__(channel=channel, **kwargs)

        if path is None:
            raise ValueError(f"{self.name}: path (the target TIF) is required")
        self.img = tifffile.imread(path)
        self.target_min = target_min
        self.target_max = target_max

        self.target_image = None

    def make_target_image(self):
        img = self.img

        h, w = self.pattern_shape

        h = int(h)
        w = int(w)

        padding_h_up = (h - img.shape[0]) // 2
        padding_h_down = h - (padding_h_up + img.shape[0])
        padding_w_left = (w - img.shape[1]) // 2
        padding_w_right = w - (padding_w_left + img.shape[1])

        padded_img = np.pad(
            img, ((padding_h_up, padding_h_down), (padding_w_left, padding_w_right))
        )
        padded_img = np.array(padded_img, dtype=np.float16)

        max_v = np.max(img)
        min_v = np.min(img)

        padded_img = (padded_img - min_v) / (max_v - min_v)
        padded_img = np.clip(padded_img, 0, 1)

        padded_img = (
            padded_img * (self.target_max - self.target_min)
        ) + self.target_min

        self.target_image = padded_img

    def get_target_intensity(self, prop):
        y, x = prop.centroid

        y = round(y)
        x = round(x)

        return self.target_image[y, x]

    def process_prop(self, prop):
        target = self.get_target_intensity(prop)

        if prop.intensity_mean > target:
            return prop.image * 0

        return prop.image

    def generate(self, context) -> np.ndarray:
        if self.target_image is None:
            self.make_target_image()

        return super().generate(context)


class GlobalCycleModel(NucleusControlMethod):
    name = "global_cycle"

    deprecated = (
        'use method = "global_pulse" with period_min = 2 * period_m, on_min = period_m'
    )
    zoo_meta = ZooMeta(
        same_as="global_pulse",
        equivalent="period_min = 2 * period_m, on_min = period_m",
    )

    def __init__(self, channel="545", period_m=10, **kwargs):
        super().__init__(channel=channel, **kwargs)

        self.period_s = period_m * 60

    def generate(self, context) -> np.ndarray:
        t = context.time

        is_on = ((t // self.period_s) % 2) == 0

        h, w = self.pattern_shape

        return np.ones((int(h), int(w))) * is_on


class PopulationClampPattern(PatternMethod):
    """
    Holds the mean nuclear intensity of the whole field (the mean of the
    nuclei's means in ``channel``) at ``target`` with one input for every
    cell. ``controller`` (``"pi"``, ``"p"`` or ``"bang_bang"``, as in
    ``nucleus_clamp``) turns the relative error into a level ``u`` in 0-1,
    which ``actuator`` applies:

    - ``"duty"``: the whole field gets a dose ``u``, delivered by
      ``dose_mode`` (gray level, dither, or a duty over ``period_min``);
    - ``"exposure"``: the field is fully lit and the stimulation exposure is
      set to ``min_value + u * (max_value - min_value)`` ms;
    - ``"property"``: the field is fully lit and the stimulation channel's
      ``device`` / ``property`` (a laser power, say) is set the same way.

    Exposure and property changes take effect from the next timepoint.
    """

    name = "population_clamp"
    ACTUATORS = ("duty", "exposure", "property")
    zoo_meta = ZooMeta(
        source="mcf10a",
        kwargs={"channel": "545", "target": 80, "controller": "pi"},
        title="Population Clamp",
        description="Holds the mean nuclear intensity of the field at a target with one input: a whole-field dose, the stimulation exposure, or a device property.",
        pending=(
            "a time series of a nuclear reporter under whole-field light "
            "(the controller's output depends on how the population responds)"
        ),
        group="per_nucleus",
    )

    def __init__(
        self,
        channel=None,
        target=None,
        controller="pi",
        kp=1.0,
        ki=0.1,
        actuator="duty",
        device=None,
        property=None,
        min_value=None,
        max_value=None,
        dose_mode="gray",
        period_min=10.0,
        **kwargs,
    ):
        """
        :param channel: the channel whose nuclei are measured
        :param target: the mean nuclear intensity to hold, in camera counts
        :param controller: "pi", "p" or "bang_bang"
        :param kp: level per unit of relative error
        :param ki: level per unit of relative error per minute, summed ("pi")
        :param actuator: "duty", "exposure" or "property"
        :param device: the MicroManager device ("property")
        :param property: its property ("property")
        :param min_value: the exposure (ms) or property value at level 0
        :param max_value: the exposure (ms) or property value at level 1
        :param dose_mode: "gray", "dither" or "duty" ("duty" actuator)
        :param period_min: the period of a "duty" dose, in minutes
        """
        super().__init__(**kwargs)
        if channel is None:
            raise AttributeError(
                f'{self.name} must be given the nuclear channel: e.g. channel = "545"'
            )
        if target is None or target <= 0:
            raise ValueError(f"{self.name}: target must be positive, got {target}")
        check_controller(self.name, controller, kp, ki)
        check_dose_mode(self.name, dose_mode, period_min)
        if actuator not in self.ACTUATORS:
            raise ValueError(
                f"{self.name}: actuator must be one of "
                f"{', '.join(map(repr, self.ACTUATORS))}, got {actuator!r}"
            )
        if actuator != "duty":
            if min_value is None or max_value is None:
                raise ValueError(
                    f"{self.name}: actuator = {actuator!r} needs min_value and max_value"
                )
            if actuator == "property" and (device is None or property is None):
                raise ValueError(
                    f'{self.name}: actuator = "property" needs device and property'
                )
        self.channel = channel
        self.target = float(target)
        self.controller = controller
        self.kp = float(kp)
        self.ki = float(ki)
        self.actuator = actuator
        self.device = device
        self.property = property
        self.min_value = min_value
        self.max_value = max_value
        self.dose_mode = dose_mode
        self.period_min = float(period_min)
        self.integral = 0.0
        self.level = 0.0
        self._last_time = None
        self._last_value = None
        self.add_requirement(channel, raw=True, seg=True)

    def measure(self, context) -> float | None:
        """The mean of the nuclei's mean intensities, or None with no nuclei."""
        regions = context.regions(self.channel)
        if len(regions) == 0:
            return None
        return float(np.mean(regions.measure(context.raw(self.channel))))

    def generate(self, context) -> np.ndarray:
        value = self.measure(context)
        dt_min = 0.0
        if self._last_time is not None:
            dt_min = max(0.0, (context.time - self._last_time) / 60)
        self._last_time = context.time
        if value is not None:
            error = (self.target - value) / self.target
            level, integral = control(
                self.controller, error, self.kp, self.ki, self.integral, dt_min
            )
            self.level, self.integral = float(level), float(integral)

        h, w = (int(v) for v in self.pattern_shape)
        if self.actuator == "duty":
            dose = np.full((h, w), self.level, dtype=np.float32)
            return deliver(dose, self.dose_mode, context.time / 60, self.period_min)

        setting = self.min_value + self.level * (self.max_value - self.min_value)
        if setting != self._last_value:
            if self.actuator == "exposure":
                context.set_exposure("stimulation", setting)
            else:
                context.set_property("stimulation", self.device, self.property, setting)
            self._last_value = setting
        return np.ones((h, w), dtype=np.float32)


class RelativeClampPattern(PatternMethod):
    """
    Holds each tracked cell at a fraction of *its own* range, so cells with
    different expression levels can be clamped together. It calibrates
    first, from ``start_min``: the field dark for ``calibrate_min`` (each
    cell's dark value is its last reading), then fully lit for
    ``calibrate_min`` (its lit value). Then it holds ``targets[0]`` for
    ``step_min``, ``targets[1]`` for the next ``step_min`` and so on,
    keeping the last: a target ``f`` is the reading ``dark + f x (lit -
    dark)``, and a cell is lit while its reading is on the dark side of it
    (bang-bang, so it works whether light raises or lowers the reading).
    Cells whose range is under ``min_range`` (log2 of lit / dark) and cells
    first seen after the calibration stay dark.

    ``channel`` is measured; ``seg_channel`` (default: ``channel``) is
    segmented and tracked, which needs a ``[tracking]`` table.
    """

    name = "relative_clamp"
    zoo_meta = ZooMeta(
        source="mcf10a",
        kwargs={
            "channel": "545",
            "calibrate_min": 30,
            "targets": [0.25, 0.5, 0.75],
            "step_min": 60,
        },
        title="Relative Clamp",
        description="Calibrates each tracked cell's own dark and lit levels, then holds it at a fraction of that range.",
        pending=(
            "a time series of a nuclear reporter through a dark and a lit "
            "phase (each cell's range is measured from the run itself)"
        ),
        group="per_nucleus",
    )

    def __init__(
        self,
        channel=None,
        seg_channel=None,
        calibrate_min=30.0,
        targets=(0.5,),
        step_min=60.0,
        min_range=0.5,
        **kwargs,
    ):
        """
        :param channel: the channel whose intensity is clamped
        :param seg_channel: the channel segmented and tracked (default: channel)
        :param calibrate_min: minutes of each calibration phase (dark, then lit)
        :param targets: fractions (0-1) of each cell's range, one per step_min
        :param step_min: minutes each target is held
        :param min_range: smallest range (log2 of lit / dark) a cell needs
        """
        super().__init__(**kwargs)
        if channel is None:
            raise AttributeError(
                f'{self.name} must be given the measured channel: e.g. channel = "545"'
            )
        targets = [float(t) for t in np.atleast_1d(targets)]
        if not targets or not all(0.0 <= t <= 1.0 for t in targets):
            raise ValueError(f"{self.name}: targets must be a list of values in 0-1")
        if calibrate_min <= 0 or step_min <= 0:
            raise ValueError(
                f"{self.name}: calibrate_min and step_min must be positive"
            )
        if min_range < 0:
            raise ValueError(f"{self.name}: min_range must not be negative")
        self.channel = channel
        self.seg_channel = channel if seg_channel is None else seg_channel
        self.calibrate_min = float(calibrate_min)
        self.targets = targets
        self.step_min = float(step_min)
        self.min_range = float(min_range)
        self.dark = PerTrack()
        self.lit = PerTrack()
        self.add_requirement(channel, raw=True)
        self.add_requirement(self.seg_channel, tracks=True)

    def phase(self, t_minutes: float) -> str:
        """``"dark"``, ``"lit"`` (calibration) or ``"clamp"``."""
        t_rel = t_minutes - self.start_min
        if t_rel < self.calibrate_min:
            return "dark"
        if t_rel < 2 * self.calibrate_min:
            return "lit"
        return "clamp"

    def target_at(self, t_minutes: float) -> float:
        """The fraction of the range held at ``t_minutes`` (during the clamp)."""
        t_rel = t_minutes - self.start_min - 2 * self.calibrate_min
        step = int(max(t_rel, 0.0) // self.step_min)
        return self.targets[min(step, len(self.targets) - 1)]

    def generate(self, context) -> np.ndarray:
        h, w = (int(v) for v in self.pattern_shape)
        t_min = context.time / 60
        phase = self.phase(t_min)
        tracks = context.tracks(self.seg_channel)
        if tracks is None:
            return np.full((h, w), float(phase == "lit"), dtype=np.float32)
        regions = Regions(tracks.labels)
        ids = regions.ids
        reading = regions.measure(context.raw(self.channel))
        if phase == "dark":
            self.dark.update(ids, reading)
            return np.zeros((h, w), dtype=np.float32)
        if phase == "lit":
            self.lit.update(ids, reading)
            return np.ones((h, w), dtype=np.float32)

        dark = self.dark.get(ids)
        lit = self.lit.get(ids)
        with np.errstate(divide="ignore", invalid="ignore"):
            fold = np.abs(np.log2(lit / dark))
        usable = np.isfinite(fold) & (dark > 0) & (lit > 0) & (fold >= self.min_range)
        goal = dark + self.target_at(t_min) * (lit - dark)
        # lit while the reading is on the dark side of the goal
        on = usable & ((reading - goal) * (lit - dark) < 0)
        return regions.paint(on.astype(np.float32), dtype=np.float32)
