# Stage 1 spike (2026-09-06). Not part of the package; run from the repo root with
#     uv run python docs/spikes/stage1_useq_spike.py
# Findings are written up in docs/stage1-plan-design.md.
"""Probe useq-schema features relevant to expressing a PyCLM schedule."""

import inspect
from importlib.metadata import version

import useq

print("useq-schema", version("useq-schema"))


def fields(cls):
    return {
        k: str(v.annotation).replace("typing.", "")[:60]
        for k, v in cls.model_fields.items()
    }


for cls in (useq.Channel, useq.Position, useq.MDAEvent, useq.MDASequence):
    print(f"\n== {cls.__name__} ==")
    for k, v in fields(cls).items():
        print(f"  {k}: {v}")

print("\n== public names with 'Action' / 'AF' / 'Time' / 'Plan' ==")
names = [
    n
    for n in dir(useq)
    if any(s in n for s in ("Action", "AF", "Time", "Plan", "Autofocus", "Custom"))
]
print(" ", names)

for name in (
    "CustomAction",
    "AxesBasedAF",
    "AutoFocusPlan",
    "TIntervalLoops",
    "MultiPhaseTimePlan",
):
    cls = getattr(useq, name, None)
    if cls is None:
        print(f"\n{name}: NOT PRESENT")
        continue
    print(f"\n== {name} ==")
    if hasattr(cls, "model_fields"):
        for k, v in fields(cls).items():
            print(f"  {k}: {v}")

print("\n== MDASequence methods of interest ==")
for m in (
    "iter_events",
    "estimate_duration",
    "sizes",
    "shape",
    "used_axes",
    "yaml",
    "replace",
):
    print(f"  {m}: {'yes' if hasattr(useq.MDASequence, m) else 'no'}")

# ---- (a) per-channel cadence via acquire_every -----------------------------
print("\n== (a) Channel.acquire_every = 2 on one of two channels, 4 timepoints ==")
seq = useq.MDASequence(
    time_plan={"interval": 60, "loops": 4},
    channels=[
        {"config": "DMD", "group": "Channel", "exposure": 500},
        {"config": "545", "group": "Channel", "exposure": 50, "acquire_every": 2},
    ],
)
for e in seq:
    print("  ", dict(e.index), e.channel.config, e.min_start_time)

# ---- (b) per-position sub-sequence with a shorter time plan ----------------
print(
    "\n== (b) position sub-sequence with its own time plan (2 loops) under a 4-loop parent =="
)
seq = useq.MDASequence(
    time_plan={"interval": 60, "loops": 4},
    channels=["545"],
    stage_positions=[
        useq.Position(x=0, y=0, name="a"),
        useq.Position(
            x=1,
            y=0,
            name="b",
            sequence=useq.MDASequence(time_plan={"interval": 60, "loops": 2}),
        ),
    ],
)
for e in seq:
    print("  ", dict(e.index), e.pos_name, e.min_start_time)

# ---- (c) multi-phase time plan -------------------------------------------
print("\n== (c) MultiPhaseTimePlan: 2 loops @60s then 2 loops @30s ==")
seq = useq.MDASequence(
    time_plan=[{"interval": 60, "loops": 2}, {"interval": 30, "loops": 2}],
    channels=["545"],
)
for e in seq:
    print("  ", dict(e.index), e.min_start_time)

# ---- (d) autofocus plan carrying a motor offset per position --------------
print("\n== (d) per-position autofocus plan (PFS offset) ==")
try:
    seq = useq.MDASequence(
        time_plan={"interval": 60, "loops": 1},
        channels=["545"],
        stage_positions=[
            useq.Position(
                x=0,
                y=0,
                z=5,
                name="a",
                sequence=useq.MDASequence(
                    autofocus_plan=useq.AxesBasedAF(
                        autofocus_device_name="PFS",
                        autofocus_motor_offset=11122.0,
                        axes=("p",),
                    )
                ),
            )
        ],
    )
    for e in seq:
        print(
            "  ",
            dict(e.index),
            e.pos_name,
            "action=",
            type(e.action).__name__,
            getattr(e.action, "autofocus_motor_offset", None),
        )
except Exception as ex:
    print("  autofocus plan failed:", type(ex).__name__, ex)

# ---- (e) custom action event ---------------------------------------------
print("\n== (e) CustomAction event constructed by hand ==")
try:
    ev = useq.MDEvent = useq.MDAEvent(
        index={"t": 3, "p": 0},
        pos_name="a",
        action=useq.CustomAction(name="update_pattern", data={"experiment": "a"}),
        metadata={"pyclm": {"kind": "update_pattern"}},
    )
    print("  ", ev.action, ev.metadata)
except Exception as ex:
    print("  custom action failed:", type(ex).__name__, ex)

# ---- (f) duration estimate -------------------------------------------------
print("\n== (f) estimate_duration ==")
seq = useq.MDASequence(
    time_plan={"interval": 60, "loops": 3},
    channels=[{"config": "DMD", "exposure": 500}, {"config": "545", "exposure": 50}],
    stage_positions=[useq.Position(x=0, y=0), useq.Position(x=100, y=0)],
)
try:
    est = seq.estimate_duration()
    print("  ", est)
    print("  fields:", [k for k in dir(est) if not k.startswith("_")][:12])
except Exception as ex:
    print("  estimate failed:", type(ex).__name__, ex)

# ---- (g) serialization round trip ----------------------------------------
print("\n== (g) json/yaml round trip ==")
js = seq.model_dump_json(exclude_defaults=True)
back = useq.MDASequence.model_validate_json(js)
print("  json bytes:", len(js), "equal after round trip:", back == seq)
print("  yaml head:", seq.yaml().splitlines()[:6])
print("  sizes:", dict(seq.sizes))
print("\n== iter_events signature ==")
print(" ", inspect.signature(useq.MDASequence.iter_events))
