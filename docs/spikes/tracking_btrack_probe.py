"""btrack probe for docs/tracking-gap-closing-design.md: frame-by-frame append + track. Run from docs/spikes/."""

import logging
import sys
import time

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
import btrack
from btrack import datasets
from tracking_gap_closing_bench import simulate

n = int(sys.argv[1])
frames = simulate(n, T=30)
cfg = datasets.cell_config()
objs = []
dt = []
with btrack.BayesianTracker(verbose=False) as tr:
    tr.configure(cfg)
    tr.max_search_radius = 15
    tr.volume = ((0, 2100), (0, 2100))
    k = 0
    for t, (c, _g) in enumerate(frames):
        arr = np.zeros(
            len(c),
            dtype=[
                ("ID", "i4"),
                ("t", "i4"),
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("label", "i4"),
                ("prob", "f4"),
                ("dummy", "?"),
                ("states", "i4"),
            ],
        )
        t0 = time.perf_counter()
        objs = [
            btrack.btypes.PyTrackObject.from_dict(
                {
                    "ID": k + i,
                    "t": t,
                    "x": float(p[1]),
                    "y": float(p[0]),
                    "z": 0.0,
                    "label": 0,
                    "prob": 0.0,
                    "dummy": False,
                    "states": 0,
                }
            )
            for i, p in enumerate(c)
        ]
        k += len(c)
        tr.append(objs)
        tr.track(step_size=1)
        dt.append(time.perf_counter() - t0)
    print("tracks", tr.n_tracks)
dt = np.array(dt[1:]) * 1e3
print(
    f"btrack n={n} append+track per frame: median {np.median(dt):.1f} ms p95 {np.percentile(dt, 95):.1f} ms"
)
