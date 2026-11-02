"""
Spike for docs/tracking-gap-closing-design.md (2026-10-28).

Run in a scratch environment with trackpy, laptrack, scipy, scikit-image:
    python tracking_gap_closing_bench.py N_OBJECTS STEP_PX [centroid,gaplap,trackpy,laptrack]

Per-frame latency and accuracy of online linkers with gap closing, on
synthetic nuclei: random walk + drift, a fraction missed per frame,
births and deaths. Only the linking step is timed (regionprops is common
to all and measured separately).
"""

import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

RNG = np.random.default_rng(0)


def simulate(n, T=60, size=2048, step=3.0, p_miss=0.05, p_miss2=0.02, p_birth=0.005):
    """Ground truth ids per frame; detections drop out for 1 or 2 frames."""
    pos = RNG.uniform(0, size, (n, 2))
    gid = np.arange(n)
    nxt = n
    frames = []
    miss_left = np.zeros(n, int)
    for _t in range(T):
        pos = pos + RNG.normal(0, step, pos.shape) + np.array([0.5, 0.3])
        # births
        nb = RNG.binomial(n, p_birth)
        if nb:
            pos = np.vstack([pos, RNG.uniform(0, size, (nb, 2))])
            gid = np.concatenate([gid, np.arange(nxt, nxt + nb)])
            miss_left = np.concatenate([miss_left, np.zeros(nb, int)])
            nxt += nb
        start = (miss_left == 0) & (RNG.random(len(gid)) < p_miss)
        miss_left[start] = 1 + (RNG.random(start.sum()) < p_miss2 / p_miss)
        seen = miss_left == 0
        miss_left = np.maximum(miss_left - 1, 0) * (~seen) + 0
        perm = RNG.permutation(seen.sum())
        frames.append((pos[seen][perm].copy(), gid[seen][perm].copy()))
    return frames


# --- linkers -------------------------------------------------------------


class Centroid:
    """The current built-in: dense Hungarian, previous frame only."""

    def __init__(self, gate, **_):
        self.gate = gate
        self.prev = np.zeros((0, 2))
        self.ids = np.zeros(0, int)
        self.n = 1

    def __call__(self, c, t):
        ids = np.zeros(len(c), int)
        if len(self.prev) and len(c):
            d = cdist(self.prev, c)
            r, k = linear_sum_assignment(d)
            ok = d[r, k] <= self.gate
            ids[k[ok]] = self.ids[r[ok]]
        new = ids == 0
        ids[new] = np.arange(self.n, self.n + new.sum())
        self.n += new.sum()
        self.prev, self.ids = c, ids
        return ids


class GapLAP:
    """
    Own implementation: active + dormant tracks (unseen for <= max_gap
    frames) vs detections, gated with a KD-tree, split into connected
    components, each solved as a LAP with 'no link' alternatives at the
    gate cost (Jaqaman's augmented matrix). Dormant candidates pay
    gap_penalty per missed frame so an active track wins a tie.
    """

    def __init__(self, gate, max_gap=2, gap_penalty=None, **_):
        self.gate = gate
        self.max_gap = max_gap
        self.pen = (
            0.0 if gap_penalty is None else gap_penalty
        )  # the 0.1-gate rows of the note: gate * 0.1
        self.pos = np.zeros((0, 2))
        self.ids = np.zeros(0, int)
        self.last = np.zeros(0, int)
        self.n = 1

    def __call__(self, c, t):
        ids = np.zeros(len(c), int)
        if len(self.pos) and len(c):
            gap = t - self.last - 1  # 0 for active
            tree = cKDTree(c)
            pairs = tree.query_ball_point(self.pos, self.gate)
            ri = np.repeat(np.arange(len(self.pos)), [len(p) for p in pairs])
            ci = (
                np.concatenate([np.asarray(p, int) for p in pairs])
                if len(ri)
                else np.zeros(0, int)
            )
            if len(ri):
                cost = np.linalg.norm(self.pos[ri] - c[ci], axis=1) + self.pen * gap[ri]
                keep = cost <= self.gate
                ri, ci, cost = ri[keep], ci[keep], cost[keep]
            if len(ri):
                nr, nc = len(self.pos), len(c)
                g = coo_matrix((np.ones(len(ri)), (ri, nr + ci)), shape=(nr + nc,) * 2)
                _ncomp, lab = connected_components(g, directed=False)
                comp_r = lab[ri]
                order = np.argsort(comp_r, kind="stable")
                ri, ci, cost, comp_r = ri[order], ci[order], cost[order], comp_r[order]
                bounds = np.flatnonzero(np.diff(comp_r)) + 1
                for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(ri)], strict=False):
                    r_, c_, w_ = ri[s:e], ci[s:e], cost[s:e]
                    ur, rl = np.unique(r_, return_inverse=True)
                    uc, cl = np.unique(c_, return_inverse=True)
                    if len(ur) == 1 or len(uc) == 1:
                        j = np.argmin(w_)
                        ids[c_[j]] = self.ids[r_[j]]
                        continue
                    a, b = len(ur), len(uc)
                    big = 1e9
                    m = np.full((a + b, b + a), big)
                    m[rl, cl] = w_
                    m[np.arange(a), b + np.arange(a)] = self.gate
                    m[a + np.arange(b), np.arange(b)] = self.gate
                    m[a:, b:] = 0
                    rr, cc = linear_sum_assignment(m)
                    link = (rr < a) & (cc < b)
                    ids[uc[cc[link]]] = self.ids[ur[rr[link]]]
        new = ids == 0
        ids[new] = np.arange(self.n, self.n + new.sum())
        self.n += new.sum()
        # update memory: seen tracks move to their new position
        seen = np.isin(self.ids, ids)
        alive = ~seen & (t - self.last <= self.max_gap)
        self.pos = np.vstack([self.pos[alive], c])
        self.ids = np.concatenate([self.ids[alive], ids])
        self.last = np.concatenate([self.last[alive], np.full(len(c), t)])
        return ids


class Trackpy:
    def __init__(self, gate, max_gap=2, **_):
        from trackpy.linking import Linker

        self.lk = Linker(gate, memory=max_gap)
        self.started = False

    def __call__(self, c, t):
        if not self.started:
            self.lk.init_level(c, t)
            self.started = True
        else:
            self.lk.next_level(c, t)
        return np.asarray(self.lk.particle_ids) + 1


class LapWindow:
    """laptrack re-run on a sliding window of the last W frames (it is offline)."""

    def __init__(self, gate, max_gap=2, window=6, **_):
        from laptrack import LapTrack

        self.lt = LapTrack(
            cutoff=gate**2,
            gap_closing_cutoff=gate**2,
            gap_closing_max_frame_count=max_gap,
            splitting_cutoff=False,
            merging_cutoff=False,
        )
        self.buf = []
        self.W = window

    def __call__(self, c, t):
        self.buf.append(c)
        self.buf = self.buf[-self.W :]
        tree = self.lt.predict(self.buf)
        # ids of the last frame (not carried across windows: timing only)
        return np.zeros(len(c), int) + 1


def score(frames, out):
    """Fraction of ground-truth links (incl. across gaps) with the same output id."""
    last_seen = {}
    ok = tot = okgap = totgap = 0
    switches = 0
    owner = {}
    for t, ((_c, gid), ids) in enumerate(zip(frames, out, strict=False)):
        for g, i in zip(gid, ids, strict=False):
            if g in last_seen:
                tp, ip = last_seen[g]
                same = ip == i
                if t - tp == 1:
                    tot += 1
                    ok += same
                else:
                    totgap += 1
                    okgap += same
            if i in owner and owner[i] != g:
                switches += 1
            owner[i] = g
            last_seen[g] = (t, i)
    return ok / max(tot, 1), okgap / max(totgap, 1), switches


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    which = (
        sys.argv[3].split(",")
        if len(sys.argv) > 3
        else ["centroid", "gaplap", "trackpy", "laptrack"]
    )
    frames = simulate(n, step=step)
    gate = 5 * step
    cls = {
        "centroid": Centroid,
        "gaplap": GapLAP,
        "trackpy": Trackpy,
        "laptrack": LapWindow,
    }
    for name in which:
        lk = cls[name](gate, max_gap=2)
        out, dt = [], []
        try:
            for t, (c, _) in enumerate(frames):
                t0 = time.perf_counter()
                out.append(lk(c, t))
                dt.append(time.perf_counter() - t0)
        except Exception as e:
            print(
                f"{name:9s} n={n} step={step}: FAILED at t={t}: {type(e).__name__}: {e}"
            )
            continue
        dt = np.array(dt[1:]) * 1e3
        a1, ag, sw = score(frames, out) if name != "laptrack" else (np.nan, np.nan, 0)
        print(
            f"{name:9s} n={n:5d} step={step:4.1f}  median {np.median(dt):7.2f} ms  "
            f"p95 {np.percentile(dt, 95):7.2f} ms  | links kept {a1:.3f}  gaps bridged {ag:.3f}  id reuse {sw}"
        )
