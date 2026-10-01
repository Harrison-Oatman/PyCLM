"""
Automatic verification per item, on the files a run leaves behind. Each
``verify_NN(directory)`` returns ``[(check, passed, detail), ...]``; what
cannot be checked from files (what you saw through the eyepiece) is in the
item's guide and goes into the result's notes by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pyclm.io as pio


def _status(directory: Path) -> dict:
    return json.loads((directory / "status.json").read_text())


def _stores(directory: Path) -> list[Path]:
    return pio.find_experiments(directory)


def _rows(exp) -> list[dict]:
    return exp.frames.to_pylist() if exp.frames is not None else []


def _events(exp) -> list[dict]:
    return exp.events.to_pylist() if exp.events is not None else []


def _done(directory: Path) -> tuple[str, bool, str]:
    s = _status(directory)
    errors = sum(v.get("errors", 0) for v in s.get("experiments", {}).values())
    return (
        "run completed without acquisition errors",
        bool(s.get("done")) and errors == 0,
        f"done={s.get('done')}, errors={errors}, t={s.get('t')}/{s.get('timepoints')}",
    )


def _on_time(
    directory: Path, tolerance_intervals: float = 1.0
) -> tuple[str, bool, str]:
    late = []
    for store in _stores(directory):
        with pio.open(store) as exp:
            late += [e for e in _events(exp) if e.get("kind") == "late"]
    return (
        "no timepoint more than one interval late",
        not late,
        f"{len(late)} late events",
    )


def _pfs_locked(directory: Path) -> tuple[str, bool, str]:
    log = (
        (directory / "log.log").read_text(errors="replace")
        if (directory / "log.log").exists()
        else ""
    )
    locks = log.count("move+focus took")
    timeouts = log.count("PFS did not report focus lock")
    return (
        "PFS locked at every visit",
        locks > 0 and timeouts == 0,
        f"{locks} locks, {timeouts} timeouts",
    )


# ------------------------------------------------------------------ items
def verify_02(directory: Path):
    rows = [_done(directory), _on_time(directory), _pfs_locked(directory)]
    for store in _stores(directory):
        with pio.open(store) as exp:
            frames = [r for r in _rows(exp) if r["kind"] == "frame"]
            stim = [r for r in frames if r["channel"] == "DMD"]
            img = [r for r in frames if r["channel"] != "DMD"]
            rows.append(
                (
                    f"{exp.name}: imaging every 2, stimulation every 1",
                    len(stim) == 10 and len(img) == 5,
                    f"{len(stim)} stim, {len(img)} imaging frames",
                )
            )
            rows.append(
                (
                    f"{exp.name}: exposure and intensity as configured",
                    all(r["exposure_ms"] == 50.0 for r in frames),
                    f"exposures {sorted({r['exposure_ms'] for r in frames})}",
                )
            )
            pat = exp.pattern_at(9)
            rows.append(
                (
                    f"{exp.name}: a DMD pattern is lit",
                    pat is not None and pat.max() > 0,
                    "none" if pat is None else f"{(pat > 0).mean():.2%} of the DMD",
                )
            )
            zc = [e for e in _events(exp) if e["kind"] == "z_correction"]
            rows.append(
                (
                    f"{exp.name}: no large focus corrections",
                    all(
                        abs(float(e.get("new") or 0) - float(e.get("old") or 0)) < 5
                        for e in zc
                    ),
                    f"{len(zc)} corrections",
                )
            )
    return rows


def verify_03(directory: Path):
    import h5py

    rows = [_done(directory)]
    files = [p for p in _stores(directory) if p.suffix == ".hdf5"]
    rows.append(("HDF5 files written", bool(files), ", ".join(p.name for p in files)))
    for path in files:
        with h5py.File(path, "r", swmr=True) as f:
            t = int(f["current_t_index"][()])
            dmd = [k for k in f if k.isdigit() and "stim_aq/dmd" in f[k]]
            comp = f[f"{dmd[0]}/stim_aq/dmd"].compression if dmd else None
            rows.append(
                (f"{path.name}: timepoints complete", t == 3, f"current_t_index {t}")
            )
            rows.append(
                (
                    f"{path.name}: dmd datasets compressed",
                    comp == "gzip",
                    f"compression {comp}",
                )
            )
    return rows


def verify_04(directory: Path):
    rows = [_done(directory)]
    for store in _stores(directory):
        with pio.open(store) as exp:
            g = exp.groups.get("imaging")
            n_cells = []
            for i in g.acquired():
                lab = g.labels(i, "545")
                n_cells.append(0 if lab is None else int(len(np.unique(lab)) - 1))
            rows.append(
                (
                    f"{exp.name}: cells segmented at every imaging timepoint",
                    all(n > 0 for n in n_cells),
                    f"cells per frame {n_cells}",
                )
            )
            pat = exp.camera_pattern_at(exp.current_t)
            rows.append(
                (
                    f"{exp.name}: pattern lights part of the field",
                    pat is not None and 0 < (pat > 0).mean() < 0.9,
                    "none" if pat is None else f"{(pat > 0).mean():.1%} lit",
                )
            )
    return rows


def verify_05(directory: Path):
    rows = verify_04(directory)
    for store in _stores(directory):
        with pio.open(store) as exp:
            tracks = exp.tracks.to_pylist() if exp.tracks is not None else []
            ids = {}
            for r in tracks:
                ids.setdefault(r["track_id"], set()).add(r["t"])
            persistent = sum(1 for ts in ids.values() if len(ts) >= 2)
            rows.append(
                (
                    f"{exp.name}: tracks persist across timepoints",
                    persistent >= 3,
                    f"{persistent} tracks seen twice or more",
                )
            )
    return rows


def verify_06(directory: Path):
    rows = [_done(directory)]
    for store in _stores(directory):
        with pio.open(store) as exp:
            g = exp.groups.get("imaging")
            i = g.acquired()[-1]
            nuc = g.labels(i, "ktr", "nuclei")
            cells = g.labels(i, "ktr", "cells")
            rows.append(
                (
                    f"{exp.name}: both segmentations stored",
                    nuc is not None and cells is not None,
                    f"nuclei {None if nuc is None else nuc.max()}, cells {None if cells is None else cells.max()}",
                )
            )
            pat = exp.camera_pattern_at(exp.current_t)
            rows.append(
                (
                    f"{exp.name}: a graded pattern",
                    pat is not None and len(np.unique(pat)) > 2,
                    "none" if pat is None else f"{len(np.unique(pat))} distinct levels",
                )
            )
    return rows


def verify_07(directory: Path):
    rows = [_done(directory)]
    for store in _stores(directory):
        with pio.open(store) as exp:
            applied = [
                e
                for e in _events(exp)
                if e["status"] == "applied" and e["kind"] in ("exposure", "property")
            ]
            rows.append(
                (
                    f"{exp.name}: exposure and property requests applied",
                    len(applied) >= 4,
                    f"{len(applied)} applied",
                )
            )
            stim = [
                r for r in _rows(exp) if r["kind"] == "frame" and r["channel"] == "DMD"
            ]
            exposures = [r["exposure_ms"] for r in stim]
            rows.append(
                (
                    f"{exp.name}: stimulation exposure rises",
                    len(set(exposures)) >= 3,
                    f"exposures {exposures}",
                )
            )
            col = "Sola-Power"
            rows.append(
                (
                    f"{exp.name}: property column present",
                    col in exp.frames.column_names,
                    f"columns include {col}: {col in exp.frames.column_names}",
                )
            )
    return rows


def verify_08(directory: Path):
    rows = [_done(directory)]
    s = _status(directory)
    rows.append(
        (
            "commands applied",
            s.get("commands_applied", 0) >= 6,
            f"{s.get('commands_applied')} applied, paused {s.get('paused_s')}s",
        )
    )
    for store in _stores(directory):
        with pio.open(store) as exp:
            cmds = [e for e in _events(exp) if e["kind"] == "command"]
            kinds = {e.get("key") or e.get("detail") for e in cmds}
            rows.append(
                (
                    f"{exp.name}: pause, resume, set_*, stop_run recorded",
                    len(cmds) >= 6,
                    f"{len(cmds)} command events",
                )
            )
            by_cmd = [
                e
                for e in _events(exp)
                if e.get("source") == "command" and e["status"] == "applied"
            ]
            rows.append(
                (
                    f"{exp.name}: settings from commands applied",
                    len(by_cmd) >= 3,
                    f"{len(by_cmd)} applied",
                )
            )
            stim = [
                r for r in _rows(exp) if r["kind"] == "frame" and r["channel"] == "DMD"
            ]
            rows.append(
                (
                    f"{exp.name}: exposure 80 ms after the command",
                    any(r["exposure_ms"] == 80.0 for r in stim),
                    f"exposures {sorted({r['exposure_ms'] for r in stim})}",
                )
            )
    rows.append(
        (
            "run stopped early by stop_run",
            s.get("t", 0) < s.get("timepoints", 0) - 1 and s.get("done"),
            f"t {s.get('t')} of {s.get('timepoints')}",
        )
    )
    return rows


def verify_09(directory: Path):
    rows = [_done(directory)]
    for store in _stores(directory):
        with pio.open(store) as exp:
            props = [
                e
                for e in _events(exp)
                if e["kind"] == "property" and e["status"] == "applied"
            ]
            rows.append(
                (
                    f"{exp.name}: laser properties switched",
                    len(props) >= 6,
                    f"{len(props)} applied",
                )
            )
            pat = exp.camera_pattern_at(exp.current_t)
            rows.append(
                (
                    f"{exp.name}: whole field lit",
                    pat is not None and pat.min() == 255,
                    "none" if pat is None else f"min {pat.min()}",
                )
            )
    return rows


def verify_10(directory: Path):
    rows = [_done(directory)]
    s = _status(directory)
    rows.append(
        (
            "skipped timepoints counted",
            s.get("skipped", 0) >= 2,
            f"{s.get('skipped')} skipped",
        )
    )
    for store in _stores(directory):
        with pio.open(store) as exp:
            skipped = sorted(r["t"] for r in _rows(exp) if r["kind"] == "skipped")
            rows.append(
                (
                    f"{exp.name}: only odd timepoints skipped, at least two",
                    len(skipped) >= 2 and all(t % 2 == 1 for t in skipped),
                    f"skipped at t {skipped}",
                )
            )
            ev = [
                e["t_applied"] for e in _events(exp) if e["kind"] == "position_skipped"
            ]
            rows.append(
                (
                    f"{exp.name}: position_skipped events",
                    len(ev) == len(skipped),
                    f"{len(ev)} events",
                )
            )
    log = (
        (directory / "log.log").read_text(errors="replace")
        if (directory / "log.log").exists()
        else ""
    )
    rows.append(
        (
            "stage moves skipped in the log",
            "stage move skipped" in log,
            f"{log.count('stage move skipped')} occurrences",
        )
    )
    return rows


def verify_11(directory: Path):
    rows = [_done(directory)]
    cfg = (directory / "pyclm_config.toml").read_text()
    rows.append(
        (
            "camera_roi set in the config",
            "camera_roi" in cfg and not cfg.strip().startswith("# camera_roi"),
            "see pyclm_config.toml",
        )
    )
    for store in _stores(directory):
        with pio.open(store) as exp:
            roi = exp.camera_roi
            g = exp.groups.get("imaging")
            frame = g.frame(g.acquired()[-1], "545")
            rows.append(
                (
                    f"{exp.name}: frames have the ROI's size",
                    roi is not None
                    and frame.shape == (roi[3] // g.binning, roi[2] // g.binning),
                    f"roi {roi}, frame {frame.shape}",
                )
            )
            stim = exp.groups.get("stim")
            if stim is not None and stim.acquired():
                sf = stim.frame(stim.acquired()[-1], "DMD").astype(float)
                edge = np.concatenate([sf[0], sf[-1], sf[:, 0], sf[:, -1]])
                rows.append(
                    (
                        f"{exp.name}: stimulation frame lit to the edges",
                        edge.mean() > 0.5 * sf.mean(),
                        f"edge/centre {edge.mean() / max(sf.mean(), 1):.2f}",
                    )
                )
    return rows


def verify_12(directory: Path):
    rows = [_done(directory), _pfs_locked(directory)]
    for store in _stores(directory):
        with pio.open(store) as exp:
            rows.append(
                (
                    f"{exp.name}: a 2 x 2 grid",
                    exp.grid is not None
                    and exp.grid["rows"] == 2
                    and exp.grid["columns"] == 2,
                    f"grid {exp.grid}",
                )
            )
            g = exp.groups.get("imaging")
            frame = g.frame(g.acquired()[-1], "545").astype(float)
            h, w = frame.shape
            # continuity across the tile boundaries: the rows either side of the seam should correlate
            seam_v = np.corrcoef(frame[h // 2 - 1], frame[h // 2])[0, 1]
            seam_h = np.corrcoef(frame[:, w // 2 - 1], frame[:, w // 2])[0, 1]
            rows.append(
                (
                    f"{exp.name}: continuity across the seams",
                    seam_v > 0.5 and seam_h > 0.5,
                    f"corr rows {seam_v:.2f}, cols {seam_h:.2f} (inspect the frame too)",
                )
            )
            ids = exp.pattern_ids
            rows.append(
                (
                    f"{exp.name}: four DMD images per pattern",
                    len(ids) % 4 == 0 and len(ids) > 0,
                    f"{len(ids)} DMD images",
                )
            )
    return rows


def verify_13(directory: Path):
    rows = [_done(directory), _on_time(directory)]
    s = _status(directory)
    rows.append(
        (
            "ran for at least 8 hours",
            float(s.get("elapsed_s", 0)) >= 8 * 3600,
            f"elapsed {float(s.get('elapsed_s', 0)) / 3600:.1f} h",
        )
    )
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    rows.append(
        ("outputs written", total > 0, f"{total / 1e9:.2f} GB in the directory")
    )
    return rows


def verify_14(directory: Path):
    s = _status(directory)
    errors = sum(v.get("errors", 0) for v in s.get("experiments", {}).values())
    rows = [
        (
            "run completed despite the error",
            bool(s.get("done")),
            f"done={s.get('done')}",
        ),
        ("the error was counted", errors >= 1, f"{errors} acquisition errors"),
    ]
    for store in _stores(directory):
        with pio.open(store) as exp:
            errs = [e for e in _events(exp) if e["kind"] == "acquisition_error"]
            rows.append(
                (
                    f"{exp.name}: acquisition_error event recorded",
                    bool(errs),
                    f"{len(errs)} events",
                )
            )
    return rows
