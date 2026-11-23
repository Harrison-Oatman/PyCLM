"""
``pyclm pack``: copy a finished run somewhere else as a few large files, with
a manifest that proves the copy is complete.

An OME-Zarr store is thousands of small chunk files, and copying a run file by
file to a network share, a cloud folder or a USB drive is slow and easy to
interrupt halfway. ``pack`` writes each top-level folder of the run (each
``<experiment>.zarr``, ...) as one uncompressed ``.zip``, copies the remaining
files as they are, writes ``manifest.json`` (every original file, its size and
the archive that holds it) and checks the copy against it. ``movies/`` is
always copied loose, so the movies can be watched where the copy lands.

::

    pyclm pack <dir> <destination>        # -> <destination>/<dir name>/
    pyclm pack --verify <copy>            # check a copy against its manifest
    pyclm pack --unpack <copy> <dest>     # unpack a packed copy, then check it

Nothing here deletes anything: keep the original until the copy is verified
(and, where it matters, backed up). A zip can be opened by any operating
system, and ``zarr`` can read a store straight from its zip.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import time
import zipfile
from datetime import datetime
from pathlib import Path

MANIFEST = "manifest.json"
SKIP_DIRS = {"preview"}  # outputs of pyclm preview, made from the run's files
LOOSE_DIRS = {"movies"}  # copied file by file even when packing
QUIET_MIN = 10.0  # no file in the run may have changed this recently


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def run_files(src: Path, skip_exports: bool = True) -> tuple[list[Path], list[str]]:
    """Files to copy (relative paths), and what is skipped with the reason."""
    from .io import find_experiments

    export_prefixes = (
        {p.name.rsplit(".", 1)[0] + "_" for p in find_experiments(src)}
        if skip_exports
        else set()
    )
    keep, skipped = [], []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if rel.parts[0] in SKIP_DIRS:
            skipped.append(f"{rel.as_posix()} (preview output)")
        elif "__pycache__" in rel.parts:
            continue
        elif (
            len(rel.parts) == 1
            and p.suffix == ".tif"
            and any(p.name.startswith(pre) for pre in export_prefixes)
        ):
            skipped.append(f"{rel.as_posix()} (ImageJ export)")
        elif rel.name == MANIFEST or p.name.endswith(".part"):
            continue
        else:
            keep.append(rel)
    return keep, skipped


def not_finished(src: Path, quiet_min: float = QUIET_MIN) -> list[str]:
    """Reasons not to copy yet: the run is not done, or its files are still changing."""
    problems = []
    status = src / "status.json"
    if not status.exists():
        problems.append("no status.json: is this a PyCLM experiment directory?")
    else:
        try:
            st = json.loads(status.read_text())
        except ValueError:
            st = {}
        if not st.get("done"):
            problems.append(
                f"status.json does not say done (t {st.get('t')}/{st.get('timepoints')})"
            )
    newest = max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()), default=0)
    age = (time.time() - newest) / 60
    if age < quiet_min:
        problems.append(
            f"a file changed {age:.1f} min ago; the writer or the ImageJ export may still be running"
        )
    return problems


def _copy(src: Path, dst: Path, hashed: bool) -> str | None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    h = hashlib.sha256() if hashed else None
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while b := fi.read(1 << 22):
            if h is not None:
                h.update(b)
            fo.write(b)
    os.replace(tmp, dst)
    shutil.copystat(src, dst)
    return h.hexdigest() if h is not None else None


def _zip(dst_zip: Path, rels: list[Path], base: Path) -> None:
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_zip.with_name(dst_zip.name + ".part")
    with zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as z:
        for rel in rels:
            z.write(base / rel, rel.as_posix())
    os.replace(tmp, dst_zip)


def _unlisted(folder: Path, listed: set[str]) -> list[str]:
    return [
        f"not in manifest: {p.relative_to(folder).as_posix()}"
        for p in folder.rglob("*")
        if p.is_file()
        and p.name != MANIFEST
        and p.relative_to(folder).as_posix() not in listed
    ]


def _check_loose(folder: Path, entries: list[dict], hashes: bool) -> list[str]:
    problems = []
    for e in entries:
        p = folder / e["path"]
        if not p.exists():
            problems.append(f"missing: {e['path']}")
        elif p.stat().st_size != e["size"]:
            problems.append(
                f"size differs: {e['path']} ({p.stat().st_size} vs {e['size']})"
            )
        elif hashes and e.get("sha256") and sha256(p) != e["sha256"]:
            problems.append(f"checksum differs: {e['path']}")
    return problems


def verify(folder: Path, hashes: bool = False) -> list[str]:
    """
    Problems with a copy against its manifest: missing files, wrong sizes,
    unlisted files, and (with ``hashes``) wrong checksums. A packed copy is
    checked archive by archive: each zip's size and index against the manifest
    (``hashes`` also tests every member's CRC). An unpacked copy is checked
    file by file. Empty means the copy is good.
    """
    folder = Path(folder)
    mpath = folder / MANIFEST
    if not mpath.exists():
        return [f"no {MANIFEST} in {folder}"]
    man = json.loads(mpath.read_text())
    archives = [a["path"] for a in man.get("archives", [])]
    packed = bool(archives) and all((folder / a).exists() for a in archives)
    if not packed:
        problems = _check_loose(folder, man["files"], hashes)
        return problems + _unlisted(folder, {e["path"] for e in man["files"]})
    problems = []
    members: dict[str, dict[str, int]] = {}
    for e in man["files"]:
        if e.get("archive"):
            members.setdefault(e["archive"], {})[e["path"]] = e["size"]
    for a in man["archives"]:
        p = folder / a["path"]
        if p.stat().st_size != a["size"]:
            problems.append(
                f"size differs: {a['path']} ({p.stat().st_size} vs {a['size']})"
            )
            continue
        with zipfile.ZipFile(p) as z:
            inside = {i.filename: i.file_size for i in z.infolist()}
            if hashes and (bad := z.testzip()) is not None:
                problems.append(f"checksum differs: {a['path']}:{bad}")
        want = members.get(a["path"], {})
        problems += [f"missing in {a['path']}: {k}" for k in want if k not in inside]
        problems += [
            f"size differs in {a['path']}: {k}"
            for k in want
            if k in inside and inside[k] != want[k]
        ]
        problems += [
            f"not in manifest: {a['path']}:{k}" for k in inside if k not in want
        ]
    loose = [e for e in man["files"] if not e.get("archive")]
    problems += _check_loose(folder, loose, hashes)
    return problems + _unlisted(folder, set(archives) | {e["path"] for e in loose})


def pack(
    src: Path,
    dest_root: Path,
    *,
    zip_folders: bool = True,
    checksums: bool = False,
    force: bool = False,
    skip_exports: bool = True,
    quiet_min: float = QUIET_MIN,
    log=print,
) -> int:
    """Copy ``src`` into ``dest_root/<src name>``; returns 0 (verified), 1 (copy bad) or 2 (not started)."""
    src = Path(src).resolve()
    problems = not_finished(src, quiet_min)
    if problems and not force:
        log("not copying (add --force to copy anyway):\n  " + "\n  ".join(problems))
        return 2
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    dst = dest_root.resolve() / src.name
    if dst.exists() and any(dst.iterdir()):
        log(f"{dst} already exists and is not empty; check it with pyclm pack --verify")
        return 2
    files, skipped = run_files(src, skip_exports)
    total = sum((src / f).stat().st_size for f in files)
    free = shutil.disk_usage(dest_root).free
    if free < total * 1.02:
        log(f"not enough space: need {total / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
        return 2
    entries, archives, done, t0 = [], [], 0, time.time()
    loose = files
    if zip_folders:
        groups: dict[str, list[Path]] = {}
        loose = []
        for rel in files:
            inside = len(rel.parts) > 1 and rel.parts[0] not in LOOSE_DIRS
            (groups.setdefault(rel.parts[0], []) if inside else loose).append(rel)
        log(
            f"packing {len(files) - len(loose)} files into {len(groups)} archive(s) and "
            f"copying {len(loose)} loose, {total / 1e9:.2f} GB, to {dst}"
        )
        for k, (top, rels) in enumerate(sorted(groups.items()), 1):
            name = f"{top}.zip"
            _zip(dst / name, rels, src)
            for rel in rels:
                size = (src / rel).stat().st_size
                entries.append(
                    {
                        "path": rel.as_posix(),
                        "size": size,
                        "sha256": sha256(src / rel) if checksums else None,
                        "archive": name,
                    }
                )
                done += size
            archives.append({"path": name, "size": (dst / name).stat().st_size})
            rate = done / max(time.time() - t0, 1e-6) / 1e6
            log(f"  {k}/{len(groups)} archives, {done / 1e9:.2f} GB, {rate:.0f} MB/s")
    else:
        log(f"copying {len(files)} files, {total / 1e9:.2f} GB, to {dst}")
    for rel in loose:
        h = _copy(src / rel, dst / rel, checksums)
        size = (src / rel).stat().st_size
        entries.append({"path": rel.as_posix(), "size": size, "sha256": h})
        done += size
    manifest = {
        "run": src.name,
        "source": str(src),
        "source_host": socket.gethostname(),
        "copied_at": datetime.now().isoformat(timespec="seconds"),
        "total_bytes": total,
        "skipped": skipped,
        "files": entries,
        **({"archives": archives} if zip_folders else {}),
    }
    (dst / MANIFEST).write_text(json.dumps(manifest, indent=1))
    problems = verify(dst, hashes=checksums)
    if problems:
        log("COPY IS NOT GOOD:\n  " + "\n  ".join(problems[:50]))
        return 1
    log(
        f"copy verified: {len(entries)} files in {time.time() - t0:.0f} s"
        + (f"; skipped {len(skipped)}" if skipped else "")
    )
    log("Keep the original until the copy is verified where it lands and backed up.")
    return 0


def unpack(copy: Path, dest_root: Path, *, checksums: bool = False, log=print) -> int:
    """Check a packed copy, unpack it into ``dest_root/<run>``, and check the result."""
    copy = Path(copy).resolve()
    problems = verify(copy, hashes=checksums)
    if problems:
        log("the copy is not good; not unpacking:\n  " + "\n  ".join(problems[:50]))
        return 1
    man = json.loads((copy / MANIFEST).read_text())
    dst = Path(dest_root).resolve() / man.get("run", copy.name)
    if dst.exists() and any(dst.iterdir()):
        log(f"{dst} already exists and is not empty")
        return 2
    for a in man.get("archives", []):
        log(f"  unpacking {a['path']}")
        with zipfile.ZipFile(copy / a["path"]) as z:
            z.extractall(dst)
    for e in man["files"]:
        if not e.get("archive"):
            _copy(copy / e["path"], dst / e["path"], hashed=False)
    unpacked = dict(man)
    unpacked.pop("archives", None)
    unpacked["files"] = [
        {k: v for k, v in e.items() if k != "archive"} for e in man["files"]
    ]
    (dst / MANIFEST).write_text(json.dumps(unpacked, indent=1))
    problems = verify(dst, hashes=checksums)
    if problems:
        log("UNPACKED COPY IS NOT GOOD:\n  " + "\n  ".join(problems[:50]))
        return 1
    log(f"unpacked and verified: {dst}")
    return 0
