#!/usr/bin/env python3
"""Download the real NASA PCoE battery dataset used by the validation suite.

Source: NASA Ames Prognostics Center of Excellence (PCoE) "Li-ion Battery
Aging" dataset (BatteryAgingARC-FY08Q4), published by NASA at
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

That archive contains one zip per batch; the batch
``1. BatteryAgingARC-FY08Q4.zip`` holds the four cells the suite validates
against (B0005 / B0006 / B0007 / B0018). This script downloads the outer
archive, unwraps the inner zip, and extracts only those four .mat files into
``missionmind/data/real_nasa/*.mat``.

It is idempotent: it exits early when all four .mat files are already present
(pass ``--force`` to re-download). It uses only the standard library
(urllib + zipfile), so it runs on a clean machine and in CI without extra
dependencies. ``--zip PATH`` skips the download and extracts from a local zip
(useful for testing).

Run:  .venv/Scripts/python.exe -m missionmind.ml.download_nasa_data
"""

import io
import os
import sys
import urllib.request
import zipfile

URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
WANTED = {"B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"}
INNER_MARK = "BatteryAgingARC-FY08Q4"
MIN_SIZE = 1_000_000  # bytes; the real files are tens of MB

REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_nasa")


def _all_present() -> bool:
    for name in WANTED:
        path = os.path.join(REAL_DIR, name)
        if not os.path.exists(path) or os.path.getsize(path) <= MIN_SIZE:
            return False
    return True


def _download_zip(tmp_path: str) -> None:
    print(f"[nasa-data] Downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "missionmind/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp_path, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r[nasa-data] {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()


def _extract(outer_zip_path: str) -> None:
    with zipfile.ZipFile(outer_zip_path) as outer:
        inner_name = None
        for name in outer.namelist():
            if name.lower().endswith(".zip") and INNER_MARK in name:
                inner_name = name
                break
        if inner_name is None:
            raise RuntimeError(f"inner zip ({INNER_MARK}) not found in {outer_zip_path}")
        inner_bytes = outer.read(inner_name)

    extracted = 0
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for info in inner.infolist():
            name = os.path.basename(info.filename)
            if name in WANTED:
                with inner.open(info) as src, open(os.path.join(REAL_DIR, name), "wb") as dst:
                    dst.write(src.read())
                extracted += 1
    if extracted != len(WANTED):
        raise RuntimeError(f"expected {len(WANTED)} .mat files in the inner zip, extracted {extracted}")
    for name in sorted(WANTED):
        size = os.path.getsize(os.path.join(REAL_DIR, name))
        if size <= MIN_SIZE:
            raise RuntimeError(f"{name} is only {size} bytes - download looks wrong")
        print(f"[nasa-data] {name}: {size / 1e6:.1f} MB")


def main() -> None:
    args = sys.argv[1:]
    local_zip = None
    if "--zip" in args:
        local_zip = args[args.index("--zip") + 1]

    if "--force" not in args and not local_zip and _all_present():
        print("[nasa-data] All four .mat files already present in "
              f"{os.path.abspath(REAL_DIR)} - nothing to do.")
        return

    os.makedirs(REAL_DIR, exist_ok=True)
    tmp_zip = os.path.join(REAL_DIR, "_download.zip")
    try:
        source = local_zip
        if source is None:
            _download_zip(tmp_zip)
            source = tmp_zip
        print("[nasa-data] Extracting B0005/B0006/B0007/B0018 ...")
        _extract(source)
        print("[nasa-data] Done. The NASA validation tests will now run in full.")
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


if __name__ == "__main__":
    main()
