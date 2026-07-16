#!/usr/bin/env python3
"""Build a single-trigger-class training dataset (<Class>Only/) from raw audio.

Given a folder of raw .wav clips for one trigger class (e.g. sneezing, cough),
this creates the scaper_fmt/{train,val}/<class>/ split for that class, and
points noise_scaper_fmt/ and hrtf/ at shared assets built once and reused
across every trigger class (so re-running this for a new class takes seconds,
not another slow from-scratch symlink/download pass).

Shared assets (built automatically on first use, skipped after):
  - data/raw/background/          raw ambient noise clips (source of truth)
  - data/raw/hrtf/CIPIC/          CIPIC .sofa files (auto-downloaded from
                                   sofacoustics.org if missing)
  - data/shared/noise_scaper_fmt/{train,val}/ambient/   symlink split of the above
  - data/shared/hrtf/             symlink copy + train_hrtf.txt/val_hrtf.txt

Usage:
    python scripts/build_trigger_dataset.py \
        --class_name sneezing \
        --raw_dir data/SneezingOnly/raw/sneezing \
        --output_dir data/SneezingOnly

    # Re-run for a different class later - shared noise/hrtf are reused instantly:
    python scripts/build_trigger_dataset.py \
        --class_name cough \
        --raw_dir data/CoughOnly/raw/cough \
        --output_dir data/CoughOnly
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_NOISE_RAW = os.path.join(REPO_ROOT, "data", "raw", "background")
DEFAULT_HRTF_RAW = os.path.join(REPO_ROOT, "data", "raw", "hrtf", "CIPIC")
DEFAULT_SHARED_NOISE = os.path.join(REPO_ROOT, "data", "shared", "noise_scaper_fmt")
DEFAULT_SHARED_HRTF = os.path.join(REPO_ROOT, "data", "shared", "hrtf")

# Canonical 45-subject CIPIC SOFA list from https://sofacoustics.org/data/database/cipic/
CIPIC_SUBJECTS = [
    "subject_003.sofa", "subject_008.sofa", "subject_009.sofa", "subject_010.sofa",
    "subject_011.sofa", "subject_012.sofa", "subject_015.sofa", "subject_017.sofa",
    "subject_018.sofa", "subject_019.sofa", "subject_020.sofa", "subject_021.sofa",
    "subject_027.sofa", "subject_028.sofa", "subject_033.sofa", "subject_040.sofa",
    "subject_044.sofa", "subject_048.sofa", "subject_050.sofa", "subject_051.sofa",
    "subject_058.sofa", "subject_059.sofa", "subject_060.sofa", "subject_061.sofa",
    "subject_065.sofa", "subject_119.sofa", "subject_124.sofa", "subject_126.sofa",
    "subject_127.sofa", "subject_131.sofa", "subject_133.sofa", "subject_134.sofa",
    "subject_135.sofa", "subject_137.sofa", "subject_147.sofa", "subject_148.sofa",
    "subject_152.sofa", "subject_153.sofa", "subject_154.sofa", "subject_155.sofa",
    "subject_156.sofa", "subject_158.sofa", "subject_162.sofa", "subject_163.sofa",
    "subject_165.sofa",
]

CIPIC_BASE_URL = "https://sofacoustics.org/data/database/cipic/"


def symlink_split(
    raw_dir: str,
    train_dir: str,
    val_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    workers: int = 16,
) -> tuple[int, int]:
    """Symlink-split every file in raw_dir into train_dir/val_dir (parallelized)."""
    files = sorted(f for f in os.listdir(raw_dir) if not f.startswith("."))
    if not files:
        sys.exit(f"No files found in {raw_dir}")

    random.Random(seed).shuffle(files)
    split = int(len(files) * (1 - val_ratio))
    train_files, val_files = files[:split], files[split:]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def link(task: tuple[str, str]) -> None:
        fname, dest_dir = task
        dest = os.path.join(dest_dir, fname)
        if not os.path.lexists(dest):
            os.symlink(os.path.join(raw_dir, fname), dest)

    tasks = [(f, train_dir) for f in train_files] + [(f, val_dir) for f in val_files]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(link, tasks))

    assert set(train_files).isdisjoint(val_files)
    return len(train_files), len(val_files)


def ensure_shared_noise(
    noise_raw: str, shared_noise: str, noise_class: str, val_ratio: float, seed: int
) -> None:
    """Build the shared noise_scaper_fmt split once; skip if already populated."""
    train_dir = os.path.join(shared_noise, "train", noise_class)
    val_dir = os.path.join(shared_noise, "val", noise_class)
    if os.path.isdir(train_dir) and os.listdir(train_dir):
        print(f"[skip] shared noise already built at {shared_noise}")
        return

    if not os.path.isdir(noise_raw):
        sys.exit(
            f"Shared noise raw source missing: {noise_raw}\n"
            f"Pull it first, e.g.:\n"
            f'  rclone copy --drive-shared-with-me "gdrive:/CSE 481 Project/MISO_Dataset/background" '
            f"{noise_raw}/ --progress"
        )

    print(f"Building shared noise_scaper_fmt from {noise_raw} (one-time)...")
    n_train, n_val = symlink_split(noise_raw, train_dir, val_dir, val_ratio, seed)
    print(f"  noise: train={n_train} val={n_val}")


def ensure_shared_hrtf(hrtf_raw: str, shared_hrtf: str) -> None:
    """Download CIPIC sofa files if needed and build the shared hrtf/CIPIC dir once."""
    cipic_dst = os.path.join(shared_hrtf, "CIPIC")
    if os.path.isdir(cipic_dst) and len(
        [f for f in os.listdir(cipic_dst) if f.endswith(".sofa")]
    ) == len(CIPIC_SUBJECTS):
        print(f"[skip] shared hrtf already built at {shared_hrtf}")
        return

    os.makedirs(hrtf_raw, exist_ok=True)
    os.makedirs(cipic_dst, exist_ok=True)

    print(f"Building shared hrtf from {hrtf_raw} (one-time)...")
    for fname in CIPIC_SUBJECTS:
        raw_path = os.path.join(hrtf_raw, fname)
        if not os.path.exists(raw_path):
            print(f"  downloading {fname}")
            urllib.request.urlretrieve(CIPIC_BASE_URL + fname, raw_path)
        dst = os.path.join(cipic_dst, fname)
        if not os.path.lexists(dst):
            os.symlink(raw_path, dst)

    # HRTF is shared across splits (not class-specific), so train/val lists match.
    subject_list = "\n".join(CIPIC_SUBJECTS) + "\n"
    for name in ("train_hrtf.txt", "val_hrtf.txt"):
        with open(os.path.join(cipic_dst, name), "w") as f:
            f.write(subject_list)

    print(f"  hrtf: {len(CIPIC_SUBJECTS)} subjects")


def link_shared(output_dir: str, name: str, target: str) -> None:
    """Point output_dir/name at target as a single directory symlink."""
    link_path = os.path.join(output_dir, name)
    if os.path.islink(link_path):
        os.remove(link_path)
    elif os.path.exists(link_path):
        sys.exit(
            f"{link_path} exists and is a real directory, not a symlink — "
            f"refusing to overwrite. Remove it manually first if you want to replace it."
        )
    os.symlink(target, link_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class_name", required=True, help="e.g. sneezing")
    parser.add_argument("--raw_dir", required=True, help="Dir of raw .wav files for this class")
    parser.add_argument("--output_dir", required=True, help="e.g. data/SneezingOnly")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_class", default="ambient", help="Noise category folder name")
    parser.add_argument("--noise_raw", default=DEFAULT_NOISE_RAW)
    parser.add_argument("--hrtf_raw", default=DEFAULT_HRTF_RAW)
    parser.add_argument("--shared_noise", default=DEFAULT_SHARED_NOISE)
    parser.add_argument("--shared_hrtf", default=DEFAULT_SHARED_HRTF)
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(raw_dir):
        sys.exit(f"raw_dir not found: {raw_dir}")
    n_files = len([f for f in os.listdir(raw_dir) if f.endswith(".wav")])
    if n_files == 0:
        sys.exit(f"No .wav files found in {raw_dir}")

    print(f"Building '{args.class_name}' dataset at {output_dir} from {n_files} raw files\n")

    # 1. Per-class scaper_fmt split (not shared - this is the actual target audio)
    train_dir = os.path.join(output_dir, "scaper_fmt", "train", args.class_name)
    val_dir = os.path.join(output_dir, "scaper_fmt", "val", args.class_name)
    n_train, n_val = symlink_split(raw_dir, train_dir, val_dir, args.val_ratio, args.seed)
    print(f"scaper_fmt: train={n_train} val={n_val}\n")

    # 2. Shared noise + hrtf (built once across all classes, reused after)
    ensure_shared_noise(args.noise_raw, args.shared_noise, args.noise_class, args.val_ratio, args.seed)
    ensure_shared_hrtf(args.hrtf_raw, args.shared_hrtf)
    print()

    # 3. Point this class at the shared assets
    link_shared(output_dir, "noise_scaper_fmt", args.shared_noise)
    link_shared(output_dir, "hrtf", args.shared_hrtf)

    print("Done:")
    print(f"  {output_dir}/scaper_fmt/{{train,val}}/{args.class_name}/")
    print(f"  {output_dir}/noise_scaper_fmt -> {args.shared_noise}")
    print(f"  {output_dir}/hrtf -> {args.shared_hrtf}")


if __name__ == "__main__":
    main()
