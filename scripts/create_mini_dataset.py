#!/usr/bin/env python3
"""Create a mini version of BinauralCuratedDataset for pipeline verification.

The full BinauralCuratedDataset is ~130GB. This script creates a small (~250MB)
subset that preserves the exact directory structure so all eval scripts work
with fewer samples. Useful for reviewers to verify their setup before
downloading the full dataset.

What gets copied:
  - scaper_fmt/{train,val,test}/{class}/    N files per class per split
  - bg_scaper_fmt/{train,val,test}/{class}/ N files per class per split
  - noise_scaper_fmt/{train,val,test}/{scene}/ N files per scene per split
  - hrtf/CIPIC/*.sofa + *_hrtf.txt          ALL files (essential, ~100MB)
  - start_times.csv

What is skipped (raw source datasets):
  - FSD50K/, ESC-50/, musdb18/, disco_noises/, TAU-acoustic-sounds/

Usage:
    python scripts/create_mini_dataset.py \
        --input_dir /path/to/BinauralCuratedDataset \
        --output_dir /path/to/BinauralCuratedDataset_mini

    # Copy more samples per class:
    python scripts/create_mini_dataset.py \
        --input_dir /path/to/BinauralCuratedDataset \
        --output_dir /path/to/BinauralCuratedDataset_mini \
        --samples_per_class 5
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def get_size_str(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def copy_file(src: str, dst: str, *, follow_symlinks: bool = True) -> int:
    """Copy a single file, creating parent directories as needed.

    Args:
        src: Source file path.
        dst: Destination file path.
        follow_symlinks: If True, resolve symlinks and copy the actual file.

    Returns:
        Size of the copied file in bytes.
    """
    if follow_symlinks and os.path.islink(src):
        src = os.path.realpath(src)
    if not os.path.isfile(src):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return os.path.getsize(dst)


def copy_scaper_dir(
    input_base: str,
    output_base: str,
    dir_name: str,
    samples_per_class: int,
    follow_symlinks: bool = True,
    max_dirs_per_split: int = 0,
) -> tuple[int, int]:
    """Copy N samples per class per split from a scaper-format directory.

    Expected layout: dir_name/{train,val,test}/{class_or_scene}/*.wav

    Args:
        input_base: Root of the full dataset.
        output_base: Root of the mini dataset.
        dir_name: Directory name (e.g. "scaper_fmt", "bg_scaper_fmt").
        samples_per_class: Number of files to copy per class per split.
        follow_symlinks: Resolve symlinks before copying.
        max_dirs_per_split: Max number of subdirectories to sample per split.
            0 means no limit. Useful for noise_scaper_fmt/test/ which has
            7200 numbered directories.

    Returns:
        (files_copied, bytes_copied)
    """
    src_dir = os.path.join(input_base, dir_name)
    if not os.path.isdir(src_dir):
        print(f"  [SKIP] {dir_name}/ not found")
        return 0, 0

    total_files = 0
    total_bytes = 0

    for split in ("train", "val", "test"):
        split_dir = os.path.join(src_dir, split)
        if not os.path.isdir(split_dir):
            continue
        classes = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )
        if max_dirs_per_split > 0 and len(classes) > max_dirs_per_split:
            classes = classes[:max_dirs_per_split]
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            # Gather audio files (WAV, FLAC, etc.), sorted for reproducibility
            files = sorted(
                f for f in os.listdir(cls_dir)
                if os.path.isfile(os.path.join(cls_dir, f))
            )
            selected = files[:samples_per_class]
            for fname in selected:
                src_path = os.path.join(cls_dir, fname)
                dst_path = os.path.join(output_base, dir_name, split, cls, fname)
                size = copy_file(src_path, dst_path, follow_symlinks=follow_symlinks)
                total_files += 1
                total_bytes += size

    print(f"  {dir_name}/: {total_files} files ({get_size_str(total_bytes)})")
    return total_files, total_bytes


def copy_hrtf(input_base: str, output_base: str) -> tuple[int, int]:
    """Copy all HRTF .sofa files and split list .txt files.

    Args:
        input_base: Root of the full dataset.
        output_base: Root of the mini dataset.

    Returns:
        (files_copied, bytes_copied)
    """
    hrtf_src = os.path.join(input_base, "hrtf", "CIPIC")
    if not os.path.isdir(hrtf_src):
        print("  [SKIP] hrtf/CIPIC/ not found")
        return 0, 0

    total_files = 0
    total_bytes = 0

    for fname in sorted(os.listdir(hrtf_src)):
        src_path = os.path.join(hrtf_src, fname)
        if not os.path.isfile(src_path):
            continue
        # Copy .sofa files and *_hrtf.txt split lists
        if fname.endswith(".sofa") or fname.endswith("_hrtf.txt"):
            dst_path = os.path.join(output_base, "hrtf", "CIPIC", fname)
            size = copy_file(src_path, dst_path)
            total_files += 1
            total_bytes += size

    print(f"  hrtf/CIPIC/: {total_files} files ({get_size_str(total_bytes)})")
    return total_files, total_bytes


def copy_start_times(input_base: str, output_base: str) -> tuple[int, int]:
    """Copy start_times.csv if it exists.

    Args:
        input_base: Root of the full dataset.
        output_base: Root of the mini dataset.

    Returns:
        (files_copied, bytes_copied)
    """
    src = os.path.join(input_base, "start_times.csv")
    if not os.path.isfile(src):
        print("  [SKIP] start_times.csv not found")
        return 0, 0

    dst = os.path.join(output_base, "start_times.csv")
    size = copy_file(src, dst)
    print(f"  start_times.csv: {get_size_str(size)}")
    return 1, size


def create_mini_dataset(
    input_dir: str, output_dir: str, samples_per_class: int
) -> None:
    """Create a mini version of BinauralCuratedDataset.

    Args:
        input_dir: Path to the full BinauralCuratedDataset.
        output_dir: Path where the mini dataset will be created.
        samples_per_class: Number of files to copy per class per split.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(output_dir):
        print(f"Error: output directory already exists: {output_dir}", file=sys.stderr)
        print("Remove it first or choose a different path.", file=sys.stderr)
        sys.exit(1)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Samples per class per split: {samples_per_class}")
    print()

    grand_files = 0
    grand_bytes = 0

    # 1. Foreground audio (20 classes — keep all classes)
    f, b = copy_scaper_dir(
        input_dir, output_dir, "scaper_fmt", samples_per_class
    )
    grand_files += f
    grand_bytes += b

    # 2. Background audio (141 classes — cap to 20 to control size)
    f, b = copy_scaper_dir(
        input_dir, output_dir, "bg_scaper_fmt", samples_per_class,
        max_dirs_per_split=20,
    )
    grand_files += f
    grand_bytes += b

    # 3. Noise audio (resolve symlinks to copy actual files)
    #    train/val have ~100 scene dirs, test has 7200+ numbered dirs.
    #    Cap to 10 dirs per split to keep the mini dataset small.
    f, b = copy_scaper_dir(
        input_dir, output_dir, "noise_scaper_fmt", samples_per_class,
        follow_symlinks=True,
        max_dirs_per_split=10,
    )
    grand_files += f
    grand_bytes += b

    # 4. HRTF files (copy all)
    f, b = copy_hrtf(input_dir, output_dir)
    grand_files += f
    grand_bytes += b

    # 5. start_times.csv
    f, b = copy_start_times(input_dir, output_dir)
    grand_files += f
    grand_bytes += b

    print()
    print(f"Done. Total: {grand_files} files, {get_size_str(grand_bytes)}")
    print(f"Mini dataset created at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a mini (~100MB) version of BinauralCuratedDataset "
            "for pipeline and environment verification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/create_mini_dataset.py \\\n"
            "      --input_dir /path/to/BinauralCuratedDataset \\\n"
            "      --output_dir /path/to/BinauralCuratedDataset_mini\n"
        ),
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the full BinauralCuratedDataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path where the mini dataset will be created (must not exist).",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=1,
        help="Number of files to copy per class/scene per split (default: 1).",
    )
    args = parser.parse_args()

    create_mini_dataset(args.input_dir, args.output_dir, args.samples_per_class)


if __name__ == "__main__":
    main()
