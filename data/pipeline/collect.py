from __future__ import annotations
from pathlib import Path
from .sources.base import BaseSource


def run_collect(
    sources: dict[str, BaseSource],
    raw_dir: Path,
    curated_dir: Path,
    reference_dir: Path | None = None,
    allow_missing: bool = False,
) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for key, source in sources.items():
        print(f"\n  Collecting {source.name} ...")
        # Try reference CSVs first (exact same splits as original pipeline)
        if source.try_use_reference_csvs(curated_dir, reference_dir):
            print(f"  ✓ {source.name} — used reference CSVs")
            continue
        src_dir = raw_dir / source.name
        if not src_dir.exists():
            print(f"  [skip] {source.name} not found in {raw_dir}")
            missing.append(source.name)
            continue
        source.collect(raw_dir, curated_dir)

    if missing:
        msg = (
            f"Missing raw datasets: {missing}. "
            f"Run 'python data/setup_data.py --stage download' "
            f"or pass --reference_dir <path>."
        )
        if allow_missing:
            print(f"\n⚠️  {msg}")
        else:
            raise FileNotFoundError(msg)
    print("\nCollect stage complete.")
