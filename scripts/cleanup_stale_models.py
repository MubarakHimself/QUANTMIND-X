#!/usr/bin/env python3
"""
Cleanup script to remove stale model files.
Keeps only the most recent .pkl and .json file per symbol/timeframe combination.

Usage:
    python scripts/cleanup_stale_models.py [--dry-run]
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict


def extract_base_symbol(path: Path) -> str:
    """
    Extract base symbol from filename like EURUSD_M5_v20260326-132446.pkl
    Returns 'EURUSD_M5' (symbol + timeframe, without version/timestamp).
    """
    name = path.stem  # e.g. EURUSD_M5_v20260326-132446
    # The version stamp always starts with 'v' followed by date-time
    # e.g. v20260326-132446
    parts = name.split("_")
    if len(parts) >= 2:
        # symbol is first part, timeframe is second
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def cleanup_model_dir(model_dir: Path, dry_run: bool = True) -> None:
    """Remove stale model files, keeping only the latest per symbol/timeframe."""
    if not model_dir.exists():
        print(f"Directory does not exist: {model_dir}")
        return

    # Group files by base symbol (symbol + timeframe)
    by_base: dict = defaultdict(list)
    for f in model_dir.glob("*.pkl"):
        base = extract_base_symbol(f)
        by_base[base].append(f)

    # Also group .json sidecar files
    for f in model_dir.glob("*.json"):
        base = extract_base_symbol(f)
        by_base[base].append(f)

    total_removed = 0
    total_size = 0

    for base, files in sorted(by_base.items()):
        if len(files) <= 1:
            print(f"  {base}: 1 file, keeping (no cleanup needed)")
            continue

        # Sort by modification time, newest first
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = files[0]
        stale = files[1:]

        size_stale = sum(f.stat().st_size for f in stale)
        print(f"  {base}: {len(files)} files, keeping {latest.name} ({len(stale)} stale, {size_stale / 1024:.1f} KB to remove)")

        if not dry_run:
            for f in stale:
                try:
                    f.unlink()
                    total_removed += 1
                    total_size += f.stat().st_size
                    print(f"    Removed: {f.name}")
                except Exception as e:
                    print(f"    Failed to remove {f.name}: {e}")

    if dry_run:
        print(f"\n[Dry run] Would remove {sum(len(f) - 1 for f in by_base.values() if len(f) > 1)} stale files.")
        print("Run with --execute to actually delete files.")
    else:
        print(f"\nRemoved {total_removed} stale files, freed {total_size / 1024:.1f} KB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up stale HMM model files")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default is dry-run)")
    parser.add_argument("--dir", default="models/hmm", help="Model directory to clean (default: models/hmm)")
    args = parser.parse_args()

    model_dir = Path(args.dir)
    dry_run = not args.execute

    print(f"Cleaning model directory: {model_dir.absolute()}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}\n")

    cleanup_model_dir(model_dir, dry_run=dry_run)
