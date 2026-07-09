"""Fail the build if the committed data bundle exceeds the size budget (students download it on
first run). Per the course design: total <= 60 MB.

  uv run python tools/check_bundle_size.py
"""
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOTAL_CAP_MB = 60
PER_FILE_WARN_MB = 15


def main():
    total = 0
    rows = []
    for dp, _, fs in os.walk(DATA):
        # skip underscore-prefixed scratch/cache dirs (data/_scratch, data/_neural_cache) — they are
        # gitignored build caches, not part of the committed bundle we budget.
        if any(part.startswith("_") for part in os.path.relpath(dp, DATA).split(os.sep)):
            continue
        for f in fs:
            p = os.path.join(dp, f)
            mb = os.path.getsize(p) / 1e6
            total += mb
            rows.append((mb, os.path.relpath(p, DATA)))
    for mb, rel in sorted(rows, reverse=True):
        flag = "  <-- >15MB" if mb > PER_FILE_WARN_MB else ""
        print(f"  {mb:6.2f} MB  {rel}{flag}")
    print(f"  {'-'*40}\n  {total:6.2f} MB  TOTAL  (cap {TOTAL_CAP_MB} MB)")
    if total > TOTAL_CAP_MB:
        sys.exit(f"BUNDLE TOO LARGE: {total:.1f} MB > {TOTAL_CAP_MB} MB")
    print("  OK — within budget.")


if __name__ == "__main__":
    main()
