#!/usr/bin/env python3
"""Check file sizes in mtl-mar directory, excluding data folder."""
import os
from pathlib import Path

base_dir = Path('/gxfs_work/cau/sunms498/work/mtl-mar')
files = []

for root, dirs, filenames in os.walk(base_dir):
    # Skip data directory
    if 'data' in root.split(os.sep):
        continue
    for filename in filenames:
        filepath = Path(root) / filename
        try:
            size = filepath.stat().st_size
            rel_path = filepath.relative_to(base_dir)
            files.append((str(rel_path), size))
        except Exception as e:
            pass

# Sort by size descending
files.sort(key=lambda x: x[1], reverse=True)

# Print top 20
print('Top 20 largest files (excluding data directory):')
print('=' * 80)
for filepath, size in files[:20]:
    size_mb = size / (1024 * 1024)
    print(f'{size_mb:>10.2f} MB  {filepath}')

