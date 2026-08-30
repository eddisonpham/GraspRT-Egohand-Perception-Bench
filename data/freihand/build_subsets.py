"""Build 'dev' (200) and 'full' (all eval) subset index files for FreiHAND."""
import json
import os
from pathlib import Path

import numpy as np

from loader import resolve_root

OUT = Path(__file__).resolve().parent / "subsets"
OUT.mkdir(exist_ok=True)
N = 3960

rng = np.random.default_rng(42)
dev = sorted(rng.choice(N, size=200, replace=False).tolist())
full = list(range(N))

with open(OUT / "dev.json", "w") as f:
    json.dump(dev, f)
with open(OUT / "full.json", "w") as f:
    json.dump(full, f)
print(f"wrote dev.json ({len(dev)} imgs) and full.json ({len(full)} imgs) to {OUT}")
print("dev head:", dev[:5])