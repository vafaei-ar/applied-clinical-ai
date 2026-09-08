from __future__ import annotations

from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parents[1]
SRC = MODULE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clinical_data_engineering.generator import generate_dataset


if __name__ == "__main__":
    output = MODULE_ROOT / "data" / "generated"
    tables = generate_dataset(output_dir=output, n_patients=5_000, seed=20260908)
    print(f"Wrote synthetic data to {output}")
    for name, frame in tables.items():
        print(f"{name:12s} {len(frame):8d} rows")
