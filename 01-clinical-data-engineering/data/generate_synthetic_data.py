from __future__ import annotations

from pathlib import Path

from clinical_data_engineering.generator import generate_dataset

MODULE_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    output = MODULE_ROOT / "data" / "generated"
    tables = generate_dataset(output_dir=output, n_patients=5_000, seed=20260908)
    print(f"Wrote synthetic data to {output}")
    for name, frame in tables.items():
        print(f"{name:12s} {len(frame):8d} rows")
