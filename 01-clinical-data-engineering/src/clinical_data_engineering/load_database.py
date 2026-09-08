from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DATE_COLUMNS = {
    "patients": ["birth_date"],
    "coverage": ["coverage_start", "coverage_end"],
    "encounters": ["encounter_start", "encounter_end"],
    "diagnoses": ["diagnosis_date"],
    "procedures": ["procedure_date"],
    "medications": ["start_date", "end_date"],
    "labs": ["lab_date"],
    "claims": ["service_date"],
}


def load_database(data_dir: str | Path, database_path: str | Path) -> Path:
    """Load generated CSV tables into a DuckDB database with typed date columns."""
    data_dir = Path(data_dir)
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(database_path)) as connection:
        for table_name, date_columns in DATE_COLUMNS.items():
            csv_path = data_dir / f"{table_name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing required table: {csv_path}")

            frame = pd.read_csv(csv_path)
            for column in date_columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")

            connection.register("_input_frame", frame)
            connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _input_frame")
            connection.unregister("_input_frame")

    return database_path


if __name__ == "__main__":
    module_root = Path(__file__).resolve().parents[2]
    db = load_database(module_root / "data" / "generated", module_root / "data" / "clinical.duckdb")
    print(f"Created {db}")
