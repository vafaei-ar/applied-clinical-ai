from __future__ import annotations

from pathlib import Path

import duckdb


def run_sql_file(connection: duckdb.DuckDBPyConnection, sql_path: Path) -> None:
    connection.execute(sql_path.read_text(encoding="utf-8"))


def build_cohort(database_path: str | Path, output_path: str | Path) -> Path:
    """Build the stroke cohort and leakage-safe feature table, then export it."""
    database_path = Path(database_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module_root = Path(__file__).resolve().parents[2]

    with duckdb.connect(str(database_path)) as connection:
        run_sql_file(connection, module_root / "sql" / "01_index_stroke_cohort.sql")
        run_sql_file(connection, module_root / "sql" / "02_features.sql")
        safe_output = str(output_path).replace("'", "''")
        connection.execute(
            f"COPY stroke_features TO '{safe_output}' (HEADER, DELIMITER ',')"
        )

    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = build_cohort(
        root / "data" / "clinical.duckdb",
        root / "results" / "stroke_features.csv",
    )
    print(f"Wrote {result}")
