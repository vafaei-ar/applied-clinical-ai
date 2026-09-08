from pathlib import Path

import duckdb

from clinical_data_engineering.build_cohort import build_cohort
from clinical_data_engineering.generator import generate_dataset
from clinical_data_engineering.load_database import load_database


def test_database_and_cohort_pipeline(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    database = tmp_path / "clinical.duckdb"
    output = tmp_path / "stroke_features.csv"

    tables = generate_dataset(generated, n_patients=700, seed=21)
    load_database(generated, database)
    build_cohort(database, output)

    assert output.exists()

    with duckdb.connect(str(database), read_only=True) as connection:
        loaded_patients = connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        cohort_rows = connection.execute("SELECT COUNT(*) FROM index_stroke_cohort").fetchone()[0]
        duplicate_patients = connection.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT patient_id) FROM index_stroke_cohort"
        ).fetchone()[0]
        underage = connection.execute(
            "SELECT COUNT(*) FROM index_stroke_cohort WHERE age_at_index < 18"
        ).fetchone()[0]
        invalid_coverage = connection.execute(
            """
            SELECT COUNT(*)
            FROM index_stroke_cohort c
            WHERE NOT EXISTS (
                SELECT 1 FROM coverage v
                WHERE v.patient_id = c.patient_id
                  AND CAST(v.coverage_start AS DATE) <= c.index_date - INTERVAL '365 days'
                  AND CAST(v.coverage_end AS DATE) >= c.index_date
            )
            """
        ).fetchone()[0]

    assert loaded_patients == len(tables["patients"])
    assert cohort_rows > 0
    assert duplicate_patients == 0
    assert underage == 0
    assert invalid_coverage == 0
