from pathlib import Path

import pandas as pd
import pytest

from clinical_data_engineering.generator import generate_dataset


def test_generator_is_reproducible(tmp_path: Path) -> None:
    first = generate_dataset(tmp_path / "a", n_patients=300, seed=7)
    second = generate_dataset(tmp_path / "b", n_patients=300, seed=7)
    pd.testing.assert_frame_equal(first["patients"], second["patients"])
    pd.testing.assert_frame_equal(first["encounters"], second["encounters"])


def test_generator_creates_expected_tables_and_edge_cases(tmp_path: Path) -> None:
    tables = generate_dataset(tmp_path, n_patients=500, seed=11)

    assert set(tables) == {
        "patients",
        "coverage",
        "encounters",
        "diagnoses",
        "procedures",
        "medications",
        "labs",
        "claims",
    }
    assert tables["patients"]["patient_id"].is_unique
    assert tables["encounters"]["encounter_id"].is_unique
    assert tables["claims"]["claim_id"].is_unique
    assert (tables["diagnoses"]["icd10_code"] == "I63.9").any()
    assert tables["encounters"]["encounter_end"].isna().any()
    assert len(tables["coverage"]) > len(tables["patients"])

    duplicate_groups = (
        tables["diagnoses"]
        .groupby(
            ["patient_id", "encounter_id", "diagnosis_date", "icd10_code", "diagnosis_name"],
            dropna=False,
        )
        .size()
    )
    assert (duplicate_groups > 1).any()


def test_generator_rejects_too_small_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 100"):
        generate_dataset(tmp_path, n_patients=50)
