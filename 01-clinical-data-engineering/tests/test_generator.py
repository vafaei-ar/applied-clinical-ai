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


def _index_strokes(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """First qualifying stroke encounter per patient, as the cohort SQL defines it."""
    stroke_ids = set(
        tables["diagnoses"].loc[tables["diagnoses"]["icd10_code"].str.startswith("I63"), "encounter_id"]
    )
    enc = tables["encounters"]
    enc = enc[enc["encounter_id"].isin(stroke_ids) & enc["encounter_type"].isin(["ED", "inpatient"])]
    enc = enc.assign(day=pd.to_datetime(enc["encounter_start"]).dt.normalize())
    return enc.sort_values(["patient_id", "day", "encounter_id"]).groupby("patient_id").head(1)


def _readmitted(tables: dict[str, pd.DataFrame], index: pd.DataFrame) -> pd.Series:
    enc = tables["encounters"]
    enc = enc[enc["encounter_type"] == "inpatient"].assign(
        day=lambda f: pd.to_datetime(f["encounter_start"]).dt.normalize()
    )
    flags = []
    for row in index.itertuples():
        mine = enc[(enc["patient_id"] == row.patient_id) & (enc["encounter_id"] != row.encounter_id)]
        in_window = (mine["day"] > row.day) & (mine["day"] <= row.day + pd.Timedelta(days=30))
        flags.append(bool(in_window.any()))
    return pd.Series(flags, index=index["patient_id"].to_numpy())


def test_readmissions_only_append_rows(tmp_path: Path) -> None:
    base = generate_dataset(tmp_path / "base", n_patients=1500, seed=5, readmission_rate=None)
    full = generate_dataset(tmp_path / "full", n_patients=1500, seed=5)

    # Every table other than the three that gain rows is untouched, and those three only grow.
    for name in ("patients", "coverage", "procedures", "medications", "labs"):
        pd.testing.assert_frame_equal(base[name], full[name])
    for name in ("encounters", "diagnoses", "claims"):
        pd.testing.assert_frame_equal(base[name], full[name].iloc[: len(base[name])])
        assert len(full[name]) > len(base[name])

    added = full["encounters"].iloc[len(base["encounters"]) :]
    assert (added["encounter_type"] == "inpatient").all()
    assert added["encounter_id"].is_unique
    added_dx = full["diagnoses"].iloc[len(base["diagnoses"]) :]
    assert not added_dx["icd10_code"].str.startswith("I63").any()
    assert full["claims"]["claim_id"].is_unique
    assert len(full["claims"]) == len(full["encounters"])  # still one claim per encounter

    # The index strokes, and so the cohort, are exactly the same with and without readmissions.
    pd.testing.assert_frame_equal(
        _index_strokes(base).reset_index(drop=True), _index_strokes(full).reset_index(drop=True)
    )


def test_readmission_rate_is_near_target_and_depends_on_age(tmp_path: Path) -> None:
    tables = generate_dataset(tmp_path, n_patients=6000, seed=9, readmission_rate=0.12)
    index = _index_strokes(tables)
    flags = _readmitted(tables, index)

    assert 0.07 <= flags.mean() <= 0.17
    births = tables["patients"].set_index("patient_id")["birth_date"]
    age = (index["day"].to_numpy() - births.loc[index["patient_id"]].to_numpy()) / pd.Timedelta(days=365.25)
    age = pd.Series(age, index=index["patient_id"].to_numpy())
    assert age[flags].mean() > age[~flags].mean()  # older patients are readmitted more often


def test_generator_rejects_invalid_readmission_rate(tmp_path: Path) -> None:
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="readmission_rate"):
            generate_dataset(tmp_path, n_patients=200, readmission_rate=bad)
