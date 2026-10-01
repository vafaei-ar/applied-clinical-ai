from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLES = (
    "patients",
    "coverage",
    "encounters",
    "diagnoses",
    "procedures",
    "medications",
    "labs",
    "claims",
)


def _random_dates(rng: np.random.Generator, start: str, end: str, n: int) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    offsets = rng.integers(0, (end_ts - start_ts).days + 1, size=n)
    return start_ts + pd.to_timedelta(offsets, unit="D")


DATA_END = pd.Timestamp("2025-12-15")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _add_readmissions(
    rng: np.random.Generator,
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    diagnoses: pd.DataFrame,
    claims: pd.DataFrame,
    *,
    rate: float,
    encounter_counter: int,
    diagnosis_counter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append 30-day inpatient readmissions after each patient's index stroke encounter.

    The base generator places encounter types at random, so it yields almost no readmissions and
    none that depend on the patient. This step runs last, on its own random stream, and only
    appends rows, so every row generated before it (and therefore the stroke cohort) is unchanged.

    Risk depends on the same pre-index quantities that ``02_features.sql`` builds (age, prior
    atrial fibrillation, hypertension, diabetes, and ED and inpatient use in the prior 365 days),
    so a model can learn real, modest signal. The intercept is solved so that the expected share
    of index strokes with a readmission equals ``rate``. Readmissions that would start after the
    end of the data are not created, so strokes near the end have less chance of one.
    """
    dx = diagnoses.assign(dx_day=pd.to_datetime(diagnoses["diagnosis_date"]).dt.normalize())
    enc = encounters.assign(day=pd.to_datetime(encounters["encounter_start"]).dt.normalize())

    stroke_encounter_ids = set(dx.loc[dx["icd10_code"].str.startswith("I63"), "encounter_id"])
    qualifying = enc[
        enc["encounter_id"].isin(stroke_encounter_ids) & enc["encounter_type"].isin(["ED", "inpatient"])
    ]
    # Same rule as the cohort SQL: earliest qualifying stroke by calendar date, ties by encounter_id.
    index = (
        qualifying.sort_values(["patient_id", "day", "encounter_id"])
        .groupby("patient_id", as_index=False)
        .head(1)
    )
    if index.empty:
        return encounters, diagnoses, claims

    birth = patients.set_index("patient_id")["birth_date"]
    enc_by_patient = {pid: group for pid, group in enc.groupby("patient_id")}
    dx_by_patient = {pid: group for pid, group in dx.groupby("patient_id")}

    day_30 = pd.Timedelta(days=30)
    year = pd.Timedelta(days=365)
    linear, already = [], []
    for row in index.itertuples():
        e = enc_by_patient[row.patient_id]
        d = dx_by_patient.get(row.patient_id)
        prior = e[(e["day"] < row.day) & (e["day"] >= row.day - year)]
        codes = set()
        if d is not None:
            codes = set(d[(d["dx_day"] < row.day) & (d["dx_day"] >= row.day - year)]["icd10_code"])
        readmitted = e[
            (e["encounter_id"] != row.encounter_id)
            & (e["encounter_type"] == "inpatient")
            & (e["day"] > row.day)
            & (e["day"] <= row.day + day_30)
        ]
        age = (row.day - pd.Timestamp(birth[row.patient_id])).days / 365.25
        linear.append(
            0.04 * (age - 70)
            + 0.8 * any(c.startswith("I48") for c in codes)
            + 0.5 * any(c.startswith("E11") for c in codes)
            + 0.3 * ("I10" in codes)
            + 0.25 * min(int((prior["encounter_type"] == "ED").sum()), 4)
            + 0.35 * min(int((prior["encounter_type"] == "inpatient").sum()), 3)
        )
        already.append(len(readmitted) > 0)
    linear_arr = np.array(linear)
    already_arr = np.array(already)

    def expected_rate(intercept: float) -> float:
        p = _sigmoid(intercept + linear_arr)
        return float(np.mean(already_arr + (~already_arr) * p))

    low, high = -12.0, 8.0
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if expected_rate(mid) < rate else (low, mid)
    probability = _sigmoid((low + high) / 2 + linear_arr)

    new_encounters: list[dict[str, object]] = []
    new_diagnoses: list[dict[str, object]] = []
    new_claims: list[dict[str, object]] = []
    chronic = [("I10", "Hypertension", 0.32), ("I48.91", "Atrial fibrillation", 0.10),
               ("E11.9", "Type 2 diabetes", 0.18), ("N18.3", "Chronic kidney disease", 0.09)]
    chronic_p = np.array([c[2] for c in chronic]) / sum(c[2] for c in chronic)

    for row, p, was_readmitted in zip(index.itertuples(), probability, already_arr, strict=True):
        offset = int(rng.integers(6, 29))  # after even the longest index stay, inside the window
        if was_readmitted or rng.random() >= p:
            continue
        start = row.day + pd.Timedelta(days=offset)
        if start > DATA_END:
            continue
        encounter_id = f"E{encounter_counter + len(new_encounters) + 1:08d}"
        end = start + pd.Timedelta(hours=int(rng.integers(2, 120)))
        new_encounters.append(
            {
                "encounter_id": encounter_id,
                "patient_id": row.patient_id,
                "encounter_start": start,
                "encounter_end": pd.NaT if rng.random() < 0.015 else end,
                "encounter_type": "inpatient",
                "facility_id": rng.choice(["HOSP_A", "HOSP_B"]),
            }
        )
        code, label, _ = chronic[int(rng.choice(len(chronic), p=chronic_p))]
        new_diagnoses.append(
            {
                "diagnosis_id": f"D{diagnosis_counter + len(new_diagnoses) + 1:09d}",
                "patient_id": row.patient_id,
                "encounter_id": encounter_id,
                "diagnosis_date": start + pd.Timedelta(days=int(rng.integers(0, 3))),
                "icd10_code": code,
                "diagnosis_name": label,
            }
        )
        allowed = float(np.round(rng.lognormal(mean=5.4, sigma=0.8), 2))
        new_claims.append(
            {
                "claim_id": f"CLM{len(claims) + len(new_claims) + 1:09d}",
                "patient_id": row.patient_id,
                "encounter_id": encounter_id,
                "service_date": start,
                "place_of_service": "inpatient",
                "primary_hcpcs": "99285",
                "allowed_amount": allowed,
                "paid_amount": float(np.round(allowed * rng.uniform(0.55, 0.95), 2)),
                "claim_status": rng.choice(["paid", "denied"], p=[0.94, 0.06]),
            }
        )

    return (
        pd.concat([encounters, pd.DataFrame(new_encounters)], ignore_index=True),
        pd.concat([diagnoses, pd.DataFrame(new_diagnoses)], ignore_index=True),
        pd.concat([claims, pd.DataFrame(new_claims)], ignore_index=True),
    )


def generate_dataset(
    output_dir: str | Path,
    n_patients: int = 5_000,
    seed: int = 20260908,
    readmission_rate: float | None = 0.12,
) -> dict[str, pd.DataFrame]:
    """Generate a reproducible longitudinal synthetic clinical dataset.

    The generator intentionally creates several edge cases for later SQL exercises:
    duplicate diagnosis rows, interrupted coverage, missing encounter end times,
    multiple stroke encounters, and diagnoses recorded after the encounter start.

    ``readmission_rate`` is the approximate share of index strokes followed by an inpatient
    readmission within 30 days, added after everything else with risk that depends on age and
    comorbidity (see ``_add_readmissions``). Pass ``None`` to skip it and get the earlier,
    purely random data, in which readmissions are almost nonexistent.
    """
    if n_patients < 100:
        raise ValueError("n_patients must be at least 100 so edge cases are represented.")
    if readmission_rate is not None and not 0 < readmission_rate < 1:
        raise ValueError("readmission_rate must be between 0 and 1, or None to skip it.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    patient_ids = np.array([f"P{i:06d}" for i in range(1, n_patients + 1)])
    birth_dates = _random_dates(rng, "1930-01-01", "2002-12-31", n_patients)
    patients = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "birth_date": birth_dates,
            "sex": rng.choice(["F", "M"], size=n_patients, p=[0.51, 0.49]),
            "race_ethnicity": rng.choice(
                ["White", "Black", "Hispanic", "Asian", "Other"],
                size=n_patients,
                p=[0.58, 0.15, 0.16, 0.07, 0.04],
            ),
            "zip3": rng.integers(100, 999, size=n_patients).astype(str),
        }
    )

    coverage_rows: list[dict[str, object]] = []
    for patient_id in patient_ids:
        start = pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 366)))
        end = pd.Timestamp("2025-12-31") - pd.Timedelta(days=int(rng.integers(0, 181)))
        if rng.random() < 0.12:
            gap_start = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
            gap_days = int(rng.integers(30, 181))
            gap_end = gap_start + pd.Timedelta(days=gap_days)
            if start < gap_start < gap_end < end:
                coverage_rows.append(
                    {
                        "coverage_id": f"C{len(coverage_rows)+1:07d}",
                        "patient_id": patient_id,
                        "coverage_start": start,
                        "coverage_end": gap_start - pd.Timedelta(days=1),
                        "plan_type": rng.choice(["Commercial", "Medicare", "Medicaid"]),
                    }
                )
                coverage_rows.append(
                    {
                        "coverage_id": f"C{len(coverage_rows)+1:07d}",
                        "patient_id": patient_id,
                        "coverage_start": gap_end,
                        "coverage_end": end,
                        "plan_type": rng.choice(["Commercial", "Medicare", "Medicaid"]),
                    }
                )
                continue
        coverage_rows.append(
            {
                "coverage_id": f"C{len(coverage_rows)+1:07d}",
                "patient_id": patient_id,
                "coverage_start": start,
                "coverage_end": end,
                "plan_type": rng.choice(["Commercial", "Medicare", "Medicaid"]),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    encounter_rows: list[dict[str, object]] = []
    stroke_patients = set(rng.choice(patient_ids, size=max(1, int(n_patients * 0.08)), replace=False))
    stroke_encounter_ids: set[str] = set()
    encounter_counter = 0
    for patient_id in patient_ids:
        n_encounters = int(rng.poisson(6)) + 1
        dates = np.sort(_random_dates(rng, "2020-01-01", "2025-12-15", n_encounters))
        patient_stroke_positions: set[int] = set()
        if patient_id in stroke_patients:
            patient_stroke_positions.add(int(rng.integers(0, n_encounters)))
            if n_encounters > 1 and rng.random() < 0.20:
                patient_stroke_positions.add(int(rng.integers(0, n_encounters)))
        for position, start in enumerate(dates):
            encounter_counter += 1
            encounter_id = f"E{encounter_counter:08d}"
            encounter_type = rng.choice(["outpatient", "ED", "inpatient"], p=[0.70, 0.16, 0.14])
            if position in patient_stroke_positions:
                encounter_type = rng.choice(["ED", "inpatient"], p=[0.35, 0.65])
                stroke_encounter_ids.add(encounter_id)
            duration_hours = int(rng.integers(2, 120 if encounter_type == "inpatient" else 24))
            encounter_end = pd.Timestamp(start) + pd.Timedelta(hours=duration_hours)
            if rng.random() < 0.015:
                encounter_end = pd.NaT
            encounter_rows.append(
                {
                    "encounter_id": encounter_id,
                    "patient_id": patient_id,
                    "encounter_start": pd.Timestamp(start),
                    "encounter_end": encounter_end,
                    "encounter_type": encounter_type,
                    "facility_id": rng.choice(["HOSP_A", "HOSP_B", "CLINIC_A", "CLINIC_B"]),
                }
            )
    encounters = pd.DataFrame(encounter_rows)

    diagnosis_rows: list[dict[str, object]] = []
    diagnosis_counter = 0
    chronic_codes = [
        ("I10", "Hypertension", 0.32),
        ("I48.91", "Atrial fibrillation", 0.10),
        ("E11.9", "Type 2 diabetes", 0.18),
        ("N18.3", "Chronic kidney disease", 0.09),
    ]
    encounter_lookup = encounters.set_index("encounter_id")
    for encounter_id, row in encounter_lookup.iterrows():
        for code, label, probability in chronic_codes:
            if rng.random() < probability:
                diagnosis_counter += 1
                dx_date = pd.Timestamp(row["encounter_start"]) + pd.Timedelta(days=int(rng.integers(0, 3)))
                diagnosis_rows.append(
                    {
                        "diagnosis_id": f"D{diagnosis_counter:09d}",
                        "patient_id": row["patient_id"],
                        "encounter_id": encounter_id,
                        "diagnosis_date": dx_date,
                        "icd10_code": code,
                        "diagnosis_name": label,
                    }
                )
        if encounter_id in stroke_encounter_ids:
            diagnosis_counter += 1
            diagnosis_rows.append(
                {
                    "diagnosis_id": f"D{diagnosis_counter:09d}",
                    "patient_id": row["patient_id"],
                    "encounter_id": encounter_id,
                    "diagnosis_date": pd.Timestamp(row["encounter_start"]),
                    "icd10_code": "I63.9",
                    "diagnosis_name": "Cerebral infarction, unspecified",
                }
            )
    diagnoses = pd.DataFrame(diagnosis_rows)

    # Inject a small number of exact clinical duplicates under new row IDs.
    n_duplicates = max(1, int(len(diagnoses) * 0.005))
    duplicate_sample = diagnoses.sample(n=n_duplicates, random_state=seed).copy()
    for idx in duplicate_sample.index:
        diagnosis_counter += 1
        duplicate_sample.loc[idx, "diagnosis_id"] = f"D{diagnosis_counter:09d}"
    diagnoses = pd.concat([diagnoses, duplicate_sample], ignore_index=True)

    # Inject a few implausibly late diagnosis timestamps for data-quality exercises.
    late_n = max(1, int(len(diagnoses) * 0.003))
    late_idx = rng.choice(diagnoses.index.to_numpy(), size=late_n, replace=False)
    diagnoses.loc[late_idx, "diagnosis_date"] = (
        pd.to_datetime(diagnoses.loc[late_idx, "diagnosis_date"]) + pd.Timedelta(days=30)
    )

    procedure_catalog = [
        ("70450", "CT head"),
        ("70551", "MRI brain"),
        ("93000", "Electrocardiogram"),
        ("93306", "Echocardiogram"),
    ]
    procedure_rows: list[dict[str, object]] = []
    procedure_counter = 0
    for _, row in encounters.iterrows():
        if rng.random() < 0.30 or row["encounter_id"] in stroke_encounter_ids:
            n_proc = 2 if row["encounter_id"] in stroke_encounter_ids and rng.random() < 0.60 else 1
            for _ in range(n_proc):
                procedure_counter += 1
                code, name = procedure_catalog[int(rng.integers(0, len(procedure_catalog)))]
                procedure_rows.append(
                    {
                        "procedure_id": f"R{procedure_counter:09d}",
                        "patient_id": row["patient_id"],
                        "encounter_id": row["encounter_id"],
                        "procedure_date": row["encounter_start"],
                        "cpt_hcpcs_code": code,
                        "procedure_name": name,
                    }
                )
    procedures = pd.DataFrame(procedure_rows)

    medication_catalog = [
        ("apixaban", "anticoagulant"),
        ("atorvastatin", "statin"),
        ("lisinopril", "antihypertensive"),
        ("metformin", "antidiabetic"),
        ("aspirin", "antiplatelet"),
    ]
    medication_rows: list[dict[str, object]] = []
    medication_counter = 0
    for patient_id in patient_ids:
        for _ in range(int(rng.poisson(1.8))):
            medication_counter += 1
            name, drug_class = medication_catalog[int(rng.integers(0, len(medication_catalog)))]
            start_date = _random_dates(rng, "2020-01-01", "2025-11-30", 1)[0]
            medication_rows.append(
                {
                    "medication_id": f"M{medication_counter:09d}",
                    "patient_id": patient_id,
                    "medication_name": name,
                    "drug_class": drug_class,
                    "start_date": start_date,
                    "end_date": start_date + pd.Timedelta(days=int(rng.integers(30, 730))),
                }
            )
    medications = pd.DataFrame(medication_rows)

    lab_specs = {
        "creatinine": (1.0, 0.35, "mg/dL"),
        "LDL": (110.0, 35.0, "mg/dL"),
        "HbA1c": (6.1, 1.2, "%"),
        "hemoglobin": (13.5, 1.8, "g/dL"),
    }
    lab_rows: list[dict[str, object]] = []
    lab_counter = 0
    for patient_id in patient_ids:
        for _ in range(int(rng.poisson(3.5)) + 1):
            lab_counter += 1
            test_name = rng.choice(list(lab_specs))
            mean, sd, unit = lab_specs[test_name]
            result_value = float(max(0.01, rng.normal(mean, sd)))
            if rng.random() < 0.03:
                result_value = np.nan
            lab_rows.append(
                {
                    "lab_id": f"L{lab_counter:09d}",
                    "patient_id": patient_id,
                    "lab_date": _random_dates(rng, "2020-01-01", "2025-12-15", 1)[0],
                    "test_name": test_name,
                    "result_value": result_value,
                    "unit": unit,
                }
            )
    labs = pd.DataFrame(lab_rows)

    claim_rows: list[dict[str, object]] = []
    for i, row in encounters.reset_index(drop=True).iterrows():
        allowed = float(np.round(rng.lognormal(mean=5.4, sigma=0.8), 2))
        paid = float(np.round(allowed * rng.uniform(0.55, 0.95), 2))
        claim_rows.append(
            {
                "claim_id": f"CLM{i+1:09d}",
                "patient_id": row["patient_id"],
                "encounter_id": row["encounter_id"],
                "service_date": row["encounter_start"],
                "place_of_service": row["encounter_type"],
                "primary_hcpcs": "99213" if row["encounter_type"] == "outpatient" else "99285",
                "allowed_amount": allowed,
                "paid_amount": paid,
                "claim_status": rng.choice(["paid", "denied"], p=[0.94, 0.06]),
            }
        )
    claims = pd.DataFrame(claim_rows)

    if readmission_rate is not None:
        encounters, diagnoses, claims = _add_readmissions(
            np.random.default_rng([seed, 30]),
            patients,
            encounters,
            diagnoses,
            claims,
            rate=readmission_rate,
            encounter_counter=encounter_counter,
            diagnosis_counter=diagnosis_counter,
        )

    tables = {
        "patients": patients,
        "coverage": coverage,
        "encounters": encounters,
        "diagnoses": diagnoses,
        "procedures": procedures,
        "medications": medications,
        "labs": labs,
        "claims": claims,
    }

    for name, frame in tables.items():
        frame.to_csv(output_path / f"{name}.csv", index=False)

    return tables


if __name__ == "__main__":
    generate_dataset(Path(__file__).resolve().parents[2] / "data" / "generated")
