# Exercise 01: Inspect the synthetic clinical database

The goal is to understand the data before writing a cohort query.

## Setup

From `01-clinical-data-engineering/`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"
python data/generate_synthetic_data.py
python src/clinical_data_engineering/load_database.py
```

Open the database in Python, DuckDB CLI, or your SQL editor.

## Tasks

Write SQL yourself before looking at the supplied cohort SQL.

1. Count patients, encounters, diagnoses, claims, and coverage rows.
2. Count encounters by `encounter_type`.
3. Find patients with more than one coverage period.
4. Count ischemic-stroke diagnosis rows using `I63%`.
5. Count unique patients with an ischemic stroke.
6. Find patients with more than one qualifying stroke encounter.
7. Detect clinical duplicate diagnosis rows while ignoring `diagnosis_id`.
8. Find diagnoses recorded more than 14 days after encounter start.
9. Calculate total and mean paid amount by place of service.
10. Identify the first stroke encounter for each patient using `ROW_NUMBER()`.

## Questions to answer in your notes

- Why should the technical diagnosis row ID not be used to define a clinical duplicate?
- Why is the first observed stroke not automatically an eligible index event?
- How can an interrupted coverage period bias a 365-day lookback feature?
- What leakage occurs if a diagnosis recorded after the index date is used as a predictor?
- Why can claim payment amounts and encounter counts represent different concepts of healthcare utilization?

## Stretch tasks

- Compare `ROW_NUMBER()`, `RANK()`, and `DENSE_RANK()` for index-event selection.
- Write an anti-join that finds diagnoses with no matching encounter.
- Use `EXPLAIN` on one multi-table query and inspect the query plan.
- Create a reusable view for continuously eligible patients.
