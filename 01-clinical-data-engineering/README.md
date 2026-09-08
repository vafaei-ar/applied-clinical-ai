# Module 01: Clinical Data Engineering

## Goal

Build a small but realistic longitudinal clinical data system from synthetic EHR and claims-like data, then use SQL to construct an analysis-ready cohort correctly.

The purpose is to learn healthcare data engineering and advanced SQL through a controlled clinical problem rather than isolated syntax exercises.

## Clinical scenario

We generate 5,000 synthetic adults with longitudinal encounters, diagnoses, procedures, medications, laboratory measurements, coverage periods, and simplified claims. The first analysis task identifies patients with an index ischemic stroke encounter and constructs pre-index features without temporal leakage.

The synthetic data intentionally includes edge cases: duplicate diagnosis rows, missing encounter end times, interrupted insurance coverage, multiple possible stroke encounters, missing lab values, and implausibly late diagnosis timestamps.

## Skills covered

- relational clinical schema design
- reproducible synthetic clinical data generation
- DuckDB and PostgreSQL-compatible SQL where practical
- joins, CTEs, subqueries, aggregation, and window functions
- index-event identification
- eligibility and lookback windows
- longitudinal feature engineering
- ICD-10 and CPT/HCPCS concepts
- simplified claims utilization and payment concepts
- temporal leakage prevention
- data quality and cohort validation tests
- Python packaging, `pytest`, Ruff, and GitHub Actions
- later: dbt concepts, PySpark, Databricks concepts, and query optimization

## Structure

```text
01-clinical-data-engineering/
├── data/
│   └── generate_synthetic_data.py
├── docs/
│   └── codebook.md
├── exercises/
│   └── 01_getting_started.md
├── sql/
│   ├── 00_quality_checks.sql
│   ├── 01_index_stroke_cohort.sql
│   └── 02_features.sql
├── src/clinical_data_engineering/
│   ├── generator.py
│   ├── load_database.py
│   └── build_cohort.py
├── tests/
├── pyproject.toml
└── README.md
```

## Quick start

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"

python data/generate_synthetic_data.py
python src/clinical_data_engineering/load_database.py
python src/clinical_data_engineering/build_cohort.py
pytest -q
```

Generated patient data and local DuckDB files are ignored by Git.

## What to do first

Do **not** begin by reading the supplied cohort solution. Start with [`exercises/01_getting_started.md`](exercises/01_getting_started.md). Inspect the schema, write the requested SQL yourself, then compare your logic with the supplied SQL files.

The codebook is in [`docs/codebook.md`](docs/codebook.md).

## Milestones

### Milestone 1: Generate and inspect the data

- generate reproducible synthetic patients and longitudinal events
- load tables into DuckDB
- inspect keys, dates, missingness, and cardinality
- detect intentional data-quality problems

### Milestone 2: Build the index stroke cohort

- identify qualifying ischemic-stroke encounters
- define the first observed index event
- require age and coverage eligibility
- understand why interrupted coverage complicates lookback logic
- validate cohort counts

### Milestone 3: Construct longitudinal features

Current examples:

- age at index
- prior atrial fibrillation
- prior hypertension
- prior diabetes
- prior ED/inpatient utilization
- latest pre-index LDL
- 30-day readmission outcome

All predictors use explicit pre-index observation windows.

### Milestone 4: Advanced SQL

Add and practice:

- conditional aggregation
- anti-joins and exclusion logic
- reusable views
- coverage-gap logic
- `EXPLAIN` and query plans
- simple query optimization

### Milestone 5: Engineering layer

- automated SQL/data tests
- Python orchestration
- CI
- dbt-style transformations
- small PySpark translation exercise
- Databricks workflow concepts

## Important limitation

This is an educational synthetic dataset, not a validated clinical or claims data model. Codes, utilization patterns, payments, and event distributions are simplified deliberately. The goal is technical reasoning and reproducibility, not epidemiologic realism.
