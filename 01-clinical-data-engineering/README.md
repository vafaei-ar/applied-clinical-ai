# Module 01: Clinical Data Engineering

## Goal

Build a small but realistic longitudinal clinical data system from synthetic EHR and claims-like data, then use SQL to construct an analysis-ready cohort correctly.

The purpose is to learn healthcare data engineering and advanced SQL through a controlled clinical problem rather than isolated syntax exercises.

## Initial clinical scenario

We will create a synthetic cohort of adults with longitudinal encounters, diagnoses, medications, procedures, and laboratory measurements. The first analysis task will identify patients with an index ischemic stroke encounter and construct pre-index and post-index features without temporal leakage.

## Learning objectives

By the end of this module, the repository should demonstrate:

- relational clinical schema design
- synthetic longitudinal clinical data generation
- DuckDB workflow
- PostgreSQL-compatible SQL where practical
- joins, CTEs, subqueries, aggregation, and window functions
- index-event identification
- eligibility and lookback windows
- longitudinal feature engineering
- healthcare coding concepts
- claims-like utilization concepts
- temporal leakage prevention
- data quality and cohort validation tests
- basic dbt/PySpark/Databricks exposure after the core SQL workflow is complete

## Proposed schema

Initial tables:

- `patients`
- `encounters`
- `diagnoses`
- `procedures`
- `medications`
- `labs`
- `coverage`

The schema and codebook will evolve as exercises are added.

## Milestones

### Milestone 1: Generate and inspect the data

- generate reproducible synthetic patients and longitudinal events
- load tables into DuckDB
- inspect keys, dates, missingness, and cardinality
- write basic validation tests

### Milestone 2: Build the index stroke cohort

- identify qualifying stroke encounters
- define first eligible index event
- require appropriate lookback/coverage
- remove invalid temporal records
- validate cohort counts

### Milestone 3: Construct longitudinal features

Examples:

- age at index
- prior AF diagnosis
- hypertension history
- prior healthcare utilization
- selected pre-index labs
- medication exposure
- post-index readmission outcome

All predictors must respect explicit observation windows.

### Milestone 4: Advanced SQL

Use:

- CTEs
- window functions
- conditional aggregation
- date arithmetic
- anti-joins / exclusion logic
- reusable views
- query plans and simple optimization

### Milestone 5: Engineering layer

- automated SQL tests
- Python orchestration
- CI
- dbt-style transformation concepts
- small PySpark translation exercise

## Expected final artifact

A reproducible pipeline that can run approximately as:

```bash
python data/generate_synthetic_data.py
python src/load_database.py
python src/build_cohort.py
pytest
```

The exact interface may change as the project develops.

## Rule for this module

Synthetic data should contain intentional edge cases so that SQL correctness can be tested. Examples include duplicate events, missing dates, diagnoses after index, interrupted coverage, and patients with multiple possible index encounters.
