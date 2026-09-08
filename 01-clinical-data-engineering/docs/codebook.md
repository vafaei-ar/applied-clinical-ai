# Module 01 Codebook

The dataset is synthetic. Values are generated for education and do not represent real patients.

## `patients`

| Column | Meaning |
|---|---|
| `patient_id` | Synthetic patient identifier |
| `birth_date` | Date of birth |
| `sex` | Recorded sex (`F`/`M`) |
| `race_ethnicity` | Synthetic demographic category |
| `zip3` | Synthetic 3-digit ZIP prefix |

## `coverage`

| Column | Meaning |
|---|---|
| `coverage_id` | Coverage-period identifier |
| `patient_id` | Patient identifier |
| `coverage_start` | Coverage start date |
| `coverage_end` | Coverage end date |
| `plan_type` | Commercial, Medicare, or Medicaid |

Some patients intentionally have two coverage periods separated by a gap.

## `encounters`

| Column | Meaning |
|---|---|
| `encounter_id` | Encounter identifier |
| `patient_id` | Patient identifier |
| `encounter_start` | Encounter start timestamp |
| `encounter_end` | Encounter end timestamp; some are intentionally missing |
| `encounter_type` | outpatient, ED, or inpatient |
| `facility_id` | Synthetic facility identifier |

## `diagnoses`

| Column | Meaning |
|---|---|
| `diagnosis_id` | Technical row identifier |
| `patient_id` | Patient identifier |
| `encounter_id` | Encounter identifier |
| `diagnosis_date` | Diagnosis timestamp/date |
| `icd10_code` | Simplified ICD-10-CM code |
| `diagnosis_name` | Human-readable diagnosis |

Current codes include `I63.9` ischemic stroke, `I48.91` atrial fibrillation, `I10` hypertension, `E11.9` diabetes, and `N18.3` chronic kidney disease. A small number of clinical duplicate rows and implausibly late timestamps are intentionally injected.

## `procedures`

| Column | Meaning |
|---|---|
| `procedure_id` | Procedure row identifier |
| `patient_id` | Patient identifier |
| `encounter_id` | Encounter identifier |
| `procedure_date` | Procedure date |
| `cpt_hcpcs_code` | Simplified CPT/HCPCS-like code |
| `procedure_name` | Procedure label |

## `medications`

| Column | Meaning |
|---|---|
| `medication_id` | Medication-exposure identifier |
| `patient_id` | Patient identifier |
| `medication_name` | Medication name |
| `drug_class` | Drug class |
| `start_date` | Exposure start |
| `end_date` | Exposure end |

## `labs`

| Column | Meaning |
|---|---|
| `lab_id` | Laboratory row identifier |
| `patient_id` | Patient identifier |
| `lab_date` | Laboratory measurement date |
| `test_name` | creatinine, LDL, HbA1c, or hemoglobin |
| `result_value` | Numeric result; some are intentionally missing |
| `unit` | Result unit |

## `claims`

| Column | Meaning |
|---|---|
| `claim_id` | Claim identifier |
| `patient_id` | Patient identifier |
| `encounter_id` | Linked encounter identifier |
| `service_date` | Date of service |
| `place_of_service` | Simplified care setting |
| `primary_hcpcs` | Simplified HCPCS/CPT-like billing code |
| `allowed_amount` | Synthetic allowed amount |
| `paid_amount` | Synthetic paid amount |
| `claim_status` | paid or denied |

The claims table is intentionally simplified. It exists to practice payer-style joins, utilization summaries, coding concepts, and cost aggregation. It is not a faithful CMS claims model.
