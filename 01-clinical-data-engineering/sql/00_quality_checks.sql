-- Milestone 1 quality checks. These queries should return small, interpretable counts.

-- 1. Primary-key duplicates. Expected: zero for IDs generated as technical keys.
SELECT 'patients.patient_id' AS check_name, COUNT(*) - COUNT(DISTINCT patient_id) AS n_issues FROM patients
UNION ALL
SELECT 'encounters.encounter_id', COUNT(*) - COUNT(DISTINCT encounter_id) FROM encounters
UNION ALL
SELECT 'claims.claim_id', COUNT(*) - COUNT(DISTINCT claim_id) FROM claims;

-- 2. Missing encounter end timestamps. These are intentionally injected.
SELECT COUNT(*) AS missing_encounter_end
FROM encounters
WHERE encounter_end IS NULL;

-- 3. Clinical duplicate diagnoses, ignoring the technical diagnosis_id.
-- These are intentionally injected and should be detected.
SELECT COUNT(*) AS duplicate_groups
FROM (
    SELECT patient_id, encounter_id, diagnosis_date, icd10_code, diagnosis_name, COUNT(*) AS n
    FROM diagnoses
    GROUP BY 1, 2, 3, 4, 5
    HAVING COUNT(*) > 1
);

-- 4. Diagnosis timestamps well after encounter start. Some are intentionally injected.
SELECT COUNT(*) AS late_diagnoses
FROM diagnoses d
JOIN encounters e USING (encounter_id)
WHERE d.diagnosis_date > e.encounter_start + INTERVAL '14 days';

-- 5. Referential-integrity check.
SELECT COUNT(*) AS orphan_diagnoses
FROM diagnoses d
LEFT JOIN encounters e USING (encounter_id)
WHERE e.encounter_id IS NULL;
