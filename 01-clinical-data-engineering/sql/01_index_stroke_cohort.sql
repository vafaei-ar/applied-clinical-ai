-- Build the first eligible ischemic-stroke cohort.
-- Learning targets: CTEs, window functions, index-event logic, date arithmetic,
-- eligibility windows, and prevention of temporal leakage.

CREATE OR REPLACE TABLE index_stroke_cohort AS
WITH qualifying_strokes AS (
    SELECT DISTINCT
        e.patient_id,
        e.encounter_id,
        CAST(e.encounter_start AS DATE) AS index_date
    FROM encounters e
    INNER JOIN diagnoses d
        ON e.encounter_id = d.encounter_id
       AND e.patient_id = d.patient_id
    WHERE d.icd10_code LIKE 'I63%'
      AND e.encounter_type IN ('ED', 'inpatient')
),
stroke_candidates AS (
    SELECT
        patient_id,
        encounter_id,
        index_date,
        ROW_NUMBER() OVER (
            PARTITION BY patient_id
            ORDER BY index_date, encounter_id
        ) AS stroke_number
    FROM qualifying_strokes
),
first_stroke AS (
    SELECT patient_id, encounter_id, index_date
    FROM stroke_candidates
    WHERE stroke_number = 1
),
eligible AS (
    SELECT
        s.patient_id,
        s.encounter_id,
        s.index_date,
        p.birth_date
    FROM first_stroke s
    INNER JOIN patients p USING (patient_id)
    WHERE CAST(p.birth_date AS DATE) <= s.index_date - INTERVAL '18 years'
      AND EXISTS (
          SELECT 1
          FROM coverage c
          WHERE c.patient_id = s.patient_id
            AND CAST(c.coverage_start AS DATE) <= s.index_date - INTERVAL '365 days'
            AND CAST(c.coverage_end AS DATE) >= s.index_date
      )
)
SELECT
    patient_id,
    encounter_id,
    index_date,
    CAST(FLOOR(date_diff('day', CAST(birth_date AS DATE), index_date) / 365.25) AS INTEGER) AS age_at_index
FROM eligible
ORDER BY patient_id;
