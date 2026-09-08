-- Add leakage-safe pre-index features and one post-index outcome.

CREATE OR REPLACE TABLE stroke_features AS
WITH cohort AS (
    SELECT * FROM index_stroke_cohort
),
prior_dx AS (
    SELECT
        c.patient_id,
        c.index_date,
        MAX(CASE WHEN d.icd10_code LIKE 'I48%' THEN 1 ELSE 0 END) AS prior_af,
        MAX(CASE WHEN d.icd10_code = 'I10' THEN 1 ELSE 0 END) AS prior_hypertension,
        MAX(CASE WHEN d.icd10_code LIKE 'E11%' THEN 1 ELSE 0 END) AS prior_diabetes
    FROM cohort c
    LEFT JOIN diagnoses d
      ON c.patient_id = d.patient_id
     AND CAST(d.diagnosis_date AS DATE) < c.index_date
     AND CAST(d.diagnosis_date AS DATE) >= c.index_date - INTERVAL '365 days'
    GROUP BY 1, 2
),
prior_utilization AS (
    SELECT
        c.patient_id,
        c.index_date,
        COUNT(DISTINCT e.encounter_id) AS encounters_prior_365d,
        SUM(CASE WHEN e.encounter_type = 'ED' THEN 1 ELSE 0 END) AS ed_visits_prior_365d,
        SUM(CASE WHEN e.encounter_type = 'inpatient' THEN 1 ELSE 0 END) AS inpatient_visits_prior_365d
    FROM cohort c
    LEFT JOIN encounters e
      ON c.patient_id = e.patient_id
     AND CAST(e.encounter_start AS DATE) < c.index_date
     AND CAST(e.encounter_start AS DATE) >= c.index_date - INTERVAL '365 days'
    GROUP BY 1, 2
),
latest_ldl AS (
    SELECT patient_id, index_date, result_value AS latest_ldl_prior_365d
    FROM (
        SELECT
            c.patient_id,
            c.index_date,
            l.result_value,
            ROW_NUMBER() OVER (
                PARTITION BY c.patient_id, c.index_date
                ORDER BY l.lab_date DESC, l.lab_id DESC
            ) AS rn
        FROM cohort c
        INNER JOIN labs l
          ON c.patient_id = l.patient_id
         AND l.test_name = 'LDL'
         AND CAST(l.lab_date AS DATE) < c.index_date
         AND CAST(l.lab_date AS DATE) >= c.index_date - INTERVAL '365 days'
         AND l.result_value IS NOT NULL
    )
    WHERE rn = 1
),
readmission AS (
    SELECT
        c.patient_id,
        c.index_date,
        CASE WHEN COUNT(e.encounter_id) > 0 THEN 1 ELSE 0 END AS readmission_30d
    FROM cohort c
    LEFT JOIN encounters e
      ON c.patient_id = e.patient_id
     AND e.encounter_id <> c.encounter_id
     AND e.encounter_type = 'inpatient'
     AND CAST(e.encounter_start AS DATE) > c.index_date
     AND CAST(e.encounter_start AS DATE) <= c.index_date + INTERVAL '30 days'
    GROUP BY 1, 2
)
SELECT
    c.*,
    COALESCE(d.prior_af, 0) AS prior_af,
    COALESCE(d.prior_hypertension, 0) AS prior_hypertension,
    COALESCE(d.prior_diabetes, 0) AS prior_diabetes,
    COALESCE(u.encounters_prior_365d, 0) AS encounters_prior_365d,
    COALESCE(u.ed_visits_prior_365d, 0) AS ed_visits_prior_365d,
    COALESCE(u.inpatient_visits_prior_365d, 0) AS inpatient_visits_prior_365d,
    l.latest_ldl_prior_365d,
    COALESCE(r.readmission_30d, 0) AS readmission_30d
FROM cohort c
LEFT JOIN prior_dx d USING (patient_id, index_date)
LEFT JOIN prior_utilization u USING (patient_id, index_date)
LEFT JOIN latest_ldl l USING (patient_id, index_date)
LEFT JOIN readmission r USING (patient_id, index_date)
ORDER BY c.patient_id;
