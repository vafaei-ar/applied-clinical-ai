# Applied Clinical AI

A hands-on learning and engineering portfolio focused on the technical skills needed to build clinical AI systems from data engineering through model development, deployment, monitoring, clinical statistics, and LLM evaluation.

This repository is organized as a 12-week sequence of small, reproducible projects. Synthetic data is used by default when it makes learning faster, safer, and more controlled. Open clinical data can be added when it provides a clear educational advantage.

## Roadmap

| Module | Focus | Main skills |
|---|---|---|
| `01-clinical-data-engineering` | Clinical SQL and healthcare data engineering | SQL, DuckDB/PostgreSQL, cohort logic, longitudinal data, claims/EHR concepts, testing, dbt/PySpark exposure |
| `02-pytorch-clinical-ml` | Modern clinical ML with PyTorch | Dataset/DataLoader, training loops, AMP, checkpointing, calibration, MLflow, GPU profiling |
| `03-production-ml` | Productionization and MLOps | FastAPI, Docker, pytest, CI/CD, Azure, model registry, monitoring, drift |
| `04-clinical-trials-survival` | Clinical development and time-to-event analysis | trial structure, endpoints, estimands, KM, Cox, competing risks, CDISC/SDTM/ADaM concepts |
| `05-clinical-llm-evaluation` | Evaluation of clinical LLM systems | RAG, structured outputs, factuality, hallucination, robustness, safety, cost/latency, agent evaluation |
| `06-technical-labs` | Short targeted labs | Spark/Databricks, Terraform, AWS/GCP concepts, Kubernetes basics, LangChain/LangGraph, healthcare standards |

## Engineering standard

Each substantive project should include:

- reproducible data generation or documented open-data access
- a clear problem definition and architecture
- tested code with `pytest`
- type hints and linting
- reproducible environments
- GitHub Actions CI where appropriate
- quantitative evaluation
- documentation of technical decisions and limitations
- one concise resume-ready evidence statement after completion

## Current focus

**Module 01: Clinical Data Engineering**

The first project builds a synthetic longitudinal clinical dataset and uses SQL to construct and validate analysis-ready cohorts. The goal is to learn practical healthcare data engineering, not just SQL syntax.

See [`ROADMAP.md`](ROADMAP.md) for the 12-week plan and [`SKILLS_MATRIX.md`](SKILLS_MATRIX.md) for the complete technical skill inventory.
