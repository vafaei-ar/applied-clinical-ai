# 12-Week Roadmap

This roadmap is designed around recurring technical gaps observed across clinical AI, healthcare data science, biotech/pharma, and research scientist roles.

## Weeks 1-3: Clinical Data Engineering

Goal: build and validate a longitudinal clinical cohort from synthetic EHR/claims-like data.

Learn and practice:

- advanced SQL
- DuckDB and PostgreSQL concepts
- relational schema design
- joins, CTEs, subqueries, window functions
- index-date logic and temporal leakage prevention
- longitudinal feature construction
- healthcare coding concepts: ICD-10, CPT/HCPCS, medications, labs, encounters
- claims-style enrollment/utilization concepts
- data quality checks and cohort validation
- dbt-style transformation concepts
- basic PySpark/Databricks exposure

Deliverable: reproducible cohort-building pipeline with tests and documentation.

## Weeks 3-6: Modern PyTorch Clinical ML

Goal: train a reproducible clinical prediction model using a modern PyTorch workflow.

Learn and practice:

- `Dataset` and `DataLoader`
- modular model definitions
- training/validation loops
- optimizers and learning-rate schedulers
- checkpointing and early stopping
- mixed precision training
- GPU profiling and memory awareness
- class imbalance handling
- AUROC, AUPRC, Brier score, calibration
- subgroup evaluation and fairness checks
- MLflow experiment tracking
- distributed/multi-GPU concepts

Deliverable: reproducible training pipeline with experiment tracking and evaluation.

## Weeks 5-8: Production ML System

Goal: productionize the model from Module 02.

Learn and practice:

- Python packaging
- FastAPI and REST APIs
- input validation
- Docker
- unit and integration testing
- type hints, linting, formatting
- GitHub Actions CI/CD
- model serialization and versioning
- MLflow model registry concepts
- Azure deployment
- container registry concepts
- secrets and identity basics
- logging and observability
- health checks
- data, feature, and prediction drift
- rollback and retraining-trigger concepts

Deliverable: containerized prediction API with automated tests, CI, deployment documentation, and monitoring.

## Weeks 8-10: Clinical Trials and Survival Analysis

Goal: analyze a synthetic clinical trial with time-to-event outcomes.

Learn and practice:

- trial phases and analysis populations
- endpoints and estimands
- censoring
- Kaplan-Meier curves
- log-rank tests
- Cox proportional hazards models
- proportional-hazards diagnostics
- time-dependent covariates
- competing risks
- concordance index and survival calibration
- bootstrap validation
- CDISC concepts
- SDTM concepts
- ADaM concepts
- traceability from source data to analysis dataset

Deliverable: reproducible synthetic trial analysis with analysis-ready datasets and statistical reporting.

## Weeks 10-12: Clinical LLM Evaluation

Goal: build a small evaluation framework for clinical LLM/RAG systems.

Learn and practice:

- hosted and local LLM interfaces
- Hugging Face basics
- embeddings and vector stores
- RAG
- structured outputs
- prompt/version management
- factuality and hallucination evaluation
- retrieval precision/recall
- answer relevance and citation accuracy
- abstention behavior
- robustness and prompt sensitivity
- subgroup and bias evaluation
- safety-oriented evaluation
- latency and cost tracking
- simple tool-use/agent evaluation
- limited LangChain/LangGraph exposure

Deliverable: reusable evaluation harness with benchmark results and documented limitations.

## Throughout

Every substantive module should progressively strengthen:

- Git and Linux fluency
- clean repository structure
- dependency management
- testing
- CI/CD
- documentation
- reproducibility
- architecture diagrams
- resume-ready evidence

## Short Technical Labs

The `06-technical-labs` section will cover targeted exposure without displacing the core projects:

- Spark/PySpark and Databricks
- Terraform and infrastructure-as-code basics
- AWS/GCP concepts alongside Azure
- Kubernetes basics
- LangChain/LangGraph basics
- healthcare data standards and terminology
