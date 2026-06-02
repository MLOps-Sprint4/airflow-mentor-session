#!/usr/bin/env bash
set -euo pipefail

# Runs the API server, scheduler, dag-processor and triggerer in a single
# process. Metadata is SQLite under AIRFLOW_HOME (airflow-home volume).
# LocalExecutor is set in docker-compose for parallel mapped tasks.
airflow standalone
