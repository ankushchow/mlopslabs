#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs working_data
if [ ! -f .env ]; then
  printf 'AIRFLOW_UID=%s\n' "$(id -u)" > .env
fi
docker compose build airflow-init
docker compose up airflow-init --exit-code-from airflow-init
docker compose up -d airflow-webserver airflow-scheduler
