"""Lab 3 (based on Airflow Lab 1): validated, mapped, quality-gated penguin clustering."""
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.python import get_current_context

from clustering import pipeline

BASE = Path(__file__).resolve().parents[1]


@dag(
    dag_id="airflow_lab1_penguin_clustering",
    description="Compare seven Gaussian Mixture models and publish only a quality-approved pipeline",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="0 6 * * *",
    catchup=False,
    max_active_runs=1,
    max_active_tasks=4,
    default_args={"owner": "ankush", "retries": 1, "retry_delay": timedelta(seconds=15),
                  "execution_timeout": timedelta(minutes=5)},
    params={"min_silhouette": Param(0.35, type="number", minimum=0, maximum=1,
                                   description="Use 1.0 to demonstrate rejection")},
    tags=["airflow-lab1", "clustering", "quality-gate"],
)
def penguin_clustering():
    @task
    def validate_data():
        return pipeline.validate_data(BASE / "data/penguins_training.csv", BASE / "working_data/penguins_gmm", get_current_context()["run_id"])

    @task
    def preprocess(directory):
        return pipeline.preprocess(directory)

    @task
    def train_candidate(directory, k):
        return pipeline.train_candidate(directory, k)

    @task
    def select_model(directory, candidates):
        return pipeline.select_model(directory, candidates)

    @task
    def quality_gate(directory):
        try:
            return pipeline.quality_gate(directory, get_current_context()["params"]["min_silhouette"])
        except pipeline.QualityGateError as error:
            raise AirflowFailException(str(error)) from error

    @task
    def publish_model(directory):
        return pipeline.publish_model(directory)

    @task
    def predict(directory):
        return pipeline.predict(directory, BASE / "data/penguins_predict.csv")

    prepared = preprocess(validate_data())
    candidates = train_candidate.partial(directory=prepared).expand(k=pipeline.CANDIDATES)
    selected = select_model(prepared, candidates)
    predict(publish_model(quality_gate(selected)))


dag = penguin_clustering()
