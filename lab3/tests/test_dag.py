from pathlib import Path

from airflow.models import DagBag
from airflow.utils.dag_cycle_tester import check_cycle


def test_dag_import_and_dependencies():
    bag = DagBag(dag_folder=str(Path(__file__).resolve().parents[1] / "dags"), include_examples=False)
    assert bag.import_errors == {}, bag.import_errors
    dag = bag.get_dag("airflow_lab1_penguin_clustering")
    assert dag is not None
    check_cycle(dag)
    assert set(dag.task_ids) == {"validate_data", "preprocess", "train_candidate", "select_model", "quality_gate", "publish_model", "predict"}
    assert dag.get_task("publish_model").upstream_task_ids == {"quality_gate"}
    assert dag.get_task("predict").upstream_task_ids == {"publish_model"}
    assert dag.get_task("train_candidate").upstream_task_ids == {"preprocess"}
    assert not dag.catchup and dag.max_active_runs == 1
