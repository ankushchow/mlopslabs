"""Exercise actual Airflow task instances for both approval and rejection."""
import json
import sys
from pathlib import Path

import pendulum

sys.path.insert(0, "/opt/airflow/dags")
from penguin_clustering import dag
from clustering.pipeline import run_directory

results = []
for threshold, expected in [(0.35, "success"), (1.0, "failed")]:
    run = dag.test(execution_date=pendulum.now("UTC"), run_conf={"min_silhouette": threshold})
    assert run.state == expected, (run.state, expected)
    states = {f"{ti.task_id}[{ti.map_index}]": ti.state for ti in run.get_task_instances()}
    assert sum(key.startswith("train_candidate[") and value == "success" for key, value in states.items()) == 7
    directory = run_directory("/opt/airflow/working_data/penguins_gmm", run.run_id)
    if expected == "failed":
        assert states["quality_gate[-1]"] == "failed"
        assert states["publish_model[-1]"] == "upstream_failed"
        assert states["predict[-1]"] == "upstream_failed"
        assert not (directory / "manifest.json").exists()
    else:
        assert (directory / "predictions.csv").exists()
    results.append({"run_id": run.run_id, "threshold": threshold, "state": run.state,
                    "tasks": states, "artifact_directory": str(directory)})
Path("/opt/airflow/working_data/penguins_gmm/dag_validation.json").write_text(json.dumps(results, indent=2) + "\n")
print("PASS: real Airflow task execution verified approval and blocked publication")
