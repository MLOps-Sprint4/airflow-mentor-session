"""
Telco churn training pipeline (Airflow 3 + MLflow).

Orchestrates a full ML lifecycle:

    prepare_data -> train_model (fan-out: logistic / rf / xgb)
                 -> select_best (quality gate)
                 -> approve_promotion (human-in-the-loop)
                 -> register_champion (MLflow Model Registry + `champion` alias)

Teaching points
----------------
* Authoring imports come from the stable ``airflow.sdk`` namespace (Airflow 3).
* Core operators live in the ``standard`` provider in Airflow 3.
* Heavy ML libraries (mlflow, sklearn, xgboost, pandas) are imported INSIDE the
  task bodies, not at module top-level. DAG files are parsed by the
  dag-processor; top-level heavy imports slow parsing and risk timeouts. They
  only need to load on the worker at execution time.
* Small metadata (run ids, metric values, file paths) travels via XCom; the
  actual datasets are written to a shared volume as parquet. Each training run
  logs train/eval datasets with ``mlflow.data.from_pandas`` + ``log_input``.
* ``register_champion`` declares the champion model as an Asset *outlet*, which
  event-triggers the separate ``telco_churn_batch_scoring`` DAG.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.sdk import Asset, Param, dag, get_current_context, task
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.hitl import ApprovalOperator

# Shared asset between this producer DAG and the scoring consumer DAG.
# When register_champion succeeds, Airflow emits an update event for this asset.
CHAMPION_MODEL_ASSET = Asset(
    uri="mlflow://TelcoChurnModel@champion",
    name="telco_churn_champion_model",
)

# Models to train in parallel via dynamic task mapping.
MODEL_CONFIGS = [
    {"model_type": "logistic", "params": {}},
    {"model_type": "rf", "params": {"n_estimators": 200, "max_depth": 12}},
    {"model_type": "xgb", "params": {"n_estimators": 200, "learning_rate": 0.05}},
]

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="telco_churn_training_pipeline",
    description="Train, evaluate, gate, and register a Telco churn model with MLflow",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ml", "mlflow", "training", "telco-churn"],
    params={
        "data_path": Param(
            "/opt/airflow/data/telco_churn_full.csv",
            type="string",
            description="Path to the training CSV on the Airflow volume",
        ),
        "experiment_name": Param("telco-churn-airflow", type="string"),
        "registered_model_name": Param("TelcoChurnModel", type="string"),
        "champion_alias": Param(
            "champion",
            type="string",
            description="MLflow model alias assigned to the promoted version",
        ),
        "primary_metric": Param(
            "f1_score",
            type="string",
            enum=["f1_score", "roc_auc", "accuracy", "precision", "recall"],
        ),
        "min_metric_threshold": Param(0.5, type="number"),
    },
)
def telco_churn_training_pipeline():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # Fail fast if the MLflow tracking server is unreachable before doing work.
    preflight = BashOperator(
        task_id="preflight_mlflow",
        bash_command=(
            'echo "Checking MLflow at $MLFLOW_TRACKING_URI" && '
            'python -c "import urllib.request, os; '
            "urllib.request.urlopen(os.environ['MLFLOW_TRACKING_URI'] + '/health')\" && "
            'echo "MLflow OK"'
        ),
    )

    @task
    def prepare_data() -> dict:
        """Load + preprocess + split, persisting parquet to the shared volume."""
        import os

        from src.data_preprocessing import load_data, preprocess, split_data

        params = get_current_context()["params"]
        data_path = params["data_path"]

        out_dir = "/opt/airflow/data/processed"
        os.makedirs(out_dir, exist_ok=True)

        df = load_data(data_path)
        X, y, _scaler, _features = preprocess(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        paths = {
            "x_train": f"{out_dir}/x_train.parquet",
            "x_test": f"{out_dir}/x_test.parquet",
            "y_train": f"{out_dir}/y_train.parquet",
            "y_test": f"{out_dir}/y_test.parquet",
            "data_source": data_path,
        }
        X_train.to_parquet(paths["x_train"])
        X_test.to_parquet(paths["x_test"])
        y_train.to_frame("Churn").to_parquet(paths["y_train"])
        y_test.to_frame("Churn").to_parquet(paths["y_test"])

        print(f"Prepared data: train={X_train.shape}, test={X_test.shape}")
        return paths

    @task
    def train_model(model_config: dict, data_paths: dict) -> dict:
        """Train one model; log datasets, params, metrics, model, plots to MLflow."""
        import os

        import mlflow
        import mlflow.data
        import mlflow.sklearn
        import mlflow.xgboost
        import pandas as pd

        from src.train import build_model, evaluate, log_plots

        params = get_current_context()["params"]
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(params["experiment_name"])

        X_train = pd.read_parquet(data_paths["x_train"])
        X_test = pd.read_parquet(data_paths["x_test"])
        y_train = pd.read_parquet(data_paths["y_train"])["Churn"]
        y_test = pd.read_parquet(data_paths["y_test"])["Churn"]

        model, model_params, flavour = build_model(
            model_config["model_type"], model_config["params"]
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        # MLflow 3 dataset tracking: schema, digest, profile, and source lineage.
        # https://mlflow.org/docs/latest/ml/dataset/
        train_df = X_train.copy()
        train_df["Churn"] = y_train.values
        eval_df = X_test.copy()
        eval_df["Churn"] = y_test.values
        data_source = data_paths.get("data_source", data_paths["x_train"])

        train_dataset = mlflow.data.from_pandas(
            train_df,
            source=data_source,
            name="telco_churn_train",
            targets="Churn",
        )
        eval_dataset = mlflow.data.from_pandas(
            eval_df,
            source=data_source,
            name="telco_churn_test",
            targets="Churn",
        )

        with mlflow.start_run(run_name=f"{model_config['model_type']}-airflow") as run:
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(eval_dataset, context="evaluation")

            mlflow.set_tag("model_type", model_config["model_type"])
            mlflow.set_tag("run_source", "airflow")
            mlflow.log_params(model_params)

            log_fn = (
                mlflow.sklearn.log_model
                if flavour == "sklearn"
                else mlflow.xgboost.log_model
            )
            model_info = log_fn(model, name="model", input_example=X_test.iloc[:5])
            log_plots(model, X_test, y_test)

            # Link evaluation metrics to the logged model and eval dataset (MLflow 3).
            mlflow.log_metrics(
                metrics,
                model_id=model_info.model_id,
                dataset=eval_dataset,
            )

            run_id = run.info.run_id

        primary = params["primary_metric"]
        print(f"{model_config['model_type']}: {primary}={metrics[primary]:.4f} (run {run_id})")
        return {
            "run_id": run_id,
            "model_type": model_config["model_type"],
            "metric": metrics[primary],
        }

    @task
    def select_best(results: list[dict]) -> dict:
        """Pick the best run by primary metric and enforce a quality gate."""
        params = get_current_context()["params"]
        best = max(results, key=lambda r: r["metric"])
        threshold = float(params["min_metric_threshold"])
        if best["metric"] < threshold:
            raise ValueError(
                f"Quality gate failed: best {params['primary_metric']}="
                f"{best['metric']:.4f} < threshold {threshold}. Halting promotion."
            )
        print(f"Best model: {best}")
        return best

    # Human-in-the-loop gate before touching the production registry alias.
    approve_promotion = ApprovalOperator(
        task_id="approve_promotion",
        subject="Promote best Telco churn model to champion?",
        body=(
            "Run {{ ti.xcom_pull(task_ids='select_best')['run_id'] }} "
            "({{ ti.xcom_pull(task_ids='select_best')['model_type'] }}) "
            "passed the quality gate.\n"
            "Approve to register it and set the 'champion' alias."
        ),
        defaults="Approve",
        execution_timeout=timedelta(hours=1), #response_timeout=timedelta(hours=1),
    )

    @task(outlets=[CHAMPION_MODEL_ASSET])
    def register_champion(best: dict) -> dict:
        """Register the best run's model and move the `champion` alias to it."""
        import os

        import mlflow
        from mlflow import MlflowClient

        params = get_current_context()["params"]
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        model_name = params["registered_model_name"]
        model_uri = f"runs:/{best['run_id']}/model"

        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        client = MlflowClient()
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=(
                f"Promoted from Airflow run of a {best['model_type']} model "
                f"(metric={best['metric']:.4f})."
            ),
        )
        champion_alias = params["champion_alias"]
        client.set_registered_model_alias(model_name, champion_alias, result.version)
        client.set_model_version_tag(
            model_name, result.version, "primary_metric", str(best["metric"])
        )

        print(f"Registered {model_name} v{result.version} as '{champion_alias}'")
        return {"model_name": model_name, "version": result.version}

    prepared = prepare_data()
    results = train_model.partial(data_paths=prepared).expand(model_config=MODEL_CONFIGS)
    best = select_best(results)
    registered = register_champion(best)

    start >> preflight >> prepared
    best >> approve_promotion >> registered >> end


telco_churn_training_pipeline()
