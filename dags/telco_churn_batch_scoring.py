"""
Telco churn batch scoring (Airflow 3 + MLflow), asset-triggered.

This DAG has NO cron schedule. Instead it is scheduled on the champion model
Asset: it runs automatically whenever the training pipeline registers a new
champion (the producer task declares this asset as an outlet).

It loads ``models:/TelcoChurnModel@champion`` via the MLflow registry and scores
a sample batch, writing predictions to the shared volume.
"""

from __future__ import annotations

import pendulum
from airflow.sdk import Asset, Param, dag, get_current_context, task

# Must match the asset declared (and emitted) by the training pipeline.
CHAMPION_MODEL_ASSET = Asset(
    uri="mlflow://TelcoChurnModel@champion",
    name="telco_churn_champion_model",
)


@dag(
    dag_id="telco_churn_batch_scoring",
    description="Score new customers whenever a new champion model is registered",
    schedule=[CHAMPION_MODEL_ASSET],
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ml", "mlflow", "scoring", "telco-churn"],
    params={
        "registered_model_name": Param("TelcoChurnModel", type="string"),
        "model_alias": Param(
            "champion",
            type="string",
            description="MLflow alias of the model to load for scoring",
        ),
        "scoring_data_path": Param(
            "/opt/airflow/data/telco_churn_scoring_sample.csv",
            type="string",
            description="CSV batch to score on the Airflow volume",
        ),
        "predictions_path": Param(
            "/opt/airflow/data/processed/predictions.csv",
            type="string",
            description="Where to write scoring output on the Airflow volume",
        ),
    },
)
def telco_churn_batch_scoring():

    @task(inlets=[CHAMPION_MODEL_ASSET])
    def score_batch(*, inlet_events) -> str:
        """Load the champion model and score the scoring sample."""
        import os

        import mlflow
        import pandas as pd

        from src.data_preprocessing import load_data, preprocess

        # Inspect metadata about the asset event that triggered this run.
        events = inlet_events[CHAMPION_MODEL_ASSET]
        if events:
            event = events[0]
            print(
                f"Triggered by asset update from DAG={event.source_dag_id} "
                f"task={event.source_task_id} at {event.timestamp}"
            )

        params = get_current_context()["params"]
        model_name = params["registered_model_name"]
        model_alias = params["model_alias"]
        scoring_data_path = params["scoring_data_path"]
        out_path = params["predictions_path"]

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Same preprocessing as training keeps the feature columns aligned.
        df = load_data(scoring_data_path)
        X, _y, _scaler, _features = preprocess(df)
        preds = model.predict(X)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame({"churn_prediction": preds}).to_csv(out_path, index=False)

        churn_rate = float(pd.Series(preds).mean())
        print(f"Scored {len(preds)} rows -> {out_path} (predicted churn rate={churn_rate:.3f})")
        return out_path

    score_batch()


telco_churn_batch_scoring()
