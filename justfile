# Run `just` to list recipes.

default:
    @just --list

# --- Stack lifecycle ---

up:
    docker compose up --build

up-d:
    docker compose up --build -d

down:
    docker compose down

# Stop containers and remove named volumes (mlflow-data, airflow-home)
down-v:
    docker compose down -v

logs:
    docker compose logs -f airflow

mlflow-logs:
    docker compose logs -f mlflow

ps:
    docker compose ps

# --- DAG operations ---

trigger:
    docker compose exec airflow airflow dags trigger telco_churn_training_pipeline

list-dags:
    docker compose exec airflow airflow dags list

clean:
    rm -rf data/processed
