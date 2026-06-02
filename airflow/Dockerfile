FROM python:3.12-bookworm

WORKDIR /opt/airflow

# Minimal system packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Airflow environment variables
ENV AIRFLOW_HOME=/opt/airflow
# Make the mounted src/ package importable from DAGs (`from src.train import ...`)
ENV PYTHONPATH=/opt/airflow
# Recommended for local dev: prevent DAGs from triggering immediately upon detection
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true
ENV AIRFLOW__CORE__LOAD_EXAMPLES=false
# Fernet key is optional for local development with dummy/local passwords
ENV AIRFLOW__CORE__FERNET_KEY=''
# Bypasses complex Role-Based Access Control (RBAC) for local development
ENV AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=true
# Faster refresh interval for DAG changes during development
ENV AIRFLOW__DAG_PROCESSOR__REFRESH_INTERVAL=3
# Prevents Python from cluttering the container with .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Install Airflow + ML libraries
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Start Airflow
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

CMD ["/startup.sh"]
