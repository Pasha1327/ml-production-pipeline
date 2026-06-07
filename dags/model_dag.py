from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
import os
from datetime import datetime

default_args = {
    "owner": "pasha1327",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="model_training_pasha1327",
    default_args=default_args,
    description="Обучение модели LightGBM в Docker и сохранение в MLflow",
    schedule_interval="@weekly",
    tags=["docker", "mlflow", "model_training", "pasha1327"],
) as dag:
    create_data_table = PostgresOperator(
        task_id="create_data_table",
        postgres_conn_id="postgres_default",
        sql="""
        DROP TABLE IF EXISTS prod.pasha1327_base;
        
        CREATE TABLE prod.pasha1327_base AS
        WITH application_features AS (
            SELECT sk_id_curr, code_gender, flag_own_car, flag_own_realty,
                cnt_children, amt_income_total, amt_credit, amt_annuity,
                (-days_birth/365.25)::integer as age
            FROM dwh.application_train 
            WHERE amt_income_total IS NOT NULL
        ),
        bureau_features AS (
            SELECT sk_id_curr, COUNT(*) as active_loans
            FROM dwh.bureau WHERE credit_active = 'Active'
            GROUP BY sk_id_curr
        ),
        credit_card_features AS (
            SELECT sk_id_curr, 
                AVG(amt_balance/NULLIF(amt_credit_limit_actual, 0)) as credit_utilization
            FROM dwh.credit_card_balance
            GROUP BY sk_id_curr
        )
        SELECT app.*, 
            COALESCE(bur.active_loans, 0) as bureau_active_loans,
            COALESCE(cc.credit_utilization, 0) as credit_utilization,
            CASE WHEN app.amt_credit > 500000 THEN 1 ELSE 0 END as target
        FROM application_features app
        LEFT JOIN bureau_features bur ON app.sk_id_curr = bur.sk_id_curr
        LEFT JOIN credit_card_features cc ON app.sk_id_curr = cc.sk_id_curr;
        """,
    )

    analyze_table = PostgresOperator(
        task_id="analyze_table",
        postgres_conn_id="postgres_default",
        sql="ANALYZE prod.pasha1327_base;",
    )

    train_model = DockerOperator(
        task_id="train_model_docker",
        image="registry.gitlab.com/glebkuzn/airflow/pasha1327-train-lgbm:v2",
        api_version="auto",
        auto_remove="success",
        force_pull=True,
        docker_url=None,
        network_mode="bridge",
        environment={
            "DB_USER": os.getenv("DB_USER", "{{ conn.postgres_default.login }}"),
            "DB_PASS": os.getenv("DB_PASS", "{{ conn.postgres_default.password }}"),
            "DB_HOST": os.getenv("DB_HOST", "{{ conn.postgres_default.host }}"),
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI", "http://82.202.137.136:8000"
            ),
        },
    )

    create_data_table >> analyze_table >> train_model
