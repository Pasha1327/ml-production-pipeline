from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta

# Конфигурация
NICK = "pasha1327"
SOURCE_TABLE = "dwh.application_train"
TARGET_TABLE = f"prod.{NICK}_base"
CONN_ID = "postgres_default"

default_args = {
    "owner": NICK,
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=f"postgres_data_processing_{NICK}",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["postgres", NICK],
) as dag:
    # Задача для загрузки данных из базы и обработки с использованием Preprocessor
    create_base_table = PostgresOperator(
        task_id="create_base_table",
        postgres_conn_id=CONN_ID,
        sql=f"""
        -- Удаляем старую таблицу, если она существует
        DROP TABLE IF EXISTS {TARGET_TABLE};

        -- Создаём новую таблицу с обработкой данных
        CREATE TABLE {TARGET_TABLE} AS
        WITH application_features AS (
            SELECT sk_id_curr, code_gender, flag_own_car, flag_own_realty,
                cnt_children, amt_income_total, amt_credit, amt_annuity,
                (-days_birth/365.25)::integer as age
            FROM {SOURCE_TABLE} 
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

    # Задача для завершения DAG
    finish = PostgresOperator(
        task_id="finish",
        postgres_conn_id=CONN_ID,
        sql=f"ANALYZE {TARGET_TABLE};",
    )

    # Зависимости
    create_base_table >> finish
