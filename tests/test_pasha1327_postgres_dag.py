import pytest
from airflow.models import DagBag


@pytest.fixture
def dag_bag():
    return DagBag(dag_folder="dags/Pasha1327/", include_examples=False)


def test_pasha1327_dags_loading(dag_bag):
    """Тестирование что DAG загружаются без ошибок"""
    # Проверяем что нет ошибок импорта
    assert len(dag_bag.import_errors) == 0, f"Ошибки импорта: {dag_bag.import_errors}"

    # Проверяем что DAG загрузился
    assert "postgres_data_processing_pasha1327" in dag_bag.dag_ids


def test_postgres_dag_structure(dag_bag):
    """Тестирование структуры Postgres DAG"""
    dag = dag_bag.dags.get("postgres_data_processing_pasha1327")
    assert dag is not None, "DAG не должен быть None"
    assert len(dag.tasks) >= 1, "Должна быть хотя бы одна задача"

    task = dag.get_task("create_base_table")
    assert task.task_id == "create_base_table"
    assert "PostgresOperator" in str(type(task)), "Задача должна быть PostgresOperator"
    assert task.postgres_conn_id == "postgres_default", (
        "Должен использоваться postgres_default"
    )


def test_dag_has_correct_owner(dag_bag):
    """Тестирование что DAG имеет правильного владельца"""
    dag = dag_bag.dags.get("postgres_data_processing_pasha1327")

    assert dag.owner == "pasha1327", f"DAG имеет неверного владельца: {dag.owner}"


def test_dag_schedule_and_tags(dag_bag):
    """Тестирование расписания и тегов DAG"""
    dag = dag_bag.dags.get("postgres_data_processing_pasha1327")

    assert dag.schedule_interval == "@daily", "Расписание должно быть daily"
    assert "pasha1327" in dag.tags, "Должен быть тег с вашим именем"
    assert "postgres" in dag.tags, "Должен быть тег postgres"


def test_sql_contains_correct_elements(dag_bag):
    """Тестирование что SQL содержит ключевые элементы"""
    dag = dag_bag.dags.get("postgres_data_processing_pasha1327")
    task = dag.get_task("create_base_table")

    # Проверяем что SQL содержит ключевые команды
    sql = task.sql.lower()  # Для case-insensitivity
    assert "drop table if exists prod.pasha1327_base" in sql
    assert "create table prod.pasha1327_base as" in sql
    assert "from dwh.application_train" in sql
    assert "from dwh.bureau" in sql
    assert "from dwh.credit_card_balance" in sql
