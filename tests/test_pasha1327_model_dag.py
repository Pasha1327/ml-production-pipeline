import pytest
from airflow.models import DagBag


@pytest.fixture
def dag_bag():
    return DagBag(dag_folder="dags/Pasha1327/", include_examples=False)


def test_model_dag_loading(dag_bag):
    """Тестирование что Model DAG загружается без ошибок"""
    assert len(dag_bag.import_errors) == 0, f"Ошибки импорта: {dag_bag.import_errors}"
    assert "model_training_pasha1327" in dag_bag.dag_ids


def test_model_dag_structure(dag_bag):
    """Тестирование структуры Model DAG"""
    dag = dag_bag.dags.get("model_training_pasha1327")
    assert dag is not None, "DAG не должен быть None"
    assert len(dag.tasks) >= 1, "Должна быть хотя бы одна задача"

    task = dag.get_task("train_model_docker")
    assert task.task_id == "train_model_docker"
    assert "DockerOperator" in str(type(task)), "Задача должна быть DockerOperator"


def test_model_dag_properties(dag_bag):
    """Тестирование свойств Model DAG"""
    dag = dag_bag.dags.get("model_training_pasha1327")

    assert dag.owner == "pasha1327", f"DAG имеет неверного владельца: {dag.owner}"
    assert dag.schedule_interval == "@weekly", "Расписание должно быть weekly"
    assert "pasha1327" in dag.tags, "Должен быть тег с вашим именем"
    assert "docker" in dag.tags, "Должен быть тег docker"
    assert "mlflow" in dag.tags, "Должен быть тег mlflow"


def test_docker_operator_properties(dag_bag):
    """Тестирование свойств DockerOperator"""
    dag = dag_bag.dags.get("model_training_pasha1327")
    task = dag.get_task("train_model_docker")

    # Проверяем важные свойства DockerOperator
    assert task.image == "registry.gitlab.com/glebkuzn/airflow/pasha1327-train-lgbm:v2"
    assert task.auto_remove == "success"
    assert task.force_pull is True

    # Проверяем что передаются правильные переменные окружения
    assert "DB_USER" in task.environment
    assert "DB_PASS" in task.environment
    assert "MLFLOW_TRACKING_URI" in task.environment
    assert task.environment["MLFLOW_TRACKING_URI"] == "http://82.202.137.136:8000"
    assert "{{ conn.postgres_default.login }}" in task.environment["DB_USER"]
    assert "{{ conn.postgres_default.password }}" in task.environment["DB_PASS"]
    assert "{{ conn.postgres_default.host }}" in task.environment["DB_HOST"]
