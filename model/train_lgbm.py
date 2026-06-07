import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.lightgbm
from sqlalchemy import create_engine
import os
import logging
from urllib.parse import quote_plus

IMAGE_VERSION = "v2_fixed_20250925"

# Константы
DB_HOST = "82.202.137.136"
DB_NAME = "homecredit"
MLFLOW_TRACKING_URI = "http://82.202.137.136:8000"
TABLE_NAME = "pasha1327_base"
DROP_COLUMNS = ["sk_id_curr", "target"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
FILLNA_VALUE = 0
MLFLOW_EXPERIMENT_NAME = "pasha1327_experiment"
MLFLOW_MODEL_NAME = "lgbm_model"
LGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.1,
    "random_state": 42,
}
OUTPUT_TABLE_NAME = "scores_pasha1327"
DB_SCHEMA = "prod"

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Проверка версии образа
        logger.info("Image version check")
        logger.info(f"Running image version: {IMAGE_VERSION}")
        logger.info("Expected fixes: - No extra quotes in schema")

        # Проверка переменных окружения
        logger.info("Environment variables")
        logger.info(f"DB_USER: {os.getenv('DB_USER', 'NOT_SET')}")
        logger.info(f"DB_SCHEMA from env: {os.getenv('DB_SCHEMA', 'NOT_SET')}")
        logger.info(f"DB_SCHEMA from code: {DB_SCHEMA}")

        # Проверка значения схемы перед использованием
        logger.info("Schema validation")
        logger.info(f"DB_SCHEMA value: '{DB_SCHEMA}'")
        logger.info(f"DB_SCHEMA type: {type(DB_SCHEMA)}")
        logger.info(f"DB_SCHEMA repr: {repr(DB_SCHEMA)}")
        logger.info(f"DB_SCHEMA length: {len(DB_SCHEMA)}")

        # Проверка на лишние кавычки
        if "'" in DB_SCHEMA:
            raise ValueError(f"SCHEMA contains extra quotes: '{DB_SCHEMA}'")
        if DB_SCHEMA != "prod":
            raise ValueError(f"SCHEMA is not 'prod': '{DB_SCHEMA}'")

        logger.info("Schema validation passed - no extra quotes")

        logger.info("Starting LightGBM model training for pasha1327")

        # 1. Подключение к БД
        db_user = os.getenv("DB_USER", "test_user")
        db_password = os.getenv("DB_PASS", "test_pass")
        db_host = os.getenv("DB_HOST", "82.202.137.136")
        if not db_user or not db_password or not db_host:
            raise ValueError("DB_USER, DB_PASS or DB_HOST not set")

        # Кодируем пароль для обработки спецсимволов
        encoded_password = quote_plus(db_password)
        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{encoded_password}@{DB_HOST}/{DB_NAME}"
        )
        logger.info("Database connection established")

        # 2. Загрузка данных
        query = f"SELECT * FROM {DB_SCHEMA}.{TABLE_NAME}"
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded data shape: {df.shape}")

        # 3. Подготовка данных
        X = df.drop(DROP_COLUMNS, axis=1, errors="ignore")
        y = df["target"]

        # Конвертируем категориальные переменные
        X = pd.get_dummies(X)
        X = X.fillna(FILLNA_VALUE)

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info("Data preprocessing completed")

        # 4. Настройка MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("MLflow configured")

        # 5. Обучение модели
        with mlflow.start_run():
            logger.info("Starting MLflow run")

            # Создаем и обучаем LightGBM
            model = LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_train, y_train)
            logger.info("Model training completed")

            # Предсказания и метрики
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Логируем параметры
            mlflow.log_param("n_estimators", LGBM_PARAMS["n_estimators"])
            mlflow.log_param("max_depth", LGBM_PARAMS["max_depth"])
            mlflow.log_param("random_state", LGBM_PARAMS["random_state"])
            mlflow.log_param("learning_rate", LGBM_PARAMS["learning_rate"])

            # Логируем метрики
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))

            # Логируем модель
            try:
                mlflow.lightgbm.log_model(model, MLFLOW_MODEL_NAME)
                logger.info("Model binary logged to MLflow")
            except Exception as e:
                logger.warning(f"Could not save model binary to MLflow: {str(e)}")
                logger.info("Only metrics and parameters will be logged")
            logger.info(f"Model logged to MLflow with accuracy: {accuracy:.3f}")

            # 6. Сохранение предсказаний
            predictions = model.predict(X)
            df_predictions = df[["sk_id_curr"]].copy()
            df_predictions["prediction"] = predictions
            df_predictions["probability"] = model.predict_proba(X)[:, 1]
            df_predictions["model_version"] = mlflow.active_run().info.run_id

            # ДОБАВИТЬ: Проверка перед сохранением
            logger.info("Before saving predictions")
            logger.info(f"Table name: {OUTPUT_TABLE_NAME}")
            logger.info(f"Schema: {DB_SCHEMA}")
            logger.info(f"Full table path: {DB_SCHEMA}.{OUTPUT_TABLE_NAME}")

            df_predictions.to_sql(
                OUTPUT_TABLE_NAME,
                engine,
                schema=DB_SCHEMA,
                if_exists="replace",
                index=False,
            )
            logger.info(f"Predictions saved to {DB_SCHEMA}.{OUTPUT_TABLE_NAME}")

        logger.info("Model training pipeline completed successfully!")
        logger.info(f"Image version {IMAGE_VERSION} executed correctly!")

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        logger.error(f"Image version: {IMAGE_VERSION}")
        raise


if __name__ == "__main__":
    main()
