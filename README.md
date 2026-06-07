# ML Production Pipeline: Airflow + PostgreSQL + Docker + MLflow

Продакшен ML-пайплайн для обучения модели кредитного скоринга. Оркестрация через Apache Airflow, обучение в Docker-контейнере, трекинг экспериментов в MLflow, данные в PostgreSQL.

## Архитектура

```
PostgreSQL (dwh) → Airflow DAG → Docker Container → MLflow
     ↓                                  ↓
  Исходные                         LightGBM модель
  данные                           + предсказания → PostgreSQL (prod)
```

## Стек

Python · LightGBM · Apache Airflow · PostgreSQL · Docker · MLflow · SQLAlchemy · pandas · GitLab CI

## Структура

```
├── dags/
│   ├── postgres_dag.py       # DAG: фиче-инжиниринг в PostgreSQL
│   └── model_dag.py          # DAG: обучение модели в Docker → MLflow
├── Pasha1327/
│   ├── train_lgbm.py         # Скрипт обучения LightGBM
│   ├── Dockerfile            # Docker-образ для обучения
│   └── requirements.txt
└── tests/
```

## DAG 1 — Подготовка данных (postgres_dag)

Фиче-инжиниринг поверх датасета Home Credit Default Risk (Kaggle). SQL-запрос с тремя джойнами:

- `dwh.application_train` — базовые данные заявок (возраст, доход, сумма кредита)
- `dwh.bureau` — количество активных кредитов из бюро
- `dwh.credit_card_balance` — утилизация кредитного лимита

Результат сохраняется в `prod.pasha1327_base`. Расписание: `@daily`.

## DAG 2 — Обучение модели (model_dag)

Три задачи с зависимостями:

```
create_data_table >> analyze_table >> train_model (DockerOperator)
```

`train_lgbm.py` внутри контейнера:
1. Загружает данные из `prod.pasha1327_base`
2. Обучает `LGBMClassifier` (100 деревьев, max_depth=10)
3. Логирует параметры, метрики и артефакт модели в MLflow
4. Сохраняет предсказания и вероятности в `prod.scores_pasha1327`

Расписание: `@weekly`.

## MLflow

Каждый запуск логирует:
- Параметры: `n_estimators`, `max_depth`, `learning_rate`, `random_state`
- Метрики: `accuracy`, `train_samples`, `test_samples`
- Артефакт: бинарник модели (`lgbm_model`)
- `run_id` сохраняется вместе с предсказаниями для воспроизводимости

## Docker

```dockerfile
FROM python:3.9-slim
# libgomp1 — системная зависимость LightGBM
RUN apt-get install -y libgomp1
COPY requirements.txt train_lgbm.py .
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "train_lgbm.py"]
```

Образ версионируется (`v2_fixed_20250925`) и хранится в GitLab Container Registry. `force_pull=True` в DAG гарантирует запуск актуальной версии.

## Данные

Home Credit Default Risk — реальный Kaggle-датасет для задачи кредитного скоринга. Целевая переменная: `amt_credit > 500 000`.

---

*Проект выполнен в рамках подготовки к DS/ML позиции. Код проходил code review, CI-проверки и реальное выполнение в продакшен-окружении Airflow.*
