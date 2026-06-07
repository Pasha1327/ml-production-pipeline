# Используем базовый образ с Python
FROM python:3.9-slim

LABEL version="v3_fixed_20250925"

# Устанавливаем рабочую директорию
WORKDIR /app

# Обновляем и устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы
COPY requirements.txt .
COPY train_lgbm.py .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем переменные окружения
ENV MLFLOW_TRACKING_URI="http://82.202.137.136:8000"

# Запускаем скрипт
ENTRYPOINT ["python", "train_lgbm.py"]
