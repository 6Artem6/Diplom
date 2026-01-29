FROM python:3.13-slim

# Общая рабочая директория
WORKDIR /app

# Копируем только requirements для тестов
COPY ./tests/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем код приложения и тесты
COPY ./sitegpt /app/sitegpt
COPY ./tests /app/tests

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Запуск pytest по директории tests
CMD ["pytest", "-v", "tests"]
