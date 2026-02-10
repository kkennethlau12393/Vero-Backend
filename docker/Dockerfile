FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install deps for psycopg2-binary (usually fine) + basic utilities
RUN pip install -U pip

# Copy project files (for pip install -e .)
COPY pyproject.toml /app/pyproject.toml
COPY app /app/app

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
