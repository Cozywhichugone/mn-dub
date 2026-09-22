FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
RUN pip install --no-cache-dir faster-whisper
COPY . .
ENV HF_HOME=/app/data/models PYTHONUNBUFFERED=1 PORT=8080
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 0 wsgi:app
