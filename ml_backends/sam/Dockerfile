FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY ml_backends/sam/requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY ml_backends ./ml_backends
COPY ml_backends/sam/entrypoint.sh /usr/local/bin/sam-backend-entrypoint
RUN chmod +x /usr/local/bin/sam-backend-entrypoint

EXPOSE 9090

CMD ["sam-backend-entrypoint"]
