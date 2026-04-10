# Math Evolution Agent — Published Research Repository
# Supports linux/amd64 and linux/arm64

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY code/ ./code/

RUN mkdir -p /app/data /app/checkpoints

ARG BUILD_VERSION=dev
ENV BUILD_VERSION=${BUILD_VERSION}
ENV RUNNING_IN_DOCKER=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/code
