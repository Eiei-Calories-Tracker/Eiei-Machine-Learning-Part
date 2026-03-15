FROM ghcr.io/mlflow/mlflow:v2.10.2

USER root
RUN pip install --no-cache-dir psycopg2-binary

RUN mkdir -p /mlruns && chown -R 1000:1000 /mlruns
WORKDIR /mlruns
USER 1000
