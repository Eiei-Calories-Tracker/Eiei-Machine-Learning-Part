FROM apache/airflow:2.9.1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"
