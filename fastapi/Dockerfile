FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set PYTHONPATH to include the parent Eiei-Machine-Learning-Part dir if needed
# But we'll copy src into the fastapi container or mount it.
# For now, let's assume we copy src into /app/src
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
