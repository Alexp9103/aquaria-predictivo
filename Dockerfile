FROM python:3.11-slim

# Dependencias del sistema
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Pull de archivos LFS (modelos)
RUN git lfs pull

# Python deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "app-mejorada-debug.py", "--server.port=8080", "--server.address=0.0.0.0"]
