FROM python:3.11-slim

# Instalamos curl para bajar los modelos del Release
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos el código primero
COPY . .

# Instalamos dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Creamos la carpeta de modelos y descargamos los archivos desde tu Release
# Sustituye 'v1.0' y los nombres si cambian en tu GitHub
RUN mkdir -p modelos && \
    curl -L https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/modelo_grupo_1_norte_cibao_hibrido.h5 -o modelos/modelo_grupo_1_norte_cibao_hibrido.h5 && \
    curl -L https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/modelo_grupo_2_sur_seco_hibrido.h5 -o modelos/modelo_grupo_2_sur_seco_hibrido.h5 && \
    curl -L https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/modelo_grupo_3_este_capital_hibrido.h5 -o modelos/modelo_grupo_3_este_capital_hibrido.h5

# Puerto para Railway / Cloud Run
EXPOSE 8080

CMD ["streamlit", "run", "app-mejorada-debug.py", "--server.port=8080", "--server.address=0.0.0.0"]