FROM python:3.11-slim

# Instalamos curl para descargar desde el Release
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos código y requerimientos
COPY . .

# Instalamos dependencias (TensorFlow, Streamlit, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Creamos la carpeta y descargamos TODO lo necesario del Release
RUN mkdir -p modelos && \
    # Modelos H5 (~7.19 MB cada uno)
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/modelo_grupo_1_norte_cibao_hibrido.h5" -o modelos/modelo_grupo_1_norte_cibao_hibrido.h5 && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/modelo_grupo_2_sur_seco_hibrido.h5" -o modelos/modelo_grupo_2_sur_seco_hibrido.h5 && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/modelo_grupo_3_este_capital_hibrido.h5" -o modelos/modelo_grupo_3_este_capital_hibrido.h5 && \
    # Metadatos PKL (~7.38 KB cada uno)
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/metadata_grupo_1_norte_cibao_hibrido.pkl" -o modelos/metadata_grupo_1_norte_cibao_hibrido.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/metadata_grupo_2_sur_seco_hibrido.pkl" -o modelos/metadata_grupo_2_sur_seco_hibrido.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/metadata_grupo_3_este_capital_hibrido.pkl" -o modelos/metadata_grupo_3_este_capital_hibrido.pkl && \
    # Calibradores PKL (816 Bytes)
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/calibrador_grupo_1_norte_cibao.pkl" -o modelos/calibrador_grupo_1_norte_cibao.pkl && \
    # Sarimax PKL (Estos son pesados: 422 MB cada uno)
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/models-v1/sarimax_grupo_1_norte_cibao_hibrido.pkl" -o modelos/sarimax_grupo_1_norte_cibao_hibrido.pkl

# Exponemos el puerto de Railway
EXPOSE 8080

# Comando para arrancar Streamlit
CMD ["streamlit", "run", "app-mejorada-debug.py", "--server.port=8080", "--server.address=0.0.0.0"]