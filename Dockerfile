FROM python:3.11-slim

# Instalamos curl para descargar desde el Release
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos código y requerimientos
COPY . .

# Instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Descarga de modelos y calibradores desde GitHub Release
RUN mkdir -p modelos && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/calibrador_grupo_1_norte_cibao.pkl" -o modelos/calibrador_grupo_1_norte_cibao.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/calibrador_grupo_2_sur_seco.pkl" -o modelos/calibrador_grupo_2_sur_seco.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/calibrador_grupo_3_este_capital.pkl" -o modelos/calibrador_grupo_3_este_capital.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/metadata_grupo_1_norte_cibao_hibrido.pkl" -o modelos/metadata_grupo_1_norte_cibao_hibrido.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/metadata_grupo_2_sur_seco_hibrido.pkl" -o modelos/metadata_grupo_2_sur_seco_hibrido.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/metadata_grupo_3_este_capital_hibrido.pkl" -o modelos/metadata_grupo_3_este_capital_hibrido.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/modelo_grupo_1_norte_cibao_hibrido.h5" -o modelos/modelo_grupo_1_norte_cibao_hibrido.h5 && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/modelo_grupo_2_sur_seco_hibrido.h5" -o modelos/modelo_grupo_2_sur_seco_hibrido.h5 && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/modelo_grupo_3_este_capital_hibrido.h5" -o modelos/modelo_grupo_3_este_capital_hibrido.h5 && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/sarimax_grupo_1_norte_cibao_hibrido.pkl" -o modelos/sarimax_grupo_1_norte_cibao_hibrido.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/sarimax_grupo_2_sur_seco_hibrido.pkl" -o modelos/sarimax_grupo_2_sur_seco_hibrido.pkl && \
    curl -fSL "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos/sarimax_grupo_3_este_capital_hibrido.pkl" -o modelos/sarimax_grupo_3_este_capital_hibrido.pkl

# Exponemos el puerto de Railway
EXPOSE 8080

# Comando para arrancar Streamlit
CMD ["streamlit", "run", "app-mejorada-debug.py", "--server.port=8080", "--server.address=0.0.0.0"]