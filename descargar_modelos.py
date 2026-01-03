import os
import requests

BASE_URL = "https://github.com/Alexp9103/aquaria-predictivo/releases/download/modelos"

MODELOS = [
    "modelo_grupo_1_norte_cibao_hibrido.h5",
    "sarimax_grupo_1_norte_cibao_hibrido.pkl",
    "metadata_grupo_1_norte_cibao_hibrido.pkl",
    "calibrador_grupo_1_norte_cibao.pkl",

    "modelo_grupo_2_sur_seco_hibrido.h5",
    "sarimax_grupo_2_sur_seco_hibrido.pkl",
    "metadata_grupo_2_sur_seco_hibrido.pkl",
    "calibrador_grupo_2_sur_seco.pkl",

    "modelo_grupo_3_este_capital_hibrido.h5",
    "sarimax_grupo_3_este_capital_hibrido.pkl",
    "metadata_grupo_3_este_capital_hibrido.pkl",
    "calibrador_grupo_3_este_capital.pkl",
]

DESTINO = "modelos"
os.makedirs(DESTINO, exist_ok=True)

def descargar_modelos():
    for archivo in MODELOS:
        ruta = os.path.join(DESTINO, archivo)

        if os.path.exists(ruta):
            continue

        url = f"{BASE_URL}/{archivo}"
        print(f"⬇️ Descargando {archivo}")

        r = requests.get(url, stream=True)
        r.raise_for_status()

        with open(ruta, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    print("✅ Modelos listos")
