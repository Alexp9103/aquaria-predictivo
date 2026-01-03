import pandas as pd
import re

# 1. Cargar todo el Excel sin saltar filas ni poner encabezados
df_raw = pd.read_excel("/home/juanpucmm/modelo predectivo/datooriginal/SANTO DOMINGO.xls", header=None)

# 2. Detectar las filas que contienen "DATOS DIARIOS" y extraer el año con regex
year_rows = df_raw[df_raw[0].astype(str).str.contains("DATOS DIARIOS", na=False)].index
years = [int(re.search(r"\d{4}", str(df_raw.iloc[i,0])).group()) for i in year_rows]

print("Años detectados:", years)

# 3. Procesar cada bloque
dataframes = []

for idx, start_row in enumerate(year_rows):
    year = years[idx]

    # Determinar el final del bloque: siguiente "DATOS DIARIOS" o final del archivo
    if idx < len(year_rows) - 1:
        end_row = year_rows[idx + 1]
    else:
        end_row = len(df_raw)

    # Detectar la primera fila donde la columna 0 es un número (DIA)
    dia_col = df_raw.iloc[start_row:end_row, 0]
    first_day_row = dia_col[dia_col.apply(lambda x: str(x).isdigit())].index[0]

    # Tomar el bloque desde la primera fila de datos hasta end_row
    block = df_raw.iloc[first_day_row:end_row, :13].copy()

    # Renombrar columnas
    block.columns = ["DIA","ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SEP","OCT","NOV","DIC"]

    # Reemplazar INAP → NaN y convertir a numérico
    block = block.replace("INAP", pd.NA)
    for col in block.columns[1:]:
        block[col] = pd.to_numeric(block[col], errors="coerce")

    # Agregar año
    block["Año"] = year

    # Pasar a formato largo
    block_long = block.melt(id_vars=["DIA","Año"], var_name="Mes", value_name="Precip_mm")

    # Mapear meses
    mapa_meses = {"ENE":1,"FEB":2,"MAR":3,"ABR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AGO":8,"SEP":9,"OCT":10,"NOV":11,"DIC":12}
    block_long["Mes"] = block_long["Mes"].map(mapa_meses)

    # Construir fecha
    block_long["Fecha"] = pd.to_datetime(dict(year=block_long["Año"],
                                              month=block_long["Mes"],
                                              day=block_long["DIA"]),
                                         errors="coerce")

    # Reemplazar NaN en precipitación por 0 (días sin datos o INAP)
    block_long["Precip_mm"] = block_long["Precip_mm"].fillna(0)

    # Solo conservar filas con fecha válida
    block_long = block_long.dropna(subset=["Fecha"])

    dataframes.append(block_long[["Fecha","Precip_mm"]])

# 4. Concatenar todos los bloques
dataset = pd.concat(dataframes).reset_index(drop=True)

# 5. Guardar CSV final
dataset.to_csv("/home/juanpucmm/modelo predectivo/formatoadecuado/santodomingo_diario_2010_2025.csv", index=False)

print(dataset.head())
print(dataset.tail())
print("Total de registros:", len(dataset))
