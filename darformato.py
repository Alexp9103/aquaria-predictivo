#!/usr/bin/env python3
"""
Script para procesar TODOS los archivos Excel de ONAMET
y generar formatoadecuado/
"""

import pandas as pd
import re
from pathlib import Path

# Crear carpeta de salida
Path('formatoadecuado').mkdir(exist_ok=True)

# Mapeo de archivos Excel a nombres de salida
# Ajusta según los archivos que tengas en datooriginal/
ARCHIVOS = {
    'SANTO DOMINGO.xls': 'santodomingo',
    'SANTIAGO.xls': 'santiago',
    'BARAHONA.xls': 'barahona',
    'JIMANI.xls': 'jimani',
    'MONTE CRISTI.xls': 'montecristi',
    'CABRERA.xls': 'cabrera',
    'CATEY.xls': 'catey',
    'UNION.xls': 'union',
    'ARROYO BARRIL.xls': 'arroyo',
    'SABANA DE LA MAR.xls': 'sabana',
    'LAS AMERICAS.xls': 'lasamericas',
    'PUNTA CANA.xls': 'puntacana',
    'LA ROMANA.xls': 'romana',
    'HIGUERO.xls': 'higuero',
    'BAYAGUANA.xls': 'bayaguana'
}

def procesar_archivo_onamet(ruta_excel, nombre_salida):
    """
    Procesa un archivo Excel de ONAMET y genera CSV
    """
    print(f"\n📁 Procesando: {ruta_excel.name}")
    
    try:
        # 1. Cargar Excel
        df_raw = pd.read_excel(ruta_excel, header=None)
        
        # 2. Detectar años
        year_rows = df_raw[df_raw[0].astype(str).str.contains("DATOS DIARIOS", na=False)].index
        
        if len(year_rows) == 0:
            print(f"   ⚠️  No se encontraron bloques 'DATOS DIARIOS'")
            return False
        
        years = [int(re.search(r"\d{4}", str(df_raw.iloc[i,0])).group()) for i in year_rows]
        print(f"   Años detectados: {min(years)} - {max(years)} ({len(years)} años)")
        
        # 3. Procesar cada bloque
        dataframes = []
        
        for idx, start_row in enumerate(year_rows):
            year = years[idx]
            
            # Determinar final del bloque
            if idx < len(year_rows) - 1:
                end_row = year_rows[idx + 1]
            else:
                end_row = len(df_raw)
            
            # Detectar primera fila de datos
            dia_col = df_raw.iloc[start_row:end_row, 0]
            dias_numericos = dia_col[dia_col.apply(lambda x: str(x).isdigit())]
            
            if len(dias_numericos) == 0:
                continue
            
            first_day_row = dias_numericos.index[0]
            
            # Extraer bloque
            block = df_raw.iloc[first_day_row:end_row, :13].copy()
            block.columns = ["DIA","ENE","FEB","MAR","ABR","MAY","JUN",
                            "JUL","AGO","SEP","OCT","NOV","DIC"]
            
            # Limpiar datos
            block = block.replace("INAP", pd.NA)
            for col in block.columns[1:]:
                block[col] = pd.to_numeric(block[col], errors="coerce")
            
            # Agregar año
            block["Año"] = year
            
            # Formato largo
            block_long = block.melt(id_vars=["DIA","Año"], var_name="Mes", value_name="Precip_mm")
            
            # Mapear meses
            mapa_meses = {"ENE":1,"FEB":2,"MAR":3,"ABR":4,"MAY":5,"JUN":6,
                         "JUL":7,"AGO":8,"SEP":9,"OCT":10,"NOV":11,"DIC":12}
            block_long["Mes"] = block_long["Mes"].map(mapa_meses)
            
            # Construir fecha
            block_long["Fecha"] = pd.to_datetime(
                dict(year=block_long["Año"], month=block_long["Mes"], day=block_long["DIA"]),
                errors="coerce"
            )
            
            # Rellenar NaN con 0
            block_long["Precip_mm"] = block_long["Precip_mm"].fillna(0)
            
            # Solo fechas válidas
            block_long = block_long.dropna(subset=["Fecha"])
            
            dataframes.append(block_long[["Fecha","Precip_mm"]])
        
        # 4. Concatenar
        if len(dataframes) == 0:
            print(f"   ❌ No se pudo extraer datos")
            return False
        
        dataset = pd.concat(dataframes).reset_index(drop=True)
        
        # 5. Guardar
        archivo_salida = Path('formatoadecuado') / f"{nombre_salida}_diario_2010_2025.csv"
        dataset.to_csv(archivo_salida, index=False)
        
        print(f"   ✅ Guardado: {archivo_salida.name}")
        print(f"      Total registros: {len(dataset):,}")
        print(f"      Rango: {dataset['Fecha'].min().date()} a {dataset['Fecha'].max().date()}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🌧️  PROCESADOR DE DATOS ONAMET")
    print("="*70)
    
    carpeta_origen = Path('datooriginal')
    
    if not carpeta_origen.exists():
        print(f"\n❌ ERROR: Carpeta '{carpeta_origen}' no existe")
        exit(1)
    
    # Listar archivos disponibles
    archivos_excel = list(carpeta_origen.glob('*.xls')) + list(carpeta_origen.glob('*.xlsx'))
    print(f"\n📂 Archivos Excel encontrados: {len(archivos_excel)}")
    
    exitosos = 0
    fallidos = 0
    
    # Procesar archivos configurados
    for nombre_archivo, nombre_ciudad in ARCHIVOS.items():
        ruta_completa = carpeta_origen / nombre_archivo
        
        if ruta_completa.exists():
            if procesar_archivo_onamet(ruta_completa, nombre_ciudad):
                exitosos += 1
            else:
                fallidos += 1
        else:
            print(f"\n⚠️  No encontrado: {nombre_archivo}")
            # Intentar variantes
            for archivo in archivos_excel:
                if nombre_ciudad.upper() in archivo.stem.upper():
                    print(f"   Probando: {archivo.name}")
                    if procesar_archivo_onamet(archivo, nombre_ciudad):
                        exitosos += 1
                    else:
                        fallidos += 1
                    break
    
    # Resumen
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    print(f"✅ Procesados exitosamente: {exitosos}")
    print(f"❌ Fallidos: {fallidos}")
    
    # Listar archivos generados
    archivos_generados = list(Path('formatoadecuado').glob('*.csv'))
    print(f"\n📁 Archivos en formatoadecuado/: {len(archivos_generados)}")
    for archivo in sorted(archivos_generados):
        tamano = archivo.stat().st_size / 1024  # KB
        print(f"   • {archivo.name} ({tamano:.1f} KB)")
    
    print("\n✅ Proceso completado")