import requests
import pandas as pd
from pathlib import Path
import time

COORDENADAS = {
    'santiago': (19.4517, -70.6970),
    'santodomingo': (18.4861, -69.9312),
    'puntacana': (18.5601, -68.3725),
    'barahona': (18.2086, -71.1005),
    'montecristi': (19.8467, -71.6422),
    'jimani': (18.4911, -71.8508),
    'romana': (18.4273, -68.9728),
    'lasamericas': (18.4297, -69.6686),
    'cabrera': (19.6400, -69.9000),
    'catey': (19.4067, -69.6700),
    'bayaguana': (18.7500, -69.6333),
    'higuero': (18.5833, -69.3833),
    'union': (19.6167, -71.3333),
    'arroyo': (19.0000, -70.0000),
    'sabana': (18.7500, -69.8833)
}

def descargar_datos_meteo():
    Path('datos_meteo').mkdir(exist_ok=True)
    
    print("Descargando datos meteorológicos de Open-Meteo...")
    print("Nota: Con espera de 3 segundos entre peticiones para evitar rate limit")
    print("=" * 60)
    
    exitosos = 0
    fallidos = []
    
    for i, (ciudad, (lat, lon)) in enumerate(COORDENADAS.items(), 1):
        print(f"\n[{i}/{len(COORDENADAS)}] {ciudad}...")
        
        # Verificar si ya existe
        archivo_salida = f"datos_meteo/{ciudad}_meteo.csv"
        if Path(archivo_salida).exists():
            print(f"  ⏭ Ya existe, saltando...")
            exitosos += 1
            continue
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2010-01-01",
            "end_date": "2025-09-28",
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "windspeed_10m_max",
                "pressure_msl_mean",
                "et0_fao_evapotranspiration"
            ],
            "timezone": "America/Santo_Domingo"
        }
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                df = pd.DataFrame(data['daily'])
                df['date'] = pd.to_datetime(df['time'])
                df = df.drop('time', axis=1)
                
                # Features derivados
                df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
                df['pressure_change'] = df['pressure_msl_mean'].diff()
                
                df.to_csv(archivo_salida, index=False)
                print(f"  ✓ {len(df)} días descargados")
                exitosos += 1
                
                # CLAVE: Esperar 3 segundos entre peticiones
                if i < len(COORDENADAS):  # No esperar después del último
                    print(f"  ⏳ Esperando 3 segundos...")
                    time.sleep(3)
                
                break  # Salir del while si fue exitoso
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_count += 1
                    espera = 10 * retry_count  # Espera exponencial
                    if retry_count < max_retries:
                        print(f"  ⚠ Rate limit. Reintento {retry_count}/{max_retries} en {espera}s...")
                        time.sleep(espera)
                    else:
                        print(f"  ✗ Rate limit - máximo de reintentos alcanzado")
                        fallidos.append(ciudad)
                        break
                else:
                    print(f"  ✗ Error HTTP {e.response.status_code}")
                    fallidos.append(ciudad)
                    break
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                fallidos.append(ciudad)
                break
    
    # Resumen
    print("\n" + "=" * 60)
    print(f"RESUMEN:")
    print(f"  Exitosos: {exitosos}/{len(COORDENADAS)}")
    print(f"  Fallidos: {len(fallidos)}")
    
    if fallidos:
        print(f"\nCiudades que fallaron:")
        for ciudad in fallidos:
            print(f"  - {ciudad}")
        print("\nPuedes ejecutar el script de nuevo. Saltará las que ya descargó.")
    else:
        print("\n✓ Todas las descargas completadas")

if __name__ == "__main__":
    descargar_datos_meteo()