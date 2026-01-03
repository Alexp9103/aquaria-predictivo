#!/usr/bin/env python3
"""
Script de actualización automática diaria
Descarga datos recientes y envía alertas
"""

import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import requests
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/admin_aquaria/logs/update_auto.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

COORDENADAS = {
    'grupo_1_norte_cibao': (19.4, -70.7),
    'grupo_2_sur_seco': (18.0, -71.1),
    'grupo_3_este_capital': (18.5, -69.9)
}

def descargar_datos_recientes(region_key, dias_atras=7):
    """Descarga últimos N días de Open-Meteo"""
    
    if region_key not in COORDENADAS:
        return None
    
    lat, lon = COORDENADAS[region_key]
    
    fecha_fin = datetime.now().date()
    fecha_inicio = fecha_fin - timedelta(days=dias_atras)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(fecha_inicio),
        "end_date": str(fecha_fin),
        "daily": "precipitation_sum",
        "timezone": "America/Santo_Domingo"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' not in data:
            return None
        
        df = pd.DataFrame({
            'fecha': pd.to_datetime(data['daily']['time']),
            'precip': data['daily']['precipitation_sum']
        })
        
        df['precip'] = df['precip'].fillna(0)
        df.set_index('fecha', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error descargando {region_key}: {e}")
        return None

def actualizar_modelo(region_key):
    """Actualiza serie de un modelo específico"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Actualizando {region_key}")
    logger.info(f"{'='*60}")
    
    try:
        # Cargar metadata
        metadata_path = f'modelos/metadata_{region_key}.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        serie_original = metadata['serie']
        logger.info(f"Serie original: {len(serie_original)} semanas → última: {serie_original.index[-1].date()}")
        
        # Cortar datos futuros
        hoy = pd.Timestamp(datetime.now().date())
        serie_limpia = serie_original[serie_original.index < hoy]
        
        # Eliminar ceros al final
        while len(serie_limpia) > 0 and serie_limpia.iloc[-1] == 0:
            serie_limpia = serie_limpia[:-1]
        
        logger.info(f"Serie limpia: {len(serie_limpia)} semanas → última: {serie_limpia.index[-1].date()}")
        
        # Descargar datos recientes
        df_nuevo = descargar_datos_recientes(region_key, dias_atras=30)
        
        if df_nuevo is not None and len(df_nuevo) > 0:
            # Agregar a semanal
            serie_nueva = df_nuevo['precip'].resample('W-SUN').mean() * 7
            serie_nueva = serie_nueva[serie_nueva > 0]
            
            if len(serie_nueva) > 0:
                # Concatenar
                serie_actualizada = pd.concat([serie_limpia, serie_nueva])
                serie_actualizada = serie_actualizada[~serie_actualizada.index.duplicated(keep='last')]
                serie_actualizada = serie_actualizada.sort_index()
                
                # Eliminar ceros finales
                while len(serie_actualizada) > 0 and serie_actualizada.iloc[-1] == 0:
                    serie_actualizada = serie_actualizada[:-1]
                
                logger.info(f"Serie actualizada: {len(serie_actualizada)} semanas → última: {serie_actualizada.index[-1].date()}")
                
                # Actualizar metadata
                metadata['serie'] = serie_actualizada
                metadata['ultima_actualizacion_auto'] = datetime.now().isoformat()
                
                # Guardar
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                logger.info(f"✅ {region_key} actualizado correctamente")
                return True
            else:
                logger.warning(f"No hay datos nuevos válidos para {region_key}")
                return False
        else:
            logger.warning(f"No se pudieron descargar datos para {region_key}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error actualizando {region_key}: {e}")
        return False

def main():
    """Actualizar todos los modelos"""
    
    logger.info("\n" + "="*70)
    logger.info("🚀 INICIO DE ACTUALIZACIÓN AUTOMÁTICA DIARIA")
    logger.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    regiones = [
        'grupo_1_norte_cibao',
        'grupo_2_sur_seco',
        'grupo_3_este_capital'
    ]
    
    resultados = {}
    
    for region in regiones:
        exito = actualizar_modelo(region)
        resultados[region] = exito
    
    # Resumen
    logger.info("\n" + "="*70)
    logger.info("📊 RESUMEN DE ACTUALIZACIÓN")
    logger.info("="*70)
    
    exitosos = sum(resultados.values())
    total = len(resultados)
    
    for region, exito in resultados.items():
        estado = "✅" if exito else "❌"
        logger.info(f"{estado} {region}")
    
    logger.info(f"\nTotal: {exitosos}/{total} actualizaciones exitosas")
    logger.info("="*70 + "\n")
    
    return exitosos == total

if __name__ == "__main__":
    try:
        exito = main()
        exit(0 if exito else 1)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        exit(1)