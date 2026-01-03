#!/usr/bin/env python3
"""
Script de Actualización Diaria - ADAPTADO para estructura grupo_X
Se ejecuta vía cron job todos los días a las 3 AM
"""

import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Importar módulo de actualización
import actualizar_datos as actualizador

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

log_dir = Path.home() / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'update_daily.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# MAPEO DE NOMBRES
# ============================================================================

MAPEO_REGIONES = {
    'grupo_1_norte_cibao': 'norte_cibao',
    'grupo_2_sur_seco': 'sur_seco',
    'grupo_3_este_capital': 'este_capital'
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def actualizar_todos_los_modelos():
    """
    Actualiza todos los modelos disponibles en la carpeta modelos/
    """
    logger.info("="*70)
    logger.info("🚀 INICIANDO ACTUALIZACIÓN DIARIA AUTOMÁTICA")
    logger.info(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    try:
        # 1. Cargar modelos existentes
        logger.info("\n📂 Cargando modelos existentes...")
        modelos = {}
        modelos_dir = Path('modelos')
        
        if not modelos_dir.exists():
            logger.error("❌ La carpeta 'modelos/' no existe")
            return False
        
        # Buscar archivos con patrón grupo_*
        for archivo in modelos_dir.glob('metadata_grupo_*.pkl'):
            nombre_completo = archivo.stem.replace('metadata_', '')
            
            try:
                logger.info(f"   Cargando {nombre_completo}...")
                
                # Cargar metadata
                with open(archivo, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Cargar SARIMAX
                with open(f'modelos/sarimax_{nombre_completo}.pkl', 'rb') as f:
                    sarimax = pickle.load(f)
                
                # Cargar LSTM si existe
                lstm = None
                lstm_path = modelos_dir / f'modelo_{nombre_completo}.h5'
                if lstm_path.exists():
                    try:
                        from tensorflow import keras
                        lstm = keras.models.load_model(str(lstm_path))
                        logger.info(f"   ✅ LSTM cargado para {nombre_completo}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ No se pudo cargar LSTM: {e}")
                
                # Guardar en diccionario con nombre mapeado
                region_key = MAPEO_REGIONES.get(nombre_completo, nombre_completo)
                
                modelos[region_key] = {
                    'metadata': metadata,
                    'sarimax': sarimax,
                    'lstm': lstm,
                    'nombre_display': nombre_completo.replace('_', ' ').title(),
                    'nombre_archivo': nombre_completo  # Para guardar después
                }
                
                logger.info(f"   ✅ {nombre_completo} cargado correctamente")
                
            except Exception as e:
                logger.error(f"   ❌ Error cargando {nombre_completo}: {e}")
                continue
        
        if not modelos:
            logger.error("❌ No se cargaron modelos. Abortando actualización.")
            return False
        
        logger.info(f"\n✅ Total modelos cargados: {len(modelos)}")
        
        # 2. Verificar y actualizar cada modelo
        logger.info("\n🔄 Verificando necesidad de actualización...")
        
        for region_key, modelo_dict in modelos.items():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"📍 PROCESANDO: {region_key.upper()}")
                logger.info(f"{'='*60}")
                
                metadata = modelo_dict['metadata']
                nombre_archivo = modelo_dict['nombre_archivo']
                
                # Actualizar datos con Open-Meteo
                serie_original = metadata['serie']
                serie_actualizada, hay_nuevos = actualizador.actualizar_serie_con_openmeteo(
                    serie_original,
                    region_key
                )
                
                if not hay_nuevos:
                    logger.info("ℹ️ No hay datos nuevos para esta región")
                    continue
                
                # Re-entrenar SARIMAX
                logger.info("🧠 Re-entrenando SARIMAX...")
                sarimax_nuevo = actualizador.reentrenar_sarimax(serie_actualizada, metadata)
                
                if sarimax_nuevo is None:
                    logger.warning("⚠️ Error en SARIMAX, manteniendo modelo anterior")
                    sarimax_nuevo = modelo_dict['sarimax']
                
                # Re-entrenar LSTM si existe
                if modelo_dict['lstm'] is not None:
                    logger.info("🧠 Re-entrenando LSTM...")
                    lstm_nuevo = actualizador.reentrenar_lstm(
                        serie_actualizada,
                        modelo_dict['lstm'],
                        metadata.get('scaler'),
                        metadata
                    )
                else:
                    lstm_nuevo = None
                
                # Actualizar metadata
                metadata['serie'] = serie_actualizada
                metadata['ultima_actualizacion'] = datetime.now().isoformat()
                metadata['semanas_totales'] = len(serie_actualizada)
                
                # Guardar modelos con nombres correctos (grupo_X_...)
                logger.info("💾 Guardando modelos actualizados...")
                
                # Guardar SARIMAX
                with open(f'modelos/sarimax_{nombre_archivo}.pkl', 'wb') as f:
                    pickle.dump(sarimax_nuevo, f)
                logger.info(f"   ✅ sarimax_{nombre_archivo}.pkl guardado")
                
                # Guardar metadata
                with open(f'modelos/metadata_{nombre_archivo}.pkl', 'wb') as f:
                    pickle.dump(metadata, f)
                logger.info(f"   ✅ metadata_{nombre_archivo}.pkl guardado")
                
                # Guardar LSTM si existe
                if lstm_nuevo is not None:
                    lstm_nuevo.save(f'modelos/modelo_{nombre_archivo}.h5')
                    logger.info(f"   ✅ modelo_{nombre_archivo}.h5 guardado")
                
                logger.info(f"✅ {region_key} actualizado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error actualizando {region_key}: {e}", exc_info=True)
                continue
        
        # 3. Resumen final
        logger.info("\n" + "="*70)
        logger.info("📊 RESUMEN DE ACTUALIZACIÓN:")
        logger.info("="*70)
        
        for region_key, modelo_dict in modelos.items():
            metadata = modelo_dict['metadata']
            ultima_fecha = metadata['serie'].index[-1]
            dias_desde_ultima = (datetime.now().date() - ultima_fecha.date()).days
            
            logger.info(f"\n📍 {region_key}:")
            logger.info(f"   - Última fecha: {ultima_fecha.date()}")
            logger.info(f"   - Días desde última: {dias_desde_ultima}")
            logger.info(f"   - Total semanas: {len(metadata['serie'])}")
            if 'ultima_actualizacion' in metadata:
                logger.info(f"   - Actualizado: {metadata['ultima_actualizacion'][:19]}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ ACTUALIZACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("="*70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ ERROR CRÍTICO EN ACTUALIZACIÓN: {e}", exc_info=True)
        return False


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    try:
        # Cambiar al directorio del proyecto
        proyecto_dir = Path(__file__).parent
        import os
        os.chdir(proyecto_dir)
        
        logger.info(f"📁 Directorio de trabajo: {Path.cwd()}")
        
        # Ejecutar actualización
        exito = actualizar_todos_los_modelos()
        
        # Exit code para cron
        sys.exit(0 if exito else 1)
        
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        sys.exit(1)