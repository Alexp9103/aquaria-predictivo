#!/usr/bin/env python3
"""
Script para limpiar datos futuros de los modelos
Corta las series temporales hasta HOY y re-guarda los metadata
"""

import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd

def limpiar_datos_futuros():
    """
    Corta las series temporales hasta la fecha actual
    """
    print("="*70)
    print("🧹 LIMPIANDO DATOS FUTUROS")
    print(f"📅 Fecha actual: {datetime.now().date()}")
    print("="*70)
    
    modelos_dir = Path('modelos')
    hoy = pd.Timestamp(datetime.now().date())
    
    # Buscar todos los metadata
    for archivo in sorted(modelos_dir.glob('metadata_grupo_*.pkl')):
        nombre = archivo.stem.replace('metadata_', '')
        
        print(f"\n{'='*60}")
        print(f"📁 PROCESANDO: {nombre}")
        print(f"{'='*60}")
        
        try:
            # Cargar metadata
            with open(archivo, 'rb') as f:
                metadata = pickle.load(f)
            
            serie_original = metadata['serie']
            
            print(f"📊 Serie original:")
            print(f"   Primera fecha: {serie_original.index[0].date()}")
            print(f"   Última fecha: {serie_original.index[-1].date()}")
            print(f"   Total semanas: {len(serie_original)}")
            
            # Verificar si hay datos futuros
            ultima_fecha = serie_original.index[-1]
            
            if ultima_fecha > hoy:
                dias_futuro = (ultima_fecha.date() - hoy.date()).days
                print(f"\n⚠️ DATOS FUTUROS DETECTADOS: {dias_futuro} días")
                
                # Cortar hasta hoy
                serie_limpia = serie_original[serie_original.index <= hoy]
                
                print(f"\n✂️ Cortando serie...")
                print(f"   Nueva última fecha: {serie_limpia.index[-1].date()}")
                print(f"   Nuevo total semanas: {len(serie_limpia)}")
                print(f"   Semanas removidas: {len(serie_original) - len(serie_limpia)}")
                
                # Actualizar metadata
                metadata['serie'] = serie_limpia
                metadata['semanas_totales'] = len(serie_limpia)
                metadata['limpiado_fecha'] = datetime.now().isoformat()
                
                # Guardar metadata actualizado
                with open(archivo, 'wb') as f:
                    pickle.dump(metadata, f)
                
                print(f"\n✅ Metadata actualizado y guardado")
                
                # IMPORTANTE: Marcar que necesita re-entrenamiento
                print(f"\n🔴 IMPORTANTE: Este modelo necesita RE-ENTRENAMIENTO")
                print(f"   Los modelos SARIMAX y LSTM fueron entrenados con datos hasta 2026")
                print(f"   Deberás re-entrenarlos con la serie limpia")
                
            else:
                print(f"\n✅ No hay datos futuros. Serie ya está actualizada.")
            
        except Exception as e:
            print(f"❌ Error procesando {nombre}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("✅ LIMPIEZA COMPLETADA")
    print("="*70)
    
    print("\n🔴 PRÓXIMOS PASOS:")
    print("1. Re-entrenar los modelos SARIMAX y LSTM con los datos limpios")
    print("2. Ejecutar el script de entrenamiento original")
    print("3. Luego configurar la actualización diaria")


if __name__ == "__main__":
    respuesta = input("\n⚠️ Esto modificará los archivos metadata. ¿Continuar? (s/n): ")
    
    if respuesta.lower() == 's':
        limpiar_datos_futuros()
    else:
        print("❌ Operación cancelada")