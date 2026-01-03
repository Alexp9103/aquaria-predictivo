#!/usr/bin/env python3
"""
Script de Diagnóstico de Modelos
Verifica el estado de todos los modelos y metadata
"""

import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd

def diagnosticar_modelos():
    """
    Realiza diagnóstico completo de modelos
    """
    print("="*70)
    print("🔍 DIAGNÓSTICO DE MODELOS")
    print("="*70)
    
    modelos_dir = Path('modelos')
    
    if not modelos_dir.exists():
        print("❌ La carpeta 'modelos/' no existe")
        return
    
    # Buscar todos los metadata
    for archivo in sorted(modelos_dir.glob('metadata_grupo_*.pkl')):
        nombre = archivo.stem.replace('metadata_', '')
        
        print(f"\n{'='*70}")
        print(f"📁 MODELO: {nombre}")
        print(f"{'='*70}")
        
        try:
            # Cargar metadata
            with open(archivo, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"\n📋 METADATA:")
            print(f"   Claves disponibles: {list(metadata.keys())}")
            
            # Serie temporal
            if 'serie' in metadata:
                serie = metadata['serie']
                print(f"\n📊 SERIE TEMPORAL:")
                print(f"   Tipo: {type(serie)}")
                print(f"   Primera fecha: {serie.index[0].date()}")
                print(f"   Última fecha: {serie.index[-1].date()}")
                print(f"   Total semanas: {len(serie)}")
                
                dias_desde_ultima = (datetime.now().date() - serie.index[-1].date()).days
                print(f"   Días desde última: {dias_desde_ultima}")
                
                if dias_desde_ultima <= 7:
                    print(f"   Estado: ✅ ACTUALIZADO")
                elif dias_desde_ultima <= 14:
                    print(f"   Estado: ⚠️ DESACTUALIZADO")
                else:
                    print(f"   Estado: ❌ MUY DESACTUALIZADO")
                
                # Estadísticas
                print(f"\n   📈 Estadísticas:")
                print(f"      Media: {serie.mean():.2f} mm/semana")
                print(f"      Desv. Est.: {serie.std():.2f} mm")
                print(f"      Máximo: {serie.max():.2f} mm")
                print(f"      Mínimo: {serie.min():.2f} mm")
            
            # Ciudades
            if 'ciudades' in metadata:
                print(f"\n🏙️ CIUDADES ({len(metadata['ciudades'])}):")
                for ciudad in metadata['ciudades']:
                    print(f"      - {ciudad}")
            
            # Métricas
            if 'metricas' in metadata:
                metricas = metadata['metricas']
                print(f"\n🎯 MÉTRICAS DEL MODELO:")
                print(f"   MAE: {metricas.get('mae', 'N/A'):.2f} mm")
                print(f"   RMSE: {metricas.get('rmse', 'N/A'):.2f} mm")
                print(f"   MAPE: {metricas.get('mape', 'N/A'):.1f}%")
                print(f"   Variabilidad: {metricas.get('variability_ratio', 'N/A'):.1%}")
            
            # Configuración SARIMAX
            if 'sarimax_order' in metadata:
                print(f"\n⚙️ CONFIGURACIÓN SARIMAX:")
                print(f"   Orden: {metadata['sarimax_order']}")
                if 'sarimax_seasonal_order' in metadata:
                    print(f"   Orden estacional: {metadata['sarimax_seasonal_order']}")
            else:
                print(f"\n⚠️ No se encontró configuración SARIMAX")
            
            # Última actualización
            if 'ultima_actualizacion' in metadata:
                print(f"\n🕒 ÚLTIMA ACTUALIZACIÓN:")
                print(f"   {metadata['ultima_actualizacion']}")
            
            # Verificar archivos asociados
            print(f"\n📦 ARCHIVOS ASOCIADOS:")
            
            sarimax_file = modelos_dir / f'sarimax_{nombre}.pkl'
            if sarimax_file.exists():
                size_mb = sarimax_file.stat().st_size / (1024**2)
                print(f"   ✅ sarimax_{nombre}.pkl ({size_mb:.0f} MB)")
            else:
                print(f"   ❌ sarimax_{nombre}.pkl NO ENCONTRADO")
            
            lstm_file = modelos_dir / f'modelo_{nombre}.h5'
            if lstm_file.exists():
                size_mb = lstm_file.stat().st_size / (1024**2)
                print(f"   ✅ modelo_{nombre}.h5 ({size_mb:.0f} MB)")
            else:
                print(f"   ⚠️ modelo_{nombre}.h5 NO ENCONTRADO")
            
        except Exception as e:
            print(f"❌ Error al analizar {nombre}: {e}")
    
    print(f"\n{'='*70}")
    print("✅ DIAGNÓSTICO COMPLETADO")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    diagnosticar_modelos()