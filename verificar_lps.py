#!/usr/bin/env python3
"""
Verificar si archivos LFS fueron descargados correctamente
"""

from pathlib import Path

print("="*70)
print("🔍 VERIFICACIÓN DE ARCHIVOS LFS")
print("="*70)

carpetas = ['formatoadecuado', 'datos_meteo']

for carpeta in carpetas:
    print(f"\n📁 {carpeta}/")
    
    path = Path(carpeta)
    
    if not path.exists():
        print(f"   ❌ Carpeta no existe")
        continue
    
    archivos_csv = list(path.glob('*.csv'))
    
    if len(archivos_csv) == 0:
        print(f"   ❌ No hay archivos CSV")
        continue
    
    print(f"   Total archivos: {len(archivos_csv)}")
    
    punteros_lfs = 0
    archivos_reales = 0
    
    for archivo in archivos_csv:
        size = archivo.stat().st_size
        
        # Verificar si es puntero LFS (< 200 bytes usualmente)
        if size < 200:
            with open(archivo, 'r') as f:
                primera_linea = f.readline()
                if 'version https://git-lfs.github.com' in primera_linea:
                    punteros_lfs += 1
                    print(f"   ⚠️  PUNTERO LFS: {archivo.name} ({size} bytes)")
                    continue
        
        archivos_reales += 1
        if archivos_reales <= 3:  # Mostrar solo primeros 3
            print(f"   ✅ {archivo.name} ({size:,} bytes)")
    
    if punteros_lfs > 0:
        print(f"\n   ❌ PROBLEMA: {punteros_lfs} archivos son punteros LFS")
        print(f"   Solución: Ejecutar 'git lfs pull' para descargar archivos reales")
    else:
        print(f"\n   ✅ Todos los archivos ({archivos_reales}) están descargados correctamente")

print("\n" + "="*70)
print("DIAGNÓSTICO:")
print("="*70)

# Verificar si git-lfs está instalado
import subprocess

try:
    result = subprocess.run(['git', 'lfs', 'version'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ Git LFS está instalado")
        print(f"   {result.stdout.strip()}")
    else:
        print("❌ Git LFS no está instalado correctamente")
except FileNotFoundError:
    print("❌ Git LFS no está instalado")
except Exception as e:
    print(f"⚠️  Error al verificar Git LFS: {e}")

print("="*70)