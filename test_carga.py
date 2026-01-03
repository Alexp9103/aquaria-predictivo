# test_carga.py
import pickle
from pathlib import Path

print("Testing calibrador loading...")

for archivo in Path('modelos').glob('calibrador_*_simple.pkl'):
    print(f"\n📦 {archivo.name}")
    
    # Procesar nombre igual que en la app
    nombre_stem = archivo.stem  # calibrador_grupo_1_norte_cibao_simple
    
    if nombre_stem.startswith('calibrador_'):
        nombre_sin_prefijo = nombre_stem[11:]  # grupo_1_norte_cibao_simple
    else:
        nombre_sin_prefijo = nombre_stem
    
    if nombre_sin_prefijo.endswith('_simple'):
        nombre_final = nombre_sin_prefijo[:-7]  # grupo_1_norte_cibao
    else:
        nombre_final = nombre_sin_prefijo
    
    print(f"  → Nombre final: '{nombre_final}'")
    
    with open(archivo, 'rb') as f:
        cal = pickle.load(f)
    
    print(f"  → Tipo: {type(cal)}")
    print(f"  → Keys: {list(cal.keys())}")
    
    if 'sesgos' in cal:
        print(f"  ✅ Tiene 'sesgos'")
        print(f"  → Niveles: {list(cal['sesgos'].keys())}")
    
    if 'configuraciones' in cal:
        print(f"  ✅ Tiene 'configuraciones'")
        print(f"  → Modos: {list(cal['configuraciones'].keys())}")

print("\n" + "="*70)
print("Verificando nombres de modelos...")

for archivo in Path('modelos').glob('metadata_*.pkl'):
    nombre = archivo.stem.replace('metadata_', '')
    if '_hibrido' not in nombre:
        print(f"  • Modelo: '{nombre}'")