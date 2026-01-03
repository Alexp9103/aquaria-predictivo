import pickle
from pathlib import Path

print("="*70)
print("TEST DE CARGA DE CALIBRADORES")
print("="*70)

# Listar archivos
print("\n1. Archivos calibrador*.pkl:")
archivos = list(Path('modelos').glob('calibrador*.pkl'))
print(f"   Total encontrados: {len(archivos)}")
for a in archivos:
    print(f"   • {a.name}")

# Intentar cargar cada uno
print("\n2. Intentando cargar:")
for archivo in archivos:
    print(f"\n   📦 {archivo.name}")
    
    try:
        with open(archivo, 'rb') as f:
            cal = pickle.load(f)
        
        print(f"      ✅ Cargado")
        print(f"      → type: {type(cal)}")
        
        if isinstance(cal, dict):
            print(f"      → keys: {list(cal.keys())}")
            
            # Verificar tipo
            if 'sesgos' in cal:
                print(f"      → TIPO: Estratificado")
            elif 'factores' in cal:
                print(f"      → TIPO: Multiplicativo")
            else:
                print(f"      → TIPO: Desconocido")
    except Exception as e:
        print(f"      ❌ ERROR: {e}")

# Verificar nombres de modelos
print("\n3. Nombres de modelos disponibles:")
for meta in Path('modelos').glob('metadata_*.pkl'):
    nombre = meta.stem.replace('metadata_', '')
    if '_hibrido' not in nombre:
        print(f"   • {nombre}")

print("\n" + "="*70)