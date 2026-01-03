
#!/usr/bin/env python3
"""
Script de inicialización para Railway
Entrena modelos solo si no existen o si FORCE_RETRAIN=true
"""

import os
import sys
from pathlib import Path

print("\n" + "="*70)
print("🔧 INICIALIZANDO AQUARIA EN RAILWAY")
print("="*70)

# Verificar si estamos en Railway
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None
FORCE_RETRAIN = os.getenv('FORCE_RETRAIN', 'false').lower() == 'true'

print(f"Entorno: {'Railway' if IS_RAILWAY else 'Local'}")
print(f"Force Retrain: {FORCE_RETRAIN}")

# Verificar si existen modelos
modelos_path = Path('modelos')
modelos_path.mkdir(exist_ok=True)

modelo_files = list(modelos_path.glob('*_hibrido.pkl'))
print(f"\nModelos encontrados: {len(modelo_files)}")

# Decidir si entrenar
DEBE_ENTRENAR = len(modelo_files) == 0 or FORCE_RETRAIN

if DEBE_ENTRENAR:
    print("\n🔄 ENTRENANDO MODELOS...")
    print("="*70)
    
    # Importar y ejecutar entrenamiento
    try:
        # Agregar directorio actual al path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Importar script de entrenamiento
        print("\n📦 Importando entrenar-mejorado.py...")
        import entrenar_mejorado
        
        # Ejecutar entrenamiento
        print("\n🚀 Iniciando entrenamiento...")
        resultados = entrenar_mejorado.main()
        
        if resultados and len(resultados) > 0:
            print("\n✅ ENTRENAMIENTO COMPLETADO")
            print(f"   Modelos creados: {len(resultados)}")
        else:
            print("\n❌ ERROR: No se crearon modelos")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR EN ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("\n✅ MODELOS EXISTENTES - SKIP ENTRENAMIENTO")
    for modelo in modelo_files:
        print(f"   • {modelo.name}")

print("\n" + "="*70)
print("✅ INICIALIZACIÓN COMPLETADA")
print("="*70 + "\n")