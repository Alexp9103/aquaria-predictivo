# modernizar_modelos.py
from tensorflow import keras
from pathlib import Path

for h5 in Path('modelos').glob('modelo_*.h5'):
    print(f"Convirtiendo {h5.name} → .keras")
    model = keras.models.load_model(h5)
    nuevo = h5.with_suffix('.keras')
    model.save(nuevo)
    print(f"✓ {nuevo.name} creado")