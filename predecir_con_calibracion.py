"""
Script de predicción AQUARIA con calibración post-modelo
Genera predicciones para las próximas 12 semanas
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras

# Importar funciones de calibración
import sys
sys.path.append('.')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

GRUPOS_DISPONIBLES = {
    'grupo_1_norte_cibao': 'Norte y Cibao',
    'grupo_2_sur_seco': 'Sur seco',
    'grupo_3_este_capital': 'Este y Capital'
}

# ============================================================================
# FUNCIONES DE CALIBRACIÓN (copiadas del calibrador)
# ============================================================================

def determinar_nivel(prediccion):
    """Determina el nivel de la predicción"""
    if prediccion < 5:
        return 'muy_bajo'
    elif prediccion < 15:
        return 'bajo'
    elif prediccion < 30:
        return 'medio'
    elif prediccion < 60:
        return 'alto'
    else:
        return 'muy_alto'

def aplicar_calibracion_estratificada(predicciones, sesgos, agresividad=0.5):
    """Aplica corrección de sesgo estratificada por nivel"""
    predicciones_corregidas = predicciones.copy()
    
    for i, pred in enumerate(predicciones):
        nivel = determinar_nivel(pred)
        sesgo = sesgos[nivel]
        
        # Protección para valores muy pequeños
        if pred < 2.0:
            agresividad_efectiva = agresividad * 0.1
        else:
            agresividad_efectiva = agresividad
        
        correccion = agresividad_efectiva * sesgo
        predicciones_corregidas[i] = pred - correccion
        
    return np.maximum(predicciones_corregidas, 0.1)

def aumentar_variabilidad(predicciones, factor=1.2, preservar_media=True):
    """Aumenta la variabilidad de las predicciones"""
    if preservar_media:
        media = np.mean(predicciones)
        predicciones_escaladas = (predicciones - media) * factor + media
    else:
        predicciones_escaladas = predicciones * factor
    
    return np.maximum(predicciones_escaladas, 0.0)

def calibrar_completo(predicciones, sesgos, agresividad=0.5, aumentar_var=True, factor_var=1.2):
    """Pipeline completo de calibración"""
    if aumentar_var:
        pred_var = aumentar_variabilidad(predicciones, factor=factor_var)
    else:
        pred_var = predicciones.copy()
    
    pred_final = aplicar_calibracion_estratificada(pred_var, sesgos, agresividad)
    
    return pred_final

# ============================================================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# ============================================================================

def predecir_con_calibracion(nombre_grupo, semanas=12, config_calibracion='balanceado'):
    """
    Genera predicciones calibradas para un grupo
    
    Args:
        nombre_grupo: str - uno de los grupos disponibles
        semanas: int - número de semanas a predecir
        config_calibracion: str - 'conservador', 'balanceado', o 'agresivo'
    
    Returns:
        DataFrame con predicciones originales y calibradas
    """
    print(f"\n{'='*70}")
    print(f"🌊 PREDICCIÓN PARA: {GRUPOS_DISPONIBLES[nombre_grupo].upper()}")
    print(f"{'='*70}")
    
    # 1. Cargar metadata del modelo
    metadata_path = Path('modelos') / f'metadata_{nombre_grupo}_hibrido.pkl'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró modelo para {nombre_grupo}")
    
    print(f"\n📂 Cargando modelo...")
    
    # Clase auxiliar para cargar pickle
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name in ['PreprocessorAvanzado', 'ModeloHibridoAQUARIA']:
                return type(name, (), {})
            return super().find_class(module, name)
    
    with open(metadata_path, 'rb') as f:
        metadata = CustomUnpickler(f).load()
    
    print(f"   ✓ Metadata cargada")
    print(f"   - MAE entrenamiento: {metadata['metricas']['mae']:.2f} mm")
    print(f"   - Variabilidad: {metadata['metricas']['variability_ratio']:.3f}")
    
    # 2. Cargar calibrador
    calibrador_path = Path('modelos') / f'calibrador_{nombre_grupo}_simple.pkl'
    
    if not calibrador_path.exists():
        print(f"\n⚠️  No se encontró calibrador. Usando predicciones sin calibrar.")
        usar_calibrador = False
    else:
        with open(calibrador_path, 'rb') as f:
            calibrador = pickle.load(f)
        print(f"\n📊 Calibrador cargado")
        print(f"   Configuración: {config_calibracion}")
        usar_calibrador = True
    
    # 3. AQUÍ IRÍAN TUS PREDICCIONES REALES DEL MODELO
    # Por ahora, genero predicciones simuladas basadas en datos históricos
    print(f"\n🔮 Generando predicciones para las próximas {semanas} semanas...")
    
    # Simulación (reemplaza esto con tu código real de predicción)
    fecha_inicio = datetime.now()
    fechas = [fecha_inicio + timedelta(weeks=i) for i in range(semanas)]
    
    # Predicciones simuladas con patrón estacional
    np.random.seed(42)
    mes_inicio = fecha_inicio.month
    
    predicciones_orig = []
    for i in range(semanas):
        mes = (mes_inicio + i // 4) % 12
        # Época húmeda (mayo-nov) vs seca
        if mes in [5, 6, 7, 8, 9, 10, 11]:
            base = 25 + np.random.randn() * 8
        else:
            base = 12 + np.random.randn() * 5
        predicciones_orig.append(max(0.1, base))
    
    predicciones_orig = np.array(predicciones_orig)
    
    print(f"   ✓ Predicciones base generadas")
    
    # 4. Aplicar calibración
    if usar_calibrador:
        config = calibrador['configuraciones'][config_calibracion]
        
        predicciones_cal = calibrar_completo(
            predicciones_orig,
            calibrador['sesgos'],
            agresividad=config['agresividad'],
            aumentar_var=True,
            factor_var=config['factor_variabilidad']
        )
        
        print(f"   ✓ Calibración aplicada ({config_calibracion})")
        print(f"      - Agresividad: {config['agresividad']:.1%}")
        print(f"      - Factor variabilidad: {config['factor_variabilidad']:.2f}")
    else:
        predicciones_cal = predicciones_orig.copy()
    
    # 5. Crear DataFrame con resultados
    df_resultados = pd.DataFrame({
        'Fecha': fechas,
        'Semana': range(1, semanas + 1),
        'Prediccion_Original_mm': predicciones_orig.round(2),
        'Prediccion_Calibrada_mm': predicciones_cal.round(2),
        'Diferencia_mm': (predicciones_cal - predicciones_orig).round(2),
        'Nivel_Original': [determinar_nivel(p) for p in predicciones_orig],
        'Nivel_Calibrado': [determinar_nivel(p) for p in predicciones_cal]
    })
    
    # 6. Estadísticas
    print(f"\n📈 Estadísticas de predicción:")
    print(f"   Original:")
    print(f"      - Media: {predicciones_orig.mean():.2f} mm/sem")
    print(f"      - Std:   {predicciones_orig.std():.2f} mm")
    print(f"      - Total: {predicciones_orig.sum():.2f} mm ({semanas} semanas)")
    
    if usar_calibrador:
        print(f"\n   Calibrada:")
        print(f"      - Media: {predicciones_cal.mean():.2f} mm/sem")
        print(f"      - Std:   {predicciones_cal.std():.2f} mm")
        print(f"      - Total: {predicciones_cal.sum():.2f} mm ({semanas} semanas)")
        print(f"\n   Cambio:")
        print(f"      - ΔMedia: {predicciones_cal.mean() - predicciones_orig.mean():+.2f} mm")
        print(f"      - ΔStd:   {predicciones_cal.std() - predicciones_orig.std():+.2f} mm")
    
    # 7. Alertas por nivel
    print(f"\n🚨 Distribución por nivel de alerta:")
    for nivel in ['muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto']:
        count_orig = (df_resultados['Nivel_Original'] == nivel).sum()
        if usar_calibrador:
            count_cal = (df_resultados['Nivel_Calibrado'] == nivel).sum()
            print(f"   {nivel:12s}: {count_orig} → {count_cal} semanas")
        else:
            print(f"   {nivel:12s}: {count_orig} semanas")
    
    return df_resultados

# ============================================================================
# FUNCIÓN PARA EXPORTAR RESULTADOS
# ============================================================================

def exportar_predicciones(df, nombre_grupo, formato='csv'):
    """Exporta predicciones a archivo"""
    Path('predicciones').mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'predicciones/pred_{nombre_grupo}_{timestamp}.{formato}'
    
    if formato == 'csv':
        df.to_csv(filename, index=False)
    elif formato == 'excel':
        df.to_excel(filename, index=False)
    
    print(f"\n💾 Predicciones exportadas: {filename}")
    return filename

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🌊 AQUARIA - SISTEMA DE PREDICCIÓN CON CALIBRACIÓN")
    print("="*70)
    
    # Predecir para todos los grupos
    resultados = {}
    
    for nombre_grupo in GRUPOS_DISPONIBLES.keys():
        try:
            df = predecir_con_calibracion(
                nombre_grupo=nombre_grupo,
                semanas=12,
                config_calibracion='balanceado'  # Cambia a 'conservador' o 'agresivo'
            )
            
            resultados[nombre_grupo] = df
            
            # Exportar
            exportar_predicciones(df, nombre_grupo, formato='csv')
            
            # Mostrar tabla resumida
            print(f"\n📋 Predicciones resumidas (primeras 6 semanas):")
            print(df[['Semana', 'Prediccion_Original_mm', 'Prediccion_Calibrada_mm', 'Nivel_Calibrado']].head(6).to_string(index=False))
            
        except Exception as e:
            print(f"\n❌ Error procesando {nombre_grupo}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✅ Predicciones completadas para {len(resultados)} grupos")
    print(f"{'='*70}")
    
    return resultados

if __name__ == "__main__":
    resultados = main()