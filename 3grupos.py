import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow import keras
import pickle

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURACIÓN DE GRUPOS CLIMÁTICOS
# ============================================================================

CARPETA_DATOS = '/home/juanpucmm/modelo predectivo/formatoadecuado'


GRUPOS_CLIMATICOS = {
    'grupo_1_norte_cibao': {
        'ciudades': ['santiago', 'montecristi', 'cabrera', 'catey', 
                     'union', 'arroyo', 'sabana'],
        'descripcion': 'Norte y Cibao - Mayor variabilidad'
    },
    'grupo_2_sur_seco': {
        'ciudades': ['barahona', 'jimani'],
        'descripcion': 'Sur seco - Zona más árida'
    },
    'grupo_3_este_capital': {
        'ciudades': ['santodomingo', 'puntacana', 'romana', 'lasamericas',
                     'higuero', 'bayaguana'],
        'descripcion': 'Este y Capital - Costa y zona metropolitana'
    }
}

# ============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

def cargar_datos_grupo(ciudades, carpeta):  # Recibe carpeta como parámetro
    """Carga y combina datos de múltiples ciudades de un grupo"""
    
    datos_grupo = []
    
    for ciudad in ciudades:
        # Buscar archivo con diferentes formatos posibles
        posibles_archivos = [
            f"{ciudad}_diario_2010_2025.csv",
            f"{ciudad.replace('_', '')}_diario_2010_2025.csv",
            f"{ciudad.replace('santo', 'santo_')}_diario_2010_2025.csv"
        ]
        
        archivo_encontrado = None
        for nombre in posibles_archivos:
            ruta = Path(carpeta) / nombre
            if ruta.exists():
                archivo_encontrado = ruta
                break
        
        if archivo_encontrado is None:
            print(f"  ⚠ No se encontró archivo para: {ciudad}")
            continue
        
        try:
            df = pd.read_csv(archivo_encontrado, parse_dates=['Fecha'])
            df = df.set_index('Fecha')
            df['Precip_mm'].fillna(0, inplace=True)
            
            # Limpiar outliers extremos
            q98 = df['Precip_mm'].quantile(0.98)
            extreme = df['Precip_mm'] > q98 * 2
            df.loc[extreme, 'Precip_mm'] = q98 + np.sqrt(df.loc[extreme, 'Precip_mm'] - q98)
            
            # Agregar a semanal
            semanal = df['Precip_mm'].resample('W-SUN').sum()
            
            datos_grupo.append({
                'ciudad': ciudad,
                'serie': semanal,
                'registros': len(semanal)
            })
            
            print(f"  ✓ {ciudad}: {len(semanal)} semanas")
            
        except Exception as e:
            print(f"  ✗ Error en {ciudad}: {e}")
    
    return datos_grupo

def combinar_series_grupo(datos_grupo):
    """Combina series de ciudades tomando el promedio ponderado"""
    
    if not datos_grupo:
        return None
    
    # Encontrar fechas comunes
    fechas_comunes = set(datos_grupo[0]['serie'].index)
    for dato in datos_grupo[1:]:
        fechas_comunes &= set(dato['serie'].index)
    
    fechas_comunes = sorted(list(fechas_comunes))
    
    if len(fechas_comunes) < 52:
        print(f"  ⚠ Pocas fechas comunes: {len(fechas_comunes)}")
        return None
    
    # Alinear y promediar
    series_alineadas = []
    for dato in datos_grupo:
        serie_aligned = dato['serie'].reindex(fechas_comunes)
        series_alineadas.append(serie_aligned.values)
    
    # Promedio simple del grupo
    serie_promedio = pd.Series(
        np.mean(series_alineadas, axis=0),
        index=fechas_comunes
    )
    
    return serie_promedio

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_enhanced_features(serie, scaled_serie, scaler):
    """Crear features que preserven variabilidad y estacionalidad"""
    
    df_features = pd.DataFrame(index=serie.index)
    df_features['precip_scaled'] = scaled_serie.flatten()
    
    # Features estacionales
    week_of_year = serie.index.isocalendar().week.astype(float)
    month = serie.index.month.astype(float)
    
    # Componentes cíclicos
    df_features['week_sin1'] = np.sin(2 * np.pi * week_of_year / 52.0)
    df_features['week_cos1'] = np.cos(2 * np.pi * week_of_year / 52.0)
    df_features['week_sin2'] = np.sin(4 * np.pi * week_of_year / 52.0)
    df_features['week_cos2'] = np.cos(4 * np.pi * week_of_year / 52.0)
    
    df_features['month_sin1'] = np.sin(2 * np.pi * month / 12.0)
    df_features['month_cos1'] = np.cos(2 * np.pi * month / 12.0)
    df_features['month_sin2'] = np.sin(4 * np.pi * month / 12.0)
    df_features['month_cos2'] = np.cos(4 * np.pi * month / 12.0)
    
    # Lags
    for lag in [1, 2, 4, 52]:
        df_features[f'lag_{lag}'] = df_features['precip_scaled'].shift(lag)
    
    # Volatilidad
    for window in [4, 8, 12]:
        rolling_std = serie.rolling(window).std()
        rolling_std_filled = rolling_std.fillna(rolling_std.mean()).replace(0, rolling_std.mean())
        try:
            volatility_scaled = scaler.transform(rolling_std_filled.values.reshape(-1, 1)).flatten()
            df_features[f'volatility_{window}'] = volatility_scaled
        except:
            df_features[f'volatility_{window}'] = (rolling_std_filled - rolling_std_filled.mean()) / (rolling_std_filled.std() + 1e-8)
    
    # Trends
    df_features['trend_4w'] = df_features['precip_scaled'].rolling(4, center=True).mean()
    df_features['trend_12w'] = df_features['precip_scaled'].rolling(12, center=True).mean()
    
    # Limpiar
    df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0.0)
    
    return df_features

# ============================================================================
# FUNCIONES DE MODELADO
# ============================================================================

def create_sequences_variability_preserving(data, target_col=0, seq_length=26):
    """Crear secuencias para LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def create_variability_preserving_lstm(input_shape, dropout_rate=0.1):
    """LSTM que preserva variabilidad natural"""
    
    model = keras.Sequential([
        keras.layers.Bidirectional(
            keras.layers.LSTM(200, return_sequences=True, dropout=dropout_rate, 
                             recurrent_dropout=dropout_rate/2),
            input_shape=input_shape
        ),
        keras.layers.Dropout(dropout_rate),
        
        keras.layers.Bidirectional(
            keras.layers.LSTM(150, return_sequences=True, dropout=dropout_rate,
                             recurrent_dropout=dropout_rate/2)
        ),
        keras.layers.Dropout(dropout_rate),
        
        keras.layers.Bidirectional(
            keras.layers.LSTM(100, dropout=dropout_rate, recurrent_dropout=dropout_rate/2)
        ),
        keras.layers.Dropout(dropout_rate),
        
        keras.layers.Dense(150, activation='relu'),
        keras.layers.Dropout(dropout_rate/2),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dropout(dropout_rate/2),
        keras.layers.Dense(50, activation='tanh'),
        keras.layers.Dense(1, activation='linear')
    ])
    
    return model

# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO POR GRUPO
# ============================================================================

def entrenar_modelo_grupo(nombre_grupo, config_grupo, carpeta):  # Recibe carpeta como parámetro
    """Entrena modelo LSTM+SARIMAX para un grupo de ciudades"""
    
    ciudades = config_grupo['ciudades']
    descripcion = config_grupo['descripcion']
    
    print(f"\n{'='*70}")
    print(f"ENTRENANDO: {nombre_grupo.upper()}")
    print(f"{'='*70}")
    print(f"Descripción: {descripcion}")
    print(f"Ciudades: {', '.join(ciudades)}")
    print(f"Carpeta datos: {carpeta}")  # Mostrar la ruta que está usando

    
    # 1. Cargar datos
    print(f"\n📁 Cargando datos...")
    datos_grupo = cargar_datos_grupo(ciudades, carpeta)
    
    if not datos_grupo:
        print(f"✗ No se pudieron cargar datos para {nombre_grupo}")
        return None
    
    print(f"Ciudades cargadas: {len(datos_grupo)}/{len(ciudades)}")
    
    # 2. Combinar series
    print(f"\n🔄 Combinando series del grupo...")
    serie = combinar_series_grupo(datos_grupo)
    
    if serie is None or len(serie) < 104:
        print(f"✗ Datos insuficientes")
        return None
    
    print(f"Serie combinada: {len(serie)} semanas")
    print(f"Período: {serie.index[0].date()} a {serie.index[-1].date()}")
    print(f"Media: {serie.mean():.1f} mm, Std: {serie.std():.1f} mm, Max: {serie.max():.1f} mm")
    
    # 3. Escalado
    scaler = StandardScaler()
    serie_scaled = scaler.fit_transform(serie.values.reshape(-1, 1))
    
    # 4. Feature engineering
    print(f"\n🔧 Creando features...")
    features_df = create_enhanced_features(serie, serie_scaled, scaler)
    features_df = features_df.astype(np.float64)
    features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    features_df = features_df.replace([np.inf, -np.inf], 0.0)
    
    print(f"Features creados: {features_df.shape[1]}")
    
    # 5. Crear secuencias
    seq_length = 26
    data_array = features_df.values.astype(np.float32)
    X, y = create_sequences_variability_preserving(data_array, seq_length=seq_length)
    
    # Limpiar NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
    y = np.nan_to_num(y, nan=0.0, posinf=3.0, neginf=-3.0)
    
    # 6. Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"División: Train={len(X_train)}, Test={len(X_test)}")
    
    # 7. Entrenar LSTM
    print(f"\n🧠 Entrenando LSTM...")
    model_lstm = create_variability_preserving_lstm(
        input_shape=(seq_length, data_array.shape[1]),
        dropout_rate=0.1
    )
    
    model_lstm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.5),
        loss='mse',
        metrics=['mae']
    )
    
    history = model_lstm.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=15, min_lr=1e-6, verbose=0)
        ],
        verbose=0
    )
    
    print(f"Épocas entrenadas: {len(history.history['loss'])}")
    
    # 8. Predicciones LSTM
    y_pred_lstm_scaled = model_lstm.predict(X_test, verbose=0)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1))
    
    # 9. SARIMAX
    print(f"📈 Entrenando SARIMAX...")
    train_sarimax = serie[:train_size + seq_length]
    
    best_sarimax = None
    best_aic = float('inf')
    
    sarimax_configs = [
        ((2, 1, 2), (1, 1, 1, 52)),
        ((1, 1, 1), (1, 1, 1, 52)),
        ((1, 1, 2), (1, 1, 2, 52))
    ]
    
    for order, seasonal_order in sarimax_configs:
        try:
            model = SARIMAX(
                train_sarimax,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                concentrate_scale=True
            )
            results = model.fit(disp=False, maxiter=300)
            
            if results.aic < best_aic:
                best_aic = results.aic
                best_sarimax = results
                print(f"  Mejor SARIMAX: {order}x{seasonal_order}, AIC={best_aic:.2f}")
        except:
            continue
    
    if best_sarimax is None:
        model = SARIMAX(train_sarimax, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        best_sarimax = model.fit(disp=False)
    
    y_pred_sarimax = best_sarimax.forecast(steps=len(X_test))
    
    # 10. Optimizar ensemble
    print(f"⚖️ Optimizando ensemble...")
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    best_alpha = 0.5
    best_rmse = float('inf')
    
    for alpha in np.arange(0.0, 1.01, 0.05):
        if len(y_pred_sarimax) < len(y_pred_lstm):
            continue
        y_pred = alpha * y_pred_lstm.flatten() + (1 - alpha) * y_pred_sarimax.values[:len(y_pred_lstm)]
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    y_pred_hybrid = best_alpha * y_pred_lstm.flatten() + (1 - best_alpha) * y_pred_sarimax.values[:len(y_pred_lstm)]
    
    # 11. Métricas
    mae = mean_absolute_error(y_test_orig, y_pred_hybrid)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_hybrid))
    
    mask = y_test_orig > 1
    mape = mean_absolute_percentage_error(y_test_orig[mask], y_pred_hybrid[mask]) * 100 if mask.any() else np.nan
    
    variability_ratio = np.std(y_pred_hybrid) / np.std(y_test_orig)
    
    print(f"\n📊 MÉTRICAS {nombre_grupo}:")
    print(f"  MAE:  {mae:.2f} mm/semana")
    print(f"  RMSE: {rmse:.2f} mm/semana")
    print(f"  MAPE: {mape:.1f}% (semanas >1mm)")
    print(f"  Variabilidad ratio: {variability_ratio:.3f}")
    print(f"  Alpha óptimo: {best_alpha:.3f}")
    
    # 12. Guardar modelo
    model_lstm.save(f'modelos/modelo_{nombre_grupo}.h5')
    
    # Guardar scaler y metadata
    metadata = {
        'nombre': nombre_grupo,
        'ciudades': ciudades,
        'descripcion': descripcion,
        'scaler': scaler,
        'alpha': best_alpha,
        'serie': serie,
        'metricas': {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'variability_ratio': variability_ratio
        },
        'seq_length': seq_length,
        'num_features': data_array.shape[1]
    }
    
    with open(f'modelos/metadata_{nombre_grupo}.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Guardar SARIMAX
    best_sarimax.save(f'modelos/sarimax_{nombre_grupo}.pkl')
    
    print(f"✓ Modelos guardados en carpeta 'modelos/'")
    
    return metadata

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal para entrenar los 3 modelos"""
    
    print("="*70)
    print("ENTRENAMIENTO DE MODELOS POR GRUPO CLIMÁTICO")
    print("República Dominicana - Predicción de Precipitación")
    print("="*70)
    print(f"Carpeta de datos: {CARPETA_DATOS}")  # Mostrar la ruta configurada

    if not Path(CARPETA_DATOS).exists():
        print(f"\n❌ ERROR: La carpeta {CARPETA_DATOS} no existe")
        return None
    
    # Mostrar archivos encontrados
    archivos_csv = list(Path(CARPETA_DATOS).glob("*_diario_*.csv"))
    print(f"Archivos CSV encontrados: {len(archivos_csv)}")

    # Crear carpeta para modelos
    Path('modelos').mkdir(exist_ok=True)
    
    # Entrenar cada grupo
    resultados = {}
    
    for nombre_grupo, config_grupo in GRUPOS_CLIMATICOS.items():
        try:
            # Usar CARPETA_DATOS aquí
            metadata = entrenar_modelo_grupo(nombre_grupo, config_grupo, carpeta=CARPETA_DATOS)
            if metadata:
                resultados[nombre_grupo] = metadata
        except Exception as e:
            print(f"\n✗ Error en {nombre_grupo}: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Modelos entrenados exitosamente: {len(resultados)}/3")
    
    for nombre, metadata in resultados.items():
        print(f"\n{nombre}:")
        print(f"  Ciudades: {len(metadata['ciudades'])}")
        print(f"  MAE: {metadata['metricas']['mae']:.2f} mm")
        print(f"  RMSE: {metadata['metricas']['rmse']:.2f} mm")
        print(f"  MAPE: {metadata['metricas']['mape']:.1f}%")
        print(f"  Alpha: {metadata['alpha']:.3f}")
    
    print(f"\n✓ Entrenamiento completado")
    print(f"Modelos guardados en: ./modelos/")
    
    return resultados

if __name__ == "__main__":
    resultados = main()