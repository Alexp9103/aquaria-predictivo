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

np.random.seed(42)
tf.random.set_seed(42)

CARPETA_DATOS = '/home/juanpucmm/modelo predectivo/formatoadecuado'
CARPETA_METEO = 'datos_meteo'

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

def cargar_datos_con_meteo(ciudad, carpeta_precip, carpeta_meteo):
    """Carga datos de precipitación combinados con variables meteorológicas"""
    
    # Precipitación
    posibles = [
        f"{ciudad}_diario_2010_2025.csv",
        f"{ciudad.replace('_', '')}_diario_2010_2025.csv"
    ]
    
    archivo_precip = None
    for nombre in posibles:
        ruta = Path(carpeta_precip) / nombre
        if ruta.exists():
            archivo_precip = ruta
            break
    
    if archivo_precip is None:
        return None
    
    df_precip = pd.read_csv(archivo_precip, parse_dates=['Fecha'])
    df_precip = df_precip.set_index('Fecha')
    df_precip['Precip_mm'].fillna(0, inplace=True)
    
    # Datos meteorológicos
    archivo_meteo = Path(carpeta_meteo) / f"{ciudad}_meteo.csv"
    
    if not archivo_meteo.exists():
        print(f"  ⚠ Sin datos meteo para {ciudad}, usando solo precipitación")
        return df_precip
    
    df_meteo = pd.read_csv(archivo_meteo, parse_dates=['date'])
    df_meteo = df_meteo.set_index('date')
    
    # Combinar
    df_combined = df_precip.join(df_meteo, how='left')
    
    # Rellenar valores faltantes
    for col in df_meteo.columns:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].fillna(method='ffill').fillna(method='bfill')
    
    return df_combined

def cargar_datos_grupo_mejorado(ciudades, carpeta_precip, carpeta_meteo):
    """Carga datos con variables meteorológicas para un grupo"""
    
    datos_grupo = []
    
    for ciudad in ciudades:
        try:
            df = cargar_datos_con_meteo(ciudad, carpeta_precip, carpeta_meteo)
            
            if df is None:
                continue
            
            # Limpiar outliers
            q98 = df['Precip_mm'].quantile(0.98)
            extreme = df['Precip_mm'] > q98 * 2
            df.loc[extreme, 'Precip_mm'] = q98 + np.sqrt(df.loc[extreme, 'Precip_mm'] - q98)
            
            # Resample a semanal
            agg_dict = {'Precip_mm': 'sum'}
            
            # Agregar columnas meteorológicas si existen
            if 'temperature_2m_mean' in df.columns:
                agg_dict['temperature_2m_mean'] = 'mean'
                agg_dict['temperature_2m_max'] = 'max'
                agg_dict['temperature_2m_min'] = 'min'
                agg_dict['temp_range'] = 'mean'
                agg_dict['windspeed_10m_max'] = 'max'
                agg_dict['pressure_msl_mean'] = 'mean'
                agg_dict['pressure_change'] = ['mean', 'std']
                agg_dict['et0_fao_evapotranspiration'] = 'sum'
            
            semanal = df.resample('W-SUN').agg(agg_dict)
            
            # Aplanar columnas multi-nivel
            if isinstance(semanal.columns, pd.MultiIndex):
                semanal.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                   for col in semanal.columns.values]
            
            datos_grupo.append({
                'ciudad': ciudad,
                'datos': semanal
            })
            
            print(f"  ✓ {ciudad}: {len(semanal)} semanas")
            
        except Exception as e:
            print(f"  ✗ Error en {ciudad}: {e}")
    
    return datos_grupo

def combinar_series_grupo_mejorado(datos_grupo):
    """Combina series preservando variables meteorológicas"""
    
    if not datos_grupo:
        return None
    
    # Encontrar fechas comunes
    fechas_comunes = set(datos_grupo[0]['datos'].index)
    for dato in datos_grupo[1:]:
        fechas_comunes &= set(dato['datos'].index)
    
    fechas_comunes = sorted(list(fechas_comunes))
    
    if len(fechas_comunes) < 52:
        return None
    
    # Promediar todas las series
    df_combined = pd.DataFrame(index=fechas_comunes)
    
    for col in datos_grupo[0]['datos'].columns:
        valores = []
        for dato in datos_grupo:
            if col in dato['datos'].columns:
                serie_aligned = dato['datos'][col].reindex(fechas_comunes)
                valores.append(serie_aligned.values)
        
        if valores:
            df_combined[col] = np.mean(valores, axis=0)
    
    return df_combined

def create_enhanced_features_with_meteo(df_combined, serie_scaled, scaler):
    """Feature engineering con variables meteorológicas"""
    
    df_features = pd.DataFrame(index=df_combined.index)
    df_features['precip_scaled'] = serie_scaled.flatten()
    
    # Features temporales
    week = df_combined.index.isocalendar().week.astype(float)
    month = df_combined.index.month.astype(float)
    
    df_features['week_sin1'] = np.sin(2 * np.pi * week / 52.0)
    df_features['week_cos1'] = np.cos(2 * np.pi * week / 52.0)
    df_features['week_sin2'] = np.sin(4 * np.pi * week / 52.0)
    df_features['week_cos2'] = np.cos(4 * np.pi * week / 52.0)
    
    df_features['month_sin1'] = np.sin(2 * np.pi * month / 12.0)
    df_features['month_cos1'] = np.cos(2 * np.pi * month / 12.0)
    df_features['month_sin2'] = np.sin(4 * np.pi * month / 12.0)
    df_features['month_cos2'] = np.cos(4 * np.pi * month / 12.0)
    
    # Variables meteorológicas normalizadas
    meteo_vars = [
        'temperature_2m_mean_mean', 'temperature_2m_max_max', 
        'temperature_2m_min_min', 'temp_range_mean',
        'windspeed_10m_max_max', 'pressure_msl_mean_mean',
        'pressure_change_mean', 'pressure_change_std',
        'et0_fao_evapotranspiration_sum'
    ]
    
    for var in meteo_vars:
        if var in df_combined.columns:
            scaler_temp = StandardScaler()
            values = df_combined[var].fillna(method='ffill').fillna(method='bfill').fillna(0)
            df_features[f'meteo_{var}'] = scaler_temp.fit_transform(values.values.reshape(-1, 1))
    
    # Interacciones temperatura-precipitación
    if 'meteo_temperature_2m_mean_mean' in df_features.columns:
        df_features['temp_precip'] = df_features['precip_scaled'] * df_features['meteo_temperature_2m_mean_mean']
    
    # Interacciones presión-precipitación
    if 'meteo_pressure_msl_mean_mean' in df_features.columns:
        df_features['pressure_precip'] = df_features['precip_scaled'] * df_features['meteo_pressure_msl_mean_mean']
    
    # Lags de precipitación
    for lag in [1, 2, 4, 52]:
        df_features[f'lag_{lag}'] = df_features['precip_scaled'].shift(lag)
    
    # Lags de temperatura (predictores fuertes)
    if 'meteo_temperature_2m_mean_mean' in df_features.columns:
        for lag in [1, 2]:
            df_features[f'temp_lag_{lag}'] = df_features['meteo_temperature_2m_mean_mean'].shift(lag)
    
    # Volatilidad
    serie_precip = df_combined['Precip_mm'] if 'Precip_mm' in df_combined.columns else df_combined.iloc[:, 0]
    for window in [4, 8, 12]:
        rolling_std = serie_precip.rolling(window).std()
        rolling_std_filled = rolling_std.fillna(rolling_std.mean()).replace(0, rolling_std.mean())
        try:
            vol_scaled = scaler.transform(rolling_std_filled.values.reshape(-1, 1)).flatten()
            df_features[f'volatility_{window}'] = vol_scaled
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

def create_sequences(data, target_col=0, seq_length=26):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def create_lstm_model(input_shape, dropout_rate=0.1):
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

def entrenar_modelo_grupo_mejorado(nombre_grupo, config_grupo, carpeta_precip, carpeta_meteo):
    """Entrena modelo con variables meteorológicas"""
    
    ciudades = config_grupo['ciudades']
    descripcion = config_grupo['descripcion']
    
    print(f"\n{'='*70}")
    print(f"ENTRENANDO: {nombre_grupo.upper()}")
    print(f"{'='*70}")
    print(f"Descripción: {descripcion}")
    print(f"Ciudades: {', '.join(ciudades)}")
    
    # Cargar datos
    print(f"\nCargando datos...")
    datos_grupo = cargar_datos_grupo_mejorado(ciudades, carpeta_precip, carpeta_meteo)
    
    if not datos_grupo:
        print(f"✗ No se pudieron cargar datos")
        return None
    
    print(f"Ciudades cargadas: {len(datos_grupo)}/{len(ciudades)}")
    
    # Combinar series
    print(f"\nCombinando series...")
    df_combined = combinar_series_grupo_mejorado(datos_grupo)
    
    if df_combined is None or len(df_combined) < 104:
        print(f"✗ Datos insuficientes")
        return None
    
    # Extraer serie de precipitación
    if 'Precip_mm' in df_combined.columns:
        serie = df_combined['Precip_mm']
    else:
        serie = df_combined.iloc[:, 0]
    
    print(f"Serie combinada: {len(serie)} semanas")
    print(f"Media: {serie.mean():.1f} mm, Std: {serie.std():.1f} mm")
    
    # Escalado
    scaler = StandardScaler()
    serie_scaled = scaler.fit_transform(serie.values.reshape(-1, 1))
    
    # Feature engineering con variables meteorológicas
    print(f"\nCreando features con variables meteorológicas...")
    features_df = create_enhanced_features_with_meteo(df_combined, serie_scaled, scaler)
    features_df = features_df.astype(np.float64)
    features_df = features_df.replace([np.inf, -np.inf], 0.0)
    
    print(f"Features creados: {features_df.shape[1]}")
    
    # Secuencias
    seq_length = 26
    data_array = features_df.values.astype(np.float32)
    X, y = create_sequences(data_array, seq_length=seq_length)
    
    X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
    y = np.nan_to_num(y, nan=0.0, posinf=3.0, neginf=-3.0)
    
    # Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"División: Train={len(X_train)}, Test={len(X_test)}")
    
    # Entrenar LSTM
    print(f"\nEntrenando LSTM...")
    model_lstm = create_lstm_model(
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
    
    # Predicciones
    y_pred_lstm_scaled = model_lstm.predict(X_test, verbose=0)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1))
    
    # SARIMAX
    print(f"Entrenando SARIMAX...")
    train_sarimax = serie[:train_size + seq_length]
    
    best_sarimax = None
    best_aic = float('inf')
    
    for order, seasonal_order in [((2,1,2),(1,1,1,52)), ((1,1,1),(1,1,1,52))]:
        try:
            model = SARIMAX(train_sarimax, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False, maxiter=300)
            if results.aic < best_aic:
                best_aic = results.aic
                best_sarimax = results
        except:
            continue
    
    if best_sarimax is None:
        model = SARIMAX(train_sarimax, order=(1,1,1), seasonal_order=(1,1,1,52))
        best_sarimax = model.fit(disp=False)
    
    y_pred_sarimax = best_sarimax.forecast(steps=len(X_test))
    
    # Optimizar ensemble
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    best_alpha = 0.5
    best_rmse = float('inf')
    
    for alpha in np.arange(0.0, 1.01, 0.05):
        if len(y_pred_sarimax) < len(y_pred_lstm):
            continue
        y_pred = alpha * y_pred_lstm.flatten() + (1-alpha) * y_pred_sarimax.values[:len(y_pred_lstm)]
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    y_pred_hybrid = best_alpha * y_pred_lstm.flatten() + (1-best_alpha) * y_pred_sarimax.values[:len(y_pred_lstm)]
    
    # Métricas
    mae = mean_absolute_error(y_test_orig, y_pred_hybrid)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_hybrid))
    
    mask = y_test_orig > 1
    mape = mean_absolute_percentage_error(y_test_orig[mask], y_pred_hybrid[mask]) * 100 if mask.any() else np.nan
    
    variability_ratio = np.std(y_pred_hybrid) / np.std(y_test_orig)
    
    print(f"\nMÉTRICAS {nombre_grupo}:")
    print(f"  MAE:  {mae:.2f} mm/semana")
    print(f"  RMSE: {rmse:.2f} mm/semana")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  Variabilidad ratio: {variability_ratio:.3f}")
    print(f"  Alpha: {best_alpha:.3f}")
    
    # Guardar
    Path('modelos').mkdir(exist_ok=True)
    model_lstm.save(f'modelos/modelo_{nombre_grupo}.h5')
    
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
        'num_features': data_array.shape[1],
        'con_variables_meteo': True
    }
    
    with open(f'modelos/metadata_{nombre_grupo}.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    best_sarimax.save(f'modelos/sarimax_{nombre_grupo}.pkl')
    
    print(f"✓ Modelos guardados")
    
    return metadata

def main():
    print("="*70)
    print("ENTRENAMIENTO CON VARIABLES METEOROLÓGICAS")
    print("Open-Meteo + Precipitación")
    print("="*70)
    
    # Verificar datos meteorológicos
    if not Path(CARPETA_METEO).exists():
        print(f"\n❌ ERROR: Carpeta {CARPETA_METEO} no existe")
        print("Ejecuta primero: python descargar_datos_meteo.py")
        return None
    
    archivos_meteo = list(Path(CARPETA_METEO).glob("*_meteo.csv"))
    print(f"Archivos meteorológicos encontrados: {len(archivos_meteo)}")
    
    resultados = {}
    
    for nombre_grupo, config_grupo in GRUPOS_CLIMATICOS.items():
        try:
            metadata = entrenar_modelo_grupo_mejorado(
                nombre_grupo, config_grupo, CARPETA_DATOS, CARPETA_METEO
            )
            if metadata:
                resultados[nombre_grupo] = metadata
        except Exception as e:
            print(f"\n✗ Error en {nombre_grupo}: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen
    print(f"\n{'='*70}")
    print(f"RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Modelos entrenados: {len(resultados)}/3")
    
    for nombre, metadata in resultados.items():
        print(f"\n{nombre}:")
        print(f"  Ciudades: {len(metadata['ciudades'])}")
        print(f"  Variables meteo: {'Sí' if metadata['con_variables_meteo'] else 'No'}")
        print(f"  MAE: {metadata['metricas']['mae']:.2f} mm")
        print(f"  RMSE: {metadata['metricas']['rmse']:.2f} mm")
        print(f"  MAPE: {metadata['metricas']['mape']:.1f}%")
    
    print(f"\n✓ Entrenamiento completado")
    
    return resultados

if __name__ == "__main__":
    resultados = main()