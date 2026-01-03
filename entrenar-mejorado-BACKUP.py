import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
import tensorflow as tf
from tensorflow import keras
import pickle

np.random.seed(42)
tf.random.set_seed(42)

CARPETA_DATOS = '/home/juanpucmm/modelo predectivo/formatoadecuado'
CARPETA_METEO = 'datos_meteo'

# ============================================================================
# CONFIGURACIÓN DE PERÍODOS Y GRUPOS
# ============================================================================

# SOLUCIÓN: Usar solo datos recientes y estables (2018-2025)
FECHA_INICIO_ENTRENAMIENTO = '2018-01-01'  # Evita períodos muy antiguos
FECHA_CORTE_VALIDACION = '2023-01-01'  # Train hasta 2023, test 2023-2025

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
# SOLUCIÓN 1: PREPROCESAMIENTO AVANZADO
# ============================================================================

class PreprocessorAvanzado:
    """Maneja outliers, estacionalidad y normalización robusta"""
    
    def __init__(self):
        self.scalers = {}
        self.estadisticas_por_epoca = {}
        
    def detectar_y_corregir_outliers(self, df, columna='Precip_mm', metodo='iqr'):
        """
        Detecta y suaviza outliers sin eliminarlos
        Método IQR: más robusto que z-score para datos con colas pesadas
        """
        serie = df[columna].copy()
        
        if metodo == 'iqr':
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1
            
            # Límites más permisivos para precipitación (3*IQR en lugar de 1.5)
            limite_inferior = Q1 - 3 * IQR
            limite_superior = Q3 + 3 * IQR
            
            # Suavizar outliers hacia los límites (winsorización)
            serie_corregida = serie.clip(lower=limite_inferior, upper=limite_superior)
            
            n_outliers = (serie != serie_corregida).sum()
            print(f"    Outliers suavizados: {n_outliers} ({n_outliers/len(serie)*100:.1f}%)")
            
        elif metodo == 'percentil':
            # Alternativa: usar percentiles
            p99 = serie.quantile(0.99)
            serie_corregida = serie.clip(upper=p99)
            
        df[f'{columna}_clean'] = serie_corregida
        return df
    
    def normalizar_por_epoca(self, df, columna='Precip_mm_clean'):
        """
        Normaliza por época del año para capturar estacionalidad
        Crucial para datos con alta variabilidad estacional
        """
        df = df.copy()
        df['mes'] = df.index.month
        
        # Definir épocas (ajusta según República Dominicana)
        # Mayo-Nov: época de huracanes (alta precipitación)
        # Dic-Abr: época seca
        df['epoca'] = df['mes'].apply(
            lambda x: 'humeda' if x in [5, 6, 7, 8, 9, 10, 11] else 'seca'
        )
        
        # Calcular estadísticas por época
        for epoca in ['humeda', 'seca']:
            mask = df['epoca'] == epoca
            datos_epoca = df.loc[mask, columna]
            
            self.estadisticas_por_epoca[epoca] = {
                'media': datos_epoca.mean(),
                'std': datos_epoca.std(),
                'mediana': datos_epoca.median(),
                'q25': datos_epoca.quantile(0.25),
                'q75': datos_epoca.quantile(0.75)
            }
            
            print(f"    Época {epoca}: media={datos_epoca.mean():.1f}mm, std={datos_epoca.std():.1f}mm")
        
        # Normalizar dentro de cada época
        df[f'{columna}_norm'] = 0.0
        
        for epoca in ['humeda', 'seca']:
            mask = df['epoca'] == epoca
            media = self.estadisticas_por_epoca[epoca]['media']
            std = self.estadisticas_por_epoca[epoca]['std']
            
            df.loc[mask, f'{columna}_norm'] = (df.loc[mask, columna] - media) / (std + 1e-6)
        
        return df
    
    def crear_features_estacionales_avanzadas(self, df):
        """Features temporales que capturan múltiples ciclos"""
        
        # Semana del año (ciclo anual)
        week = df.index.isocalendar().week.astype(float)
        df['week_sin1'] = np.sin(2 * np.pi * week / 52.0)
        df['week_cos1'] = np.cos(2 * np.pi * week / 52.0)
        df['week_sin2'] = np.sin(4 * np.pi * week / 52.0)  # Armónico
        df['week_cos2'] = np.cos(4 * np.pi * week / 52.0)
        
        # Mes (ciclo mensual)
        month = df.index.month.astype(float)
        df['month_sin1'] = np.sin(2 * np.pi * month / 12.0)
        df['month_cos1'] = np.cos(2 * np.pi * month / 12.0)
        df['month_sin2'] = np.sin(4 * np.pi * month / 12.0)
        df['month_cos2'] = np.cos(4 * np.pi * month / 12.0)
        
        # Tendencia lineal (captura cambio climático)
        df['tendencia_lineal'] = np.arange(len(df)) / len(df)
        
        # Indicador de época húmeda/seca
        df['es_epoca_humeda'] = df.index.month.isin([5, 6, 7, 8, 9, 10, 11]).astype(float)
        
        return df

# ============================================================================
# SOLUCIÓN 2: MODELO HÍBRIDO LSTM + SARIMAX
# ============================================================================

class ModeloHibridoAQUARIA:
    """
    Combina LSTM (patrones no lineales) + SARIMAX (estacionalidad)
    """
    
    def __init__(self, seq_length=26, alpha_lstm=0.6):
        self.seq_length = seq_length
        self.alpha_lstm = alpha_lstm  # Peso LSTM vs SARIMAX
        self.modelo_lstm = None
        self.modelo_sarimax = None
        self.scaler = RobustScaler()  # Más robusto que StandardScaler
        self.preprocessor = PreprocessorAvanzado()
        self.historia_entrenamiento = {}
        
    def crear_lstm_mejorado(self, input_shape):
        """
        LSTM con arquitectura optimizada para series temporales
        - Menos capas pero más neuronas
        - Dropout adaptativo
        - Batch Normalization para estabilidad
        """
        model = keras.Sequential([
            # Capa 1: Bidirectional LSTM con más neuronas
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    128, 
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.1,
                    kernel_regularizer=keras.regularizers.l2(0.001)
                ),
                input_shape=input_shape
            ),
            keras.layers.BatchNormalization(),
            
            # Capa 2: LSTM con atención implícita
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    96,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.1,
                    kernel_regularizer=keras.regularizers.l2(0.001)
                )
            ),
            keras.layers.BatchNormalization(),
            
            # Capa 3: LSTM final
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    64,
                    dropout=0.15,
                    recurrent_dropout=0.1,
                    kernel_regularizer=keras.regularizers.l2(0.001)
                )
            ),
            keras.layers.BatchNormalization(),
            
            # Capas densas con dropout progresivo
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            
            # Salida
            keras.layers.Dense(1, activation='linear')
        ])
        
        # Optimizador con gradient clipping (evita explosión de gradientes)
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999
        )
        
        # Huber loss: robusto a outliers
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mse']
        )
        
        return model
    
    def crear_sarimax_robusto(self, serie_train, max_intentos=3):
        """
        Prueba múltiples configuraciones SARIMAX
        Retorna el mejor modelo según AIC
        """
        configuraciones = [
            # (p,d,q) x (P,D,Q,s)
            ((2, 1, 2), (1, 1, 1, 52)),  # Configuración completa
            ((1, 1, 1), (1, 1, 1, 52)),  # Configuración simple
            ((2, 1, 1), (1, 1, 0, 52)),  # Sin MA estacional
            ((1, 0, 1), (1, 1, 1, 52)),  # Sin diferenciación regular
        ]
        
        mejor_modelo = None
        mejor_aic = float('inf')
        
        for orden, orden_estacional in configuraciones:
            try:
                modelo = SARIMAX(
                    serie_train,
                    order=orden,
                    seasonal_order=orden_estacional,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization='approximate_diffuse'
                )
                
                resultado = modelo.fit(
                    disp=False,
                    maxiter=500,
                    method='lbfgs'
                )
                
                if resultado.aic < mejor_aic:
                    mejor_aic = resultado.aic
                    mejor_modelo = resultado
                    print(f"    SARIMAX {orden}x{orden_estacional}: AIC={resultado.aic:.1f} ✓")
                    
            except Exception as e:
                print(f"    SARIMAX {orden}x{orden_estacional}: falló")
                continue
        
        if mejor_modelo is None:
            # Fallback: modelo simple
            print("    Usando modelo SARIMAX simple (fallback)")
            modelo = SARIMAX(serie_train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 52))
            mejor_modelo = modelo.fit(disp=False)
        
        return mejor_modelo
    
    def preparar_datos(self, df_combined, columna_target='Precip_mm'):
        """
        Pipeline completo de preprocesamiento
        """
        print("\n  📊 Preprocesamiento de datos:")
        
        # DEBUG: Ver columnas disponibles
        print(f"    Columnas disponibles: {list(df_combined.columns)[:10]}")
        
        # Encontrar columna de precipitación (puede tener sufijos)
        col_precip = None
        for col in df_combined.columns:
            if 'Precip_mm' in col or 'precip' in col.lower():
                col_precip = col
                break
        
        if col_precip is None:
            raise ValueError(f"No se encontró columna de precipitación. Columnas: {list(df_combined.columns)}")
        
        # Si la columna no es exactamente 'Precip_mm', renombrarla
        if col_precip != 'Precip_mm':
            df_combined = df_combined.rename(columns={col_precip: 'Precip_mm'})
            print(f"    Columna renombrada: {col_precip} → Precip_mm")
        
        columna_target = 'Precip_mm'  # Asegurar que usamos el nombre correcto
        
        # 1. Filtrar por fecha de inicio
        df = df_combined[df_combined.index >= FECHA_INICIO_ENTRENAMIENTO].copy()
        print(f"    Período: {df.index[0].date()} a {df.index[-1].date()} ({len(df)} semanas)")
        
        # 2. Detectar y corregir outliers
        df = self.preprocessor.detectar_y_corregir_outliers(df, columna_target)
        
        # 3. Normalización por época
        df = self.preprocessor.normalizar_por_epoca(df, columna_target + '_clean')
        
        # 4. Features estacionales
        df = self.preprocessor.crear_features_estacionales_avanzadas(df)
        
        # 5. Features meteorológicos (si existen)
        df = self._agregar_features_meteo(df)
        
        # 6. Features de lags y ventanas móviles
        df = self._crear_features_temporales(df)
        
        # 7. Limpiar NaNs
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        print(f"    Features totales: {len([c for c in df.columns if c not in ['mes', 'epoca']])} ")
        
        return df
    
    def _agregar_features_meteo(self, df):
        """Normaliza variables meteorológicas si existen"""
        
        # Mapeo flexible de nombres de columnas meteorológicas
        meteo_patterns = {
            'temperature_2m_mean': ['temperature_2m_mean', 'temperature_2m_mean_mean'],
            'temperature_2m_max': ['temperature_2m_max', 'temperature_2m_max_max'],
            'temperature_2m_min': ['temperature_2m_min', 'temperature_2m_min_min'],
            'temp_range': ['temp_range', 'temp_range_mean'],
            'windspeed_10m_max': ['windspeed_10m_max', 'windspeed_10m_max_max'],
            'pressure_msl_mean': ['pressure_msl_mean', 'pressure_msl_mean_mean'],
            'pressure_change': ['pressure_change', 'pressure_change_mean'],
            'pressure_change_std': ['pressure_change_std'],
            'et0_fao_evapotranspiration': ['et0_fao_evapotranspiration', 'et0_fao_evapotranspiration_sum']
        }
        
        features_meteo_agregados = 0
        
        for var_base, posibles_nombres in meteo_patterns.items():
            # Buscar la primera columna que coincida
            col_encontrada = None
            for nombre in posibles_nombres:
                if nombre in df.columns:
                    col_encontrada = nombre
                    break
            
            if col_encontrada:
                scaler = RobustScaler()
                values = df[col_encontrada].fillna(method='ffill').fillna(method='bfill').fillna(0)
                df[f'meteo_{var_base}'] = scaler.fit_transform(values.values.reshape(-1, 1))
                features_meteo_agregados += 1
                
                # Interacciones con precipitación normalizada
                if 'Precip_mm_clean_norm' in df.columns:
                    df[f'interact_{var_base}'] = df[f'meteo_{var_base}'] * df['Precip_mm_clean_norm']
        
        if features_meteo_agregados > 0:
            print(f"    Features meteorológicos agregados: {features_meteo_agregados}")
        
        return df
    
    def _crear_features_temporales(self, df):
        """Lags, ventanas móviles y tendencias"""
        
        target_col = 'Precip_mm_clean_norm' if 'Precip_mm_clean_norm' in df.columns else 'Precip_mm_clean'
        
        # Lags múltiples
        for lag in [1, 2, 4, 8, 12, 26, 52]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Ventanas móviles (media y std)
        for window in [4, 8, 12, 26]:
            df[f'ma_{window}'] = df[target_col].rolling(window, min_periods=1).mean()
            df[f'std_{window}'] = df[target_col].rolling(window, min_periods=1).std()
        
        # Tendencia local
        df['tendencia_4w'] = df[target_col].rolling(4, center=True, min_periods=1).mean()
        df['tendencia_12w'] = df[target_col].rolling(12, center=True, min_periods=1).mean()
        
        # Aceleración (segunda derivada)
        df['aceleracion'] = df[target_col].diff().diff()
        
        return df
    
    def entrenar(self, df_combined, columna_target='Precip_mm'):
        """
        Entrena el modelo híbrido
        """
        # Preprocesar
        df = self.preparar_datos(df_combined, columna_target)
        
        # Serie target
        serie_target = df[f'{columna_target}_clean'].copy()
        
        # Train/test split temporal
        fecha_corte = pd.to_datetime(FECHA_CORTE_VALIDACION)
        train_mask = df.index < fecha_corte
        test_mask = df.index >= fecha_corte
        
        serie_train = serie_target[train_mask]
        serie_test = serie_target[test_mask]
        
        print(f"\n  📈 División temporal:")
        print(f"    Train: {serie_train.index[0].date()} a {serie_train.index[-1].date()} ({len(serie_train)} semanas)")
        print(f"    Test:  {serie_test.index[0].date()} a {serie_test.index[-1].date()} ({len(serie_test)} semanas)")
        
        # ==================== ENTRENAR SARIMAX ====================
        print(f"\n  🔄 Entrenando SARIMAX...")
        self.modelo_sarimax = self.crear_sarimax_robusto(serie_train)
        
        # ==================== ENTRENAR LSTM ====================
        print(f"\n  🧠 Entrenando LSTM...")
        
        # Preparar features para LSTM
        feature_cols = [c for c in df.columns if c not in ['mes', 'epoca', columna_target, f'{columna_target}_clean']]
        features_array = df[feature_cols].values.astype(np.float32)
        
        # Escalar features
        features_scaled = self.scaler.fit_transform(features_array)
        
        # Crear secuencias
        X, y = self._crear_secuencias(features_scaled, serie_target.values)
        
        # Split temporal de secuencias
        n_train = len(serie_train) - self.seq_length
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        print(f"    Secuencias: Train={len(X_train)}, Test={len(X_test)}")
        
        # Crear modelo LSTM
        self.modelo_lstm = self.crear_lstm_mejorado(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Entrenar
        history = self.modelo_lstm.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1
        )
        
        self.historia_entrenamiento = history.history
        
        # ==================== OPTIMIZAR ENSEMBLE ====================
        print(f"\n  ⚖️  Optimizando combinación híbrida...")
        
        # Predicciones en test set
        y_pred_lstm = self.modelo_lstm.predict(X_test, verbose=0).flatten()
        y_pred_sarimax = self.modelo_sarimax.forecast(steps=len(y_test))
        
        # Buscar mejor alpha
        mejor_alpha = self._optimizar_alpha(y_test, y_pred_lstm, y_pred_sarimax.values)
        self.alpha_lstm = mejor_alpha
        
        print(f"    Alpha óptimo: {mejor_alpha:.3f} (LSTM={mejor_alpha:.1%}, SARIMAX={(1-mejor_alpha):.1%})")
        
        # ==================== EVALUACIÓN FINAL ====================
        y_pred_hybrid = mejor_alpha * y_pred_lstm + (1 - mejor_alpha) * y_pred_sarimax.values
        
        mae = mean_absolute_error(y_test, y_pred_hybrid)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))
        
        # MAPE solo para valores > 1mm
        mask_mape = y_test > 1
        mape = mean_absolute_percentage_error(y_test[mask_mape], y_pred_hybrid[mask_mape]) * 100 if mask_mape.any() else np.nan
        
        variability_ratio = np.std(y_pred_hybrid) / (np.std(y_test) + 1e-6)
        
        print(f"\n  📊 MÉTRICAS FINALES:")
        print(f"    MAE:  {mae:.2f} mm/semana")
        print(f"    RMSE: {rmse:.2f} mm/semana")
        print(f"    MAPE: {mape:.1f}%")
        print(f"    Variabilidad ratio: {variability_ratio:.3f}")
        
        # Guardar métricas Y PREDICCIONES DE VALIDACIÓN
        self.metricas = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'variability_ratio': variability_ratio
        }
        
        # ⭐ NUEVO: Guardar predicciones de validación para calibración
        self.validacion = {
            'y_test': y_test,
            'y_pred_lstm': y_pred_lstm,
            'y_pred_sarimax': y_pred_sarimax.values,
            'y_pred_hybrid': y_pred_hybrid,
            'fechas_test': serie_test.index[len(serie_test) - len(y_test):]
        }
        
        return self.metricas
    
    def _crear_secuencias(self, features, target):
        """Crea secuencias para LSTM"""
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i + self.seq_length])
            y.append(target[i + self.seq_length])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def _optimizar_alpha(self, y_true, y_pred_lstm, y_pred_sarimax):
        """Encuentra el mejor peso para combinar LSTM y SARIMAX"""
        mejor_alpha = 0.5
        mejor_rmse = float('inf')
        
        for alpha in np.arange(0.0, 1.01, 0.05):
            y_pred = alpha * y_pred_lstm + (1 - alpha) * y_pred_sarimax
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            if rmse < mejor_rmse:
                mejor_rmse = rmse
                mejor_alpha = alpha
        
        return mejor_alpha
    
    def predecir(self, pasos=12):
        """Genera predicciones futuras (implementar según necesidad)"""
        # TODO: Implementar predicción multi-step
        pass

# ============================================================================
# FUNCIONES AUXILIARES (CARGA DE DATOS)
# ============================================================================

def cargar_datos_con_meteo(ciudad, carpeta_precip, carpeta_meteo):
    """Carga datos de precipitación combinados con variables meteorológicas"""
    
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
    
    archivo_meteo = Path(carpeta_meteo) / f"{ciudad}_meteo.csv"
    
    if not archivo_meteo.exists():
        return df_precip
    
    df_meteo = pd.read_csv(archivo_meteo, parse_dates=['date'])
    df_meteo = df_meteo.set_index('date')
    
    df_combined = df_precip.join(df_meteo, how='left')
    
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
            
            # Resample a semanal
            agg_dict = {'Precip_mm': 'sum'}
            
            if 'temperature_2m_mean' in df.columns:
                agg_dict.update({
                    'temperature_2m_mean': 'mean',
                    'temperature_2m_max': 'max',
                    'temperature_2m_min': 'min',
                    'temp_range': 'mean',
                    'windspeed_10m_max': 'max',
                    'pressure_msl_mean': 'mean',
                    'pressure_change': ['mean', 'std'],
                    'et0_fao_evapotranspiration': 'sum'
                })
            
            semanal = df.resample('W-SUN').agg(agg_dict)
            
            if isinstance(semanal.columns, pd.MultiIndex):
                semanal.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                   for col in semanal.columns.values]
            
            datos_grupo.append({'ciudad': ciudad, 'datos': semanal})
            print(f"  ✓ {ciudad}: {len(semanal)} semanas")
            
        except Exception as e:
            print(f"  ✗ Error en {ciudad}: {e}")
    
    return datos_grupo

def combinar_series_grupo_mejorado(datos_grupo):
    """Combina series preservando variables meteorológicas"""
    
    if not datos_grupo:
        return None
    
    fechas_comunes = set(datos_grupo[0]['datos'].index)
    for dato in datos_grupo[1:]:
        fechas_comunes &= set(dato['datos'].index)
    
    fechas_comunes = sorted(list(fechas_comunes))
    
    if len(fechas_comunes) < 52:
        return None
    
    df_combined = pd.DataFrame(index=fechas_comunes)
    
    # Mapeo de nombres de columnas con sufijos a nombres limpios
    col_mapping = {}
    
    for col in datos_grupo[0]['datos'].columns:
        valores = []
        for dato in datos_grupo:
            if col in dato['datos'].columns:
                serie_aligned = dato['datos'][col].reindex(fechas_comunes)
                valores.append(serie_aligned.values)
        
        if valores:
            # Limpiar nombre de columna (quitar sufijos de agregación)
            col_limpio = col
            
            # Si tiene sufijos de agregación, limpiarlos
            sufijos = ['_sum', '_mean', '_max', '_min', '_std']
            for sufijo in sufijos:
                if col.endswith(sufijo):
                    col_limpio = col[:-len(sufijo)]
                    break
            
            # Caso especial: Precip_mm
            if col in ['Precip_mm', 'Precip_mm_sum']:
                col_limpio = 'Precip_mm'
            
            df_combined[col_limpio] = np.mean(valores, axis=0)
            col_mapping[col] = col_limpio
    
    # Debug: imprimir columnas encontradas
    print(f"  📋 Columnas en serie combinada: {list(df_combined.columns)[:10]}")
    
    return df_combined

# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def entrenar_modelo_grupo_hibrido(nombre_grupo, config_grupo, carpeta_precip, carpeta_meteo):
    """Entrena modelo híbrido mejorado con preprocesamiento avanzado"""
    
    ciudades = config_grupo['ciudades']
    descripcion = config_grupo['descripcion']
    
    print(f"\n{'='*70}")
    print(f"🌊 ENTRENANDO: {nombre_grupo.upper()}")
    print(f"{'='*70}")
    print(f"Descripción: {descripcion}")
    print(f"Ciudades: {', '.join(ciudades)}")
    
    # Cargar datos
    print(f"\n📂 Cargando datos...")
    datos_grupo = cargar_datos_grupo_mejorado(ciudades, carpeta_precip, carpeta_meteo)
    
    if not datos_grupo:
        print(f"✗ No se pudieron cargar datos")
        return None
    
    print(f"Ciudades cargadas: {len(datos_grupo)}/{len(ciudades)}")
    
    # Combinar series
    print(f"\n🔗 Combinando series...")
    df_combined = combinar_series_grupo_mejorado(datos_grupo)
    
    if df_combined is None or len(df_combined) < 104:
        print(f"✗ Datos insuficientes")
        return None
    
    print(f"Serie combinada: {len(df_combined)} semanas totales")
    print(f"Rango: {df_combined.index[0].date()} a {df_combined.index[-1].date()}")
    
    # Entrenar modelo híbrido
    modelo = ModeloHibridoAQUARIA(seq_length=26, alpha_lstm=0.6)
    metricas = modelo.entrenar(df_combined, columna_target='Precip_mm')
    
    # Guardar modelo
    print(f"\n💾 Guardando modelos...")
    Path('modelos').mkdir(exist_ok=True)
    
    modelo.modelo_lstm.save(f'modelos/modelo_{nombre_grupo}_hibrido.h5')
    modelo.modelo_sarimax.save(f'modelos/sarimax_{nombre_grupo}_hibrido.pkl')
    
    metadata = {
        'nombre': nombre_grupo,
        'ciudades': ciudades,
        'descripcion': descripcion,
        'preprocessor': modelo.preprocessor,
        'scaler': modelo.scaler,
        'alpha_lstm': modelo.alpha_lstm,
        'metricas': metricas,
        'seq_length': modelo.seq_length,
        'fecha_inicio_entrenamiento': FECHA_INICIO_ENTRENAMIENTO,
        'fecha_corte_validacion': FECHA_CORTE_VALIDACION,
        'con_variables_meteo': True,
        'version': '2.0_hibrido_mejorado',
        'validacion': modelo.validacion  # ⭐ NUEVO: Datos para calibración
    }
    
    with open(f'modelos/metadata_{nombre_grupo}_hibrido.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Modelos guardados exitosamente")
    
    return metadata

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🌊 AQUARIA - ENTRENAMIENTO MODELO HÍBRIDO MEJORADO")
    print("="*70)
    print(f"📅 Período de entrenamiento: {FECHA_INICIO_ENTRENAMIENTO} en adelante")
    print(f"✂️  Corte validación: {FECHA_CORTE_VALIDACION}")
    print("="*70)
    
    # Verificar datos meteorológicos
    if not Path(CARPETA_METEO).exists():
        print(f"\n❌ ERROR: Carpeta {CARPETA_METEO} no existe")
        print("Ejecuta primero: python descargar_datos_meteo.py")
        return None
    
    archivos_meteo = list(Path(CARPETA_METEO).glob("*_meteo.csv"))
    print(f"\n🌤️  Archivos meteorológicos encontrados: {len(archivos_meteo)}")
    
    resultados = {}
    
    for nombre_grupo, config_grupo in GRUPOS_CLIMATICOS.items():
        try:
            metadata = entrenar_modelo_grupo_hibrido(
                nombre_grupo, config_grupo, CARPETA_DATOS, CARPETA_METEO
            )
            if metadata:
                resultados[nombre_grupo] = metadata
        except Exception as e:
            print(f"\n✗ Error en {nombre_grupo}: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Modelos entrenados exitosamente: {len(resultados)}/3\n")
    
    for nombre, metadata in resultados.items():
        print(f"🌍 {nombre.upper()}")
        print(f"   Ciudades: {len(metadata['ciudades'])}")
        print(f"   Período: {metadata['fecha_inicio_entrenamiento']} - presente")
        print(f"   Alpha LSTM: {metadata['alpha_lstm']:.3f}")
        print(f"   MAE:  {metadata['metricas']['mae']:.2f} mm")
        print(f"   RMSE: {metadata['metricas']['rmse']:.2f} mm")
        print(f"   MAPE: {metadata['metricas']['mape']:.1f}%")
        print(f"   Variabilidad: {metadata['metricas']['variability_ratio']:.3f}")
        print()
    
    print(f"✅ Entrenamiento completado exitosamente")
    print(f"📁 Modelos guardados en: ./modelos/")
    
    return resultados

if __name__ == "__main__":
    resultados = main()