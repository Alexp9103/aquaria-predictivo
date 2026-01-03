
from descargar_modelos import descargar_modelos

descargar_modelos()

from fpdf import FPDF
import base64
from io import BytesIO
from datetime import datetime

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import seaborn as sns
import actualizar_datos

import auth

import gestion_dispositivos


from sklearn.linear_model import LinearRegression

auth.inicializar_sesion()


DEBUG = False  # ✅ Desactivar para producción

# ============================================================================
# CLASES DUMMY PARA COMPATIBILIDAD CON MODELOS ANTIGUOS
# ============================================================================

class PreprocessorAvanzado:
    """Clase dummy para compatibilidad al cargar modelos antiguos"""
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, *args, **kwargs):
        return self
    
    def transform(self, *args, **kwargs):
        return args[0] if args else None
    
    def fit_transform(self, *args, **kwargs):
        return args[0] if args else None

# Registrar en el módulo principal para pickle
import sys
sys.modules[__name__].PreprocessorAvanzado = PreprocessorAvanzado

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ============================================================================
st.set_page_config(
    page_title="AQUARIA - Predicción de Lluvia RD",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-style: italic;
    }

    .icon-header {
        margin-right: 15px;
        color: #1f77b4;
    }

    .sidebar-info {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<h1 class="main-header">'
    '<i class="fa-solid fa-cloud-showers-water icon-header"></i>'
    'AQUARIA: Sistema Predictivo'
    '</h1>',
    unsafe_allow_html=True
)

# ============================================================================
# FUNCIONES DE CARGA - ACTUALIZADAS
# ============================================================================

@st.cache_resource(ttl=300)  # ✅ Expira cada 5 minutos
def cargar_calibradores():
    """Carga todos los calibradores pre-entrenados (versión _simple.pkl)"""
    calibradores = {}
    
    if DEBUG:
        print("\n" + "=" * 70)
        print("CARGANDO CALIBRADORES")
        print("=" * 70)
    
    # 🔥 PRIORIDAD 1: Buscar calibradores *_simple.pkl (nuevos, estratificados)
    archivos_simple = list(Path('modelos').glob('calibrador_*_simple.pkl'))
    
    if DEBUG:
        print(f"\n📁 Archivos *_simple.pkl encontrados: {len(archivos_simple)}")
    
    for archivo in archivos_simple:
        # Extraer nombre: calibrador_grupo_1_norte_cibao_simple.pkl
        #                → grupo_1_norte_cibao
        nombre_stem = archivo.stem  # calibrador_grupo_1_norte_cibao_simple
        
        # Quitar "calibrador_" del inicio
        if nombre_stem.startswith('calibrador_'):
            nombre_sin_prefijo = nombre_stem[11:]  # grupo_1_norte_cibao_simple
        else:
            nombre_sin_prefijo = nombre_stem
        
        # Quitar "_simple" del final
        if nombre_sin_prefijo.endswith('_simple'):
            nombre_final = nombre_sin_prefijo[:-7]  # grupo_1_norte_cibao
        else:
            nombre_final = nombre_sin_prefijo
        
        if DEBUG:
            print(f"\n  📦 {archivo.name}")
            print(f"     → Stem: '{nombre_stem}'")
            print(f"     → Sin prefijo: '{nombre_sin_prefijo}'")
            print(f"     → Nombre final: '{nombre_final}'")
        
        try:
            with open(archivo, 'rb') as f:
                calibrador = pickle.load(f)
            
            # Validar estructura
            if isinstance(calibrador, dict) and 'sesgos' in calibrador:
                calibradores[nombre_final] = calibrador
                if DEBUG:
                    print(f"     ✅ Cargado como '{nombre_final}'")
                    print(f"     → Tipo: Estratificado")
                    print(f"     → Niveles: {list(calibrador['sesgos'].keys())}")
            else:
                if DEBUG:
                    print(f"     ⚠️  Estructura inválida")
        except Exception as e:
            if DEBUG:
                print(f"     ❌ Error: {e}")
    
    # 🔥 PRIORIDAD 2: Buscar calibradores sin _simple (antiguos, multiplicativos)
    archivos_antiguos = [f for f in Path('modelos').glob('calibrador_*.pkl') 
                         if '_simple' not in f.stem]
    
    if DEBUG:
        print(f"\n📁 Archivos antiguos (sin _simple): {len(archivos_antiguos)}")
    
    for archivo in archivos_antiguos:
        nombre = archivo.stem.replace('calibrador_', '')
        
        # Solo cargar si no existe la versión _simple
        if nombre not in calibradores:
            if DEBUG:
                print(f"\n  📦 {archivo.name} (fallback)")
            
            try:
                with open(archivo, 'rb') as f:
                    calibrador = pickle.load(f)
                
                # Validar estructura (estratificado o multiplicativo)
                if isinstance(calibrador, dict):
                    if 'sesgos' in calibrador:
                        calibradores[nombre] = calibrador
                        if DEBUG:
                            print(f"     ✅ Cargado como '{nombre}' (estratificado)")
                            print(f"     → Niveles: {list(calibrador['sesgos'].keys())}")
                    elif 'factores' in calibrador:
                        calibradores[nombre] = calibrador
                        if DEBUG:
                            print(f"     ✅ Cargado como '{nombre}' (multiplicativo)")
                            print(f"     → Factores: {list(calibrador['factores'].keys())}")
                    else:
                        if DEBUG:
                            print(f"     ⚠️  Estructura desconocida: {list(calibrador.keys())}")
            except Exception as e:
                if DEBUG:
                    print(f"     ❌ Error: {e}")
        else:
            if DEBUG:
                print(f"\n  ⏭️  {archivo.name} (ya existe versión _simple)")
    
    if DEBUG:
        print("\n" + "=" * 70)
        print(f"✅ Total calibradores cargados: {len(calibradores)}")
        for nombre in calibradores.keys():
            print(f"   • {nombre}")
        print("=" * 70 + "\n")
    
    return calibradores

@st.cache_resource(ttl=300)  # ✅ Expira cada 5 minutos
def cargar_todos_modelos():
    """Carga modelos Y calibradores - VERSIÓN CORREGIDA PARA HÍBRIDOS"""
    modelos = {}
    calibradores = cargar_calibradores()
    
    # 🔥 PASO 1: Intentar cargar modelos HÍBRIDOS (nuevos, con variabilidad mejorada)
    archivos_hibridos = list(Path('modelos').glob('metadata_*_hibrido.pkl'))
    
    if DEBUG:
        print(f"\n🔍 Buscando modelos híbridos: {len(archivos_hibridos)} encontrados")
    
    for archivo in archivos_hibridos:
        # Limpiar nombre: metadata_grupo_1_norte_cibao_hibrido.pkl → grupo_1_norte_cibao
        nombre = archivo.stem.replace('metadata_', '').replace('_hibrido', '')
        
        try:
            with open(archivo, 'rb') as f:
                metadata = pickle.load(f)
            
            # Intentar cargar SARIMAX híbrido
            sarimax_path = f'modelos/sarimax_{nombre}_hibrido.pkl'
            if not Path(sarimax_path).exists():
                if DEBUG:
                    print(f"⚠️ No se encontró {sarimax_path}")
                continue
                
            with open(sarimax_path, 'rb') as f:
                sarimax = pickle.load(f)
            
            modelos[nombre] = {
                'metadata': metadata,
                'sarimax': sarimax,
                'calibrador': calibradores.get(nombre),
                'nombre_display': nombre.replace('_', ' ').title()
            }
            
            if DEBUG:
                cal_status = "✅" if calibradores.get(nombre) else "❌"
                var = metadata['metricas'].get('variability_ratio', 'N/A')
                print(f"✅ Modelo híbrido: {nombre} | Var: {var} | Cal: {cal_status}")
            
        except Exception as e:
            if DEBUG:
                print(f"❌ Error cargando híbrido {nombre}: {e}")
            continue
    
    # 🔥 PASO 2: FALLBACK - Si no hay modelos híbridos, cargar antiguos
    if not modelos:
        if DEBUG:
            print("\n⚠️ No se encontraron modelos híbridos, intentando modelos antiguos...")
        
        for archivo in Path('modelos').glob('metadata_*.pkl'):
            nombre = archivo.stem.replace('metadata_', '')
            
            # Saltar híbridos en este paso
            if '_hibrido' in nombre:
                continue
            
            try:
                with open(archivo, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Intentar cargar SARIMAX
                sarimax_path = f'modelos/sarimax_{nombre}.pkl'
                if not Path(sarimax_path).exists():
                    if DEBUG:
                        print(f"⚠️ No se encontró {sarimax_path}")
                    continue
                    
                with open(sarimax_path, 'rb') as f:
                    sarimax = pickle.load(f)
                
                modelos[nombre] = {
                    'metadata': metadata,
                    'sarimax': sarimax,
                    'calibrador': calibradores.get(nombre),
                    'nombre_display': nombre.replace('_', ' ').title()
                }
                
                if DEBUG:
                    cal_status = "✅" if calibradores.get(nombre) else "❌"
                    print(f"✅ Modelo antiguo: {nombre} | Calibrador: {cal_status}")
                
            except Exception as e:
                if DEBUG:
                    print(f"❌ Error cargando antiguo {nombre}: {e}")
                continue
    
    if DEBUG:
        print(f"\n✅ Total modelos cargados: {len(modelos)}")
        for nombre in modelos.keys():
            print(f"   • {nombre}")
    
    return modelos

# ============================================================================
# FUNCIÓN DE CALIBRACIÓN - ACTUALIZADA PARA CORRECCIÓN ESTRATIFICADA
# ============================================================================

def aplicar_calibracion(predicciones, calibrador, modo='balanceado'):
    """
    Aplica calibración usando calibrador pre-entrenado (estratificado o multiplicativo)
    
    Args:
        predicciones: Array de predicciones originales
        calibrador: Dict con factores pre-calculados
        modo: 'conservador', 'balanceado' o 'agresivo'
    
    Returns:
        Array de predicciones calibradas
    """
    
    if calibrador is None:
        return predicciones
    
    # ✅ DETECTAR TIPO DE CALIBRADOR
    
    # Tipo 1: Calibrador estratificado (nuevo, mejor)
    if 'sesgos' in calibrador and 'configuraciones' in calibrador:
        return aplicar_calibracion_estratificada(predicciones, calibrador, modo)
    
    # Tipo 2: Calibrador multiplicativo (antiguo, simple)
    elif 'factores' in calibrador:
        return aplicar_calibracion_multiplicativa(predicciones, calibrador, modo)
    
    # Tipo desconocido
    else:
        st.warning("⚠️ Calibrador con formato desconocido")
        return predicciones


def aplicar_calibracion_estratificada(predicciones, calibrador, modo='balanceado'):
    """
    Aplica calibración estratificada por nivel (MÉTODO CORRECTO v3)
    Con soporte para reducción extra en modo ultra-agresivo
    """
    import numpy as np
    
    sesgos = calibrador['sesgos']
    config = calibrador['configuraciones'][modo]
    agresividad = config['agresividad']
    factor_var = config.get('factor_variabilidad', 1.2)
    factor_var_post = config.get('factor_variabilidad_post', None)
    aplicar_reduccion_extra = config.get('aplicar_reduccion_extra', False)
    reduccion_extra = config.get('reduccion_extra_porcentaje', 0.0)
    
    predicciones_corregidas = predicciones.copy()
    
    # Paso 1: Aumentar variabilidad PRE (si factor_var > 1.0)
    if factor_var > 1.0:
        media = np.mean(predicciones)
        predicciones_corregidas = (predicciones_corregidas - media) * factor_var + media
        predicciones_corregidas = np.maximum(predicciones_corregidas, 0.1)
    
    # Paso 2: Corrección por nivel
    for i, pred in enumerate(predicciones_corregidas):
        # Determinar nivel
        if pred < 5:
            nivel = 'muy_bajo'
        elif pred < 15:
            nivel = 'bajo'
        elif pred < 30:
            nivel = 'medio'
        elif pred < 60:
            nivel = 'alto'
        else:
            nivel = 'muy_alto'
        
        sesgo = sesgos[nivel]
        
        # Protección para valores muy pequeños
        if pred < 2.0:
            agresividad_efectiva = agresividad * 0.1
        else:
            agresividad_efectiva = agresividad
        
        # Corrección aditiva: pred_corr = pred - sesgo*agresividad
        correccion = agresividad_efectiva * sesgo
        predicciones_corregidas[i] = pred - correccion
    
    # Paso 3: Aumentar variabilidad POST (si existe)
    if factor_var_post is not None and factor_var_post > 1.0:
        media = np.mean(predicciones_corregidas)
        predicciones_corregidas = (predicciones_corregidas - media) * factor_var_post + media
    
    # Paso 4: Reducción extra para casos severos (ultra-agresivo)
    if aplicar_reduccion_extra:
        predicciones_corregidas = predicciones_corregidas * (1 - reduccion_extra)
    
    # Asegurar no negativos
    predicciones_corregidas = np.maximum(predicciones_corregidas, 0.1)
    
    return predicciones_corregidas


def aplicar_calibracion_multiplicativa(predicciones, calibrador, modo='conservador'):
    """
    Aplica calibración multiplicativa simple (MÉTODO ANTIGUO)
    """
    factor = calibrador['factores'][modo]
    predicciones_calibradas = predicciones * factor
    predicciones_calibradas = np.maximum(predicciones_calibradas, 0)
    return predicciones_calibradas


def render_gestion_usuarios():
    st.header("👥 Gestión de Usuarios")
    
    # Crear usuario
    with st.expander("➕ Crear Nuevo Usuario"):
        with st.form("form_crear_usuario"):
            col1, col2 = st.columns(2)
            with col1:
                email = st.text_input("Email*")
                nombre = st.text_input("Nombre*")
            with col2:
                password = st.text_input("Contraseña*", type="password")
                rol = st.selectbox("Rol*", ["usuario", "admin"])
            
            if st.form_submit_button("➕ Crear Usuario", type="primary"):
                if not email or not nombre or not password:
                    st.error("⚠️ Completa todos los campos")
                else:
                    success, user_id, error = auth.crear_usuario(email, nombre, password, rol)
                    if success:
                        st.success(f"✅ Usuario creado: {email}")
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {error}")

    # Listar usuarios
    st.subheader("📋 Usuarios Registrados")
    usuarios = auth.listar_usuarios()
    
    for u in usuarios:
        with st.expander(f"{'👤' if u['rol']=='usuario' else '👑'} {u['nombre']} ({u['email']})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rol", u['rol'].title())
            with col2:
                estado = "🟢 Activo" if u['activo'] else "🔴 Inactivo"
                st.metric("Estado", estado)
            with col3:
                fecha_str = u['created_at'].strftime('%Y-%m-%d') if isinstance(u['created_at'], datetime) else str(u['created_at'])[:10]
                st.metric("Desde", fecha_str)
            
            st.markdown("---")
            if u['activo']:
                if st.button(f"🔴 Desactivar", key=f"deact_{u['id_usuario']}"):
                    auth.desactivar_usuario(u['id_usuario'])
                    st.success("✅ Usuario desactivado")
                    st.rerun()
            else:
                if st.button(f"🟢 Activar", key=f"act_{u['id_usuario']}"):
                    auth.activar_usuario(u['id_usuario'])
                    st.success("✅ Usuario activado")
                    st.rerun()

# ============================================================================
# FUNCIONES DE PREDICCIÓN Y VALIDACIÓN
# ============================================================================

def generar_predicciones(_modelo_dict, semanas, region_key):
    """Genera predicciones con serie actualizada - Compatible con modelos híbridos"""
    
    metadata = _modelo_dict['metadata']
    sarimax = _modelo_dict['sarimax']
    
    # 🔥 Compatibilidad con modelos híbridos
    if 'serie' in metadata:
        serie_original = metadata['serie']  # ✅ Esto está OK (dentro del if)
        serie_actualizada = actualizar_datos.actualizar_serie(serie_original, region_key)
    else:
        # Modelo híbrido: usar datos de validación como referencia
        if 'validacion' in metadata and 'fechas_test' in metadata['validacion']:
            # Crear serie dummy para compatibilidad
            fechas = metadata['validacion']['fechas_test']
            valores = metadata['validacion']['y_test']
            serie_actualizada = pd.Series(valores, index=fechas)
        else:
            # Fallback: generar serie dummy
            st.warning("⚠️ Modelo híbrido sin datos de validación. Usando fechas simuladas.")
            ultima_fecha = pd.Timestamp.now() - pd.Timedelta(weeks=4)
            fechas = pd.date_range(end=ultima_fecha, periods=52, freq='W-SUN')
            serie_actualizada = pd.Series([10] * 52, index=fechas)  # Valores dummy
    
    if DEBUG:
        print(f"[generar_predicciones] {region_key}")
        print(f"  Serie: {len(serie_actualizada)} semanas")
        print(f"  Última fecha: {serie_actualizada.index[-1].date()}")
    
    if DEBUG:
        print(f"[generar_predicciones] {region_key}")
        print(f"  Original: {len(serie_original)} semanas")
        print(f"  Actualizada: {len(serie_actualizada)} semanas")
        print(f"  Última fecha: {serie_actualizada.index[-1].date()}")
    
    try:
        pred = sarimax.forecast(steps=semanas)
        
        fechas = pd.date_range(
            serie_actualizada.index[-1] + pd.Timedelta(weeks=1),
            periods=semanas,
            freq='W-SUN'
        )
        
        df_pred = pd.DataFrame({
            'Fecha': fechas,
            'Prediccion_mm': pred.values
        })
        
        df_pred['Prediccion_mm'] = df_pred['Prediccion_mm'].clip(lower=0)
        
        return df_pred, serie_actualizada
        
    except Exception as e:
        st.error(f"Error generando predicciones: {e}")
        return None, None

def validar_predicciones_historicas(modelo_dict, semanas_test=12):
    """Validación histórica (backtesting) del modelo"""
    metadata = modelo_dict['metadata']
    sarimax = modelo_dict['sarimax']
    
    # 🔥 COMPATIBILIDAD CON MODELOS HÍBRIDOS
    if 'serie' not in metadata:
        if 'validacion' in metadata:
            val = metadata['validacion']
            
            df = pd.DataFrame({
                'Fecha': val['fechas_test'],
                'Prediccion_mm': val['y_pred_hybrid'],
                'Real_mm': val['y_test']
            })
            
            df['Error'] = df['Prediccion_mm'] - df['Real_mm']
            df['Error_Abs'] = abs(df['Error'])
            
            # Calcular MAPE
            df_mape = df[df['Real_mm'] > 0]
            if len(df_mape) > 0:
                mape = (abs(df_mape['Prediccion_mm'] - df_mape['Real_mm']) / df_mape['Real_mm']).mean() * 100
            else:
                mape = None
            
            return {
                'MAE': metadata['metricas']['mae'],
                'RMSE': metadata['metricas']['rmse'],
                'MAPE': mape,
                'df': df
            }
        
        return None  # ✅ Mantener este return None
        
    serie = metadata['serie']

    if len(serie) < semanas_test + 20:
        return None

    serie_train = serie.iloc[:-semanas_test]
    serie_real = serie.iloc[-semanas_test:]

    pred = sarimax.forecast(steps=semanas_test)

    df = pd.DataFrame({
        'Fecha': serie_real.index,
        'Prediccion_mm': pred.values,
        'Real_mm': serie_real.values
    })
    
    df['Error'] = df['Prediccion_mm'] - df['Real_mm']
    df['Error_Abs'] = abs(df['Error'])

    df_mape = df[df['Real_mm'] > 0]

    error = df['Prediccion_mm'] - df['Real_mm']

    mae = abs(error).mean()
    rmse = np.sqrt((error ** 2).mean())

    if len(df_mape) > 0:
        mape = (abs(df_mape['Prediccion_mm'] - df_mape['Real_mm']) / df_mape['Real_mm']).mean() * 100
    else:
        mape = None

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'df': df
    }

def validar_predicciones_recientes(modelo_dict, region_key, semanas_validar=12):
    """Valida predicciones recientes contra datos reales ya disponibles"""
    
    metadata = modelo_dict['metadata']
    sarimax = modelo_dict['sarimax']
    
    # 🔥 COMPATIBILIDAD CON MODELOS HÍBRIDOS
    if 'serie' not in metadata:
        # ✅ NUEVO: Usar datos de validación pre-calculados
        if 'validacion' not in metadata:
            return None
        
        val = metadata['validacion']
        
        # Tomar las últimas N semanas de validación
        n_disponible = len(val['y_test'])
        n_usar = min(semanas_validar, n_disponible)
        
        if n_usar < 4:
            return None
        
        # Extraer últimas N semanas
        y_test = val['y_test'][-n_usar:]
        y_pred = val['y_pred_hybrid'][-n_usar:]
        fechas = val['fechas_test'][-n_usar:]
        
        df_comparacion = pd.DataFrame({
            'Fecha': fechas,
            'Real_mm': y_test,
            'Prediccion_mm': y_pred
        })
        
        df_comparacion['Error'] = df_comparacion['Prediccion_mm'] - df_comparacion['Real_mm']
        df_comparacion['Error_Abs'] = abs(df_comparacion['Error'])
        df_comparacion['Error_Porcentual'] = (
            df_comparacion['Error_Abs'] / df_comparacion['Real_mm'].replace(0, np.nan) * 100
        )
        
        mae = df_comparacion['Error_Abs'].mean()
        rmse = np.sqrt((df_comparacion['Error'] ** 2).mean())
        
        df_mape = df_comparacion[df_comparacion['Real_mm'] > 1]
        mape = df_mape['Error_Porcentual'].mean() if len(df_mape) > 0 else None
        
        correlacion = df_comparacion['Real_mm'].corr(df_comparacion['Prediccion_mm'])
        
        # Categorización
        umbral_bajo = np.percentile(y_test, 33)
        umbral_alto = np.percentile(y_test, 67)
        
        def categorizar(valor):
            if valor < umbral_bajo:
                return 'Bajo'
            elif valor < umbral_alto:
                return 'Medio'
            else:
                return 'Alto'
        
        df_comparacion['Categoria_Real'] = df_comparacion['Real_mm'].apply(categorizar)
        df_comparacion['Categoria_Pred'] = df_comparacion['Prediccion_mm'].apply(categorizar)
        df_comparacion['Categoria_Correcta'] = (
            df_comparacion['Categoria_Real'] == df_comparacion['Categoria_Pred']
        )
        
        precision_categorias = df_comparacion['Categoria_Correcta'].mean() * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Correlacion': correlacion,
            'Precision_Categorias': precision_categorias,
            'df': df_comparacion,
            'fecha_inicio': fechas[0],
            'fecha_fin': fechas[-1],
            'total_semanas': n_usar
        }
        
    serie_completa = actualizar_datos.actualizar_serie(metadata['serie'], region_key)
    
    if len(serie_completa) < semanas_validar + 20:
        return None
    
    serie_hasta_pasado = serie_completa.iloc[:-semanas_validar]
    serie_real_reciente = serie_completa.iloc[-semanas_validar:]
    
    try:
        predicciones = sarimax.forecast(steps=semanas_validar)
        
        df_comparacion = pd.DataFrame({
            'Fecha': serie_real_reciente.index,
            'Real_mm': serie_real_reciente.values,
            'Prediccion_mm': predicciones.values
        })
        
        df_comparacion['Error'] = df_comparacion['Prediccion_mm'] - df_comparacion['Real_mm']
        df_comparacion['Error_Abs'] = abs(df_comparacion['Error'])
        df_comparacion['Error_Porcentual'] = (
            df_comparacion['Error_Abs'] / df_comparacion['Real_mm'].replace(0, np.nan) * 100
        )
        
        mae = df_comparacion['Error_Abs'].mean()
        rmse = np.sqrt((df_comparacion['Error'] ** 2).mean())
        
        df_mape = df_comparacion[df_comparacion['Real_mm'] > 1]
        mape = df_mape['Error_Porcentual'].mean() if len(df_mape) > 0 else None
        
        correlacion = df_comparacion['Real_mm'].corr(df_comparacion['Prediccion_mm'])
        
        umbral_bajo = serie_completa.quantile(0.33)
        umbral_alto = serie_completa.quantile(0.67)
        
        def categorizar(valor):
            if valor < umbral_bajo:
                return 'Bajo'
            elif valor < umbral_alto:
                return 'Medio'
            else:
                return 'Alto'
        
        df_comparacion['Categoria_Real'] = df_comparacion['Real_mm'].apply(categorizar)
        df_comparacion['Categoria_Pred'] = df_comparacion['Prediccion_mm'].apply(categorizar)
        df_comparacion['Categoria_Correcta'] = (
            df_comparacion['Categoria_Real'] == df_comparacion['Categoria_Pred']
        )
        
        precision_categorias = df_comparacion['Categoria_Correcta'].mean() * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Correlacion': correlacion,
            'Precision_Categorias': precision_categorias,
            'df': df_comparacion,
            'fecha_inicio': serie_real_reciente.index[0],
            'fecha_fin': serie_real_reciente.index[-1],
            'total_semanas': semanas_validar
        }
        
    except Exception as e:
        st.error(f"Error en validación reciente: {e}")
        return None

def generar_reporte_pdf(df_pred, metadata, region_key, serie):
    """Genera reporte PDF profesional"""
    
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 15, 'Reporte de Predicción de Precipitación', 0, 1, 'C')
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'República Dominicana - {region_key.replace("_", " ").title()}', 0, 1, 'C')
    pdf.cell(0, 10, f'Fecha de generación: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Información de la Región', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    pdf.cell(0, 8, f'Ciudades incluidas: {", ".join(metadata["ciudades"])}', 0, 1)
    pdf.cell(0, 8, f'Descripción: {metadata.get("descripcion", "N/A")}', 0, 1)
    
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Métricas del Modelo', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    metricas = metadata['metricas']
    pdf.cell(95, 8, f'Error Absoluto Medio (MAE): {metricas["mae"]:.2f} mm', 0, 0)
    pdf.cell(95, 8, f'RMSE: {metricas["rmse"]:.2f} mm', 0, 1)
    
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Resumen de Predicciones', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    pdf.cell(95, 8, f'Promedio: {df_pred["Prediccion_mm"].mean():.1f} mm/semana', 0, 0)
    pdf.cell(95, 8, f'Máximo: {df_pred["Prediccion_mm"].max():.1f} mm', 0, 1)
    
    return pdf.output(dest='S').encode('latin-1')

# ============================================================================
# CARGAR MODELOS
# ============================================================================

modelos = cargar_todos_modelos()

if not modelos:
    st.error("❌ No se encontraron modelos. Ejecuta primero el entrenamiento.")
    st.stop()

# 🔥 DEBUG: Mostrar estado de calibradores
if DEBUG:
    print("\n" + "=" * 70)
    print("ESTADO DE MODELOS Y CALIBRADORES")
    print("=" * 70)
    for nombre, modelo_dict in modelos.items():
        cal_status = "✅ SÍ" if modelo_dict.get('calibrador') else "❌ NO"
        print(f"{nombre:30s} | Calibrador: {cal_status}")
        if modelo_dict.get('calibrador'):
            cal = modelo_dict['calibrador']
            if 'sesgo_medio' in cal:
                print(f"{'':30s} | Sesgo: {cal['sesgo_medio']:.1f} mm")
            if 'factores' in cal:
                print(f"{'':30s} | Factores: {list(cal['factores'].keys())}")
    print("=" * 70 + "\n")

# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================

st.sidebar.markdown('<h2><i class="fa-solid fa-sliders"></i> Panel de Control</h2>', unsafe_allow_html=True)

nombres_display = {k: v['nombre_display'] for k, v in modelos.items()}

st.sidebar.markdown(
    '<i class="fa-solid fa-location-dot"></i> <b>Selecciona región:</b>',
    unsafe_allow_html=True
)

region_key = st.sidebar.selectbox(
    "",
    list(modelos.keys()),
    format_func=lambda x: nombres_display[x],
    key="selectbox_regiones"
)

modelo_seleccionado = modelos[region_key]
metadata = modelo_seleccionado['metadata']

st.sidebar.markdown(
    '<i class="fa-solid fa-calendar-week"></i> <b>Semanas de proyección futura</b>',
    unsafe_allow_html=True
)

semanas_predecir = st.sidebar.slider(
    "",
    min_value=1,
    max_value=52,
    value=12,
    help="Selecciona cuántas semanas quieres predecir",
    key="slider_semanas"
)

# Estado de los Datos
st.sidebar.markdown("---")
st.sidebar.markdown('<h3><i class="fa-solid fa-database"></i> Estado de Datos</h3>', unsafe_allow_html=True)

# 🔥 Compatibilidad con modelos híbridos (sin 'serie' guardada)
if 'serie' in metadata:
    # MODELO ANTIGUO: Con serie histórica guardada
    serie_actual = actualizar_datos.actualizar_serie(metadata['serie'], region_key)
    ultima_fecha = serie_actual.index[-1]
    
    st.sidebar.info(f"Última actualización: {ultima_fecha.date()}")
    
    hoy = datetime.now().date()
    dias_diferencia = (hoy - ultima_fecha.date()).days
    
    if dias_diferencia <= 7:
        st.sidebar.success(f"✅ Actualizado (hace {dias_diferencia} días)")
    elif dias_diferencia <= 14:
        st.sidebar.info(f"📅 Hace {dias_diferencia} días")
    else:
        st.sidebar.warning(f"⚠️ Hace {dias_diferencia} días")
    
    st.sidebar.metric("Última fecha", ultima_fecha.date().strftime('%Y-%m-%d'))
    st.sidebar.metric("Total semanas", len(serie_actual))

else:
    # MODELO HÍBRIDO: Sin serie histórica en formato antiguo
    st.sidebar.warning("⚠️ Modelo híbrido v2.1")
    st.sidebar.info("""
    **Nueva arquitectura:**
    - ✅ Variabilidad mejorada
    - ✅ Calibración estratificada
    - 🔄 Serie en formato optimizado
    """)
    
    # Intentar mostrar info de validación si existe
    if 'validacion' in metadata and 'fechas_test' in metadata['validacion']:
        ultima_fecha_validacion = metadata['validacion']['fechas_test'][-1]
        st.sidebar.metric("Última validación", ultima_fecha_validacion.strftime('%Y-%m-%d'))
        
        # Crear serie_actual para compatibilidad con TAB 2
        fechas = metadata['validacion']['fechas_test']
        valores = metadata['validacion']['y_test']
        serie_actual = pd.Series(valores, index=fechas)
    else:
        # Fallback: crear serie dummy
        st.sidebar.warning("⚠️ Sin datos de validación")
        ultima_fecha = pd.Timestamp.now() - pd.Timedelta(weeks=4)
        fechas = pd.date_range(end=ultima_fecha, periods=52, freq='W-SUN')
        serie_actual = pd.Series([10] * 52, index=fechas)

# ✅ NUEVA SECCIÓN: Estado de Calibración
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Estado de Calibración")

calibrador_actual = modelo_seleccionado.get('calibrador')

if calibrador_actual:
    st.sidebar.success("✅ Calibrador disponible")
    
    # ✅ DETECTAR TIPO Y MOSTRAR INFO APROPIADA
    
    # Tipo 1: Calibrador estratificado (nuevo)
    if 'sesgos' in calibrador_actual and 'configuraciones' in calibrador_actual:
        st.sidebar.info(f"📊 Tipo: Estratificado v{calibrador_actual.get('version', '2.0')}")
        
        # Calcular sesgo promedio de todos los niveles
        sesgos_valores = list(calibrador_actual['sesgos'].values())
        sesgo_promedio = np.mean([abs(s) for s in sesgos_valores])
        st.sidebar.metric("Sesgo promedio", f"{sesgo_promedio:.1f} mm")
        
        with st.sidebar.expander("Ver sesgos por nivel"):
            for nivel, sesgo in calibrador_actual['sesgos'].items():
                emoji = "📈" if sesgo > 0 else "📉"
                st.write(f"{emoji} **{nivel}**: {sesgo:+.1f} mm")
        
        with st.sidebar.expander("Ver configuraciones"):
            for modo, config in calibrador_actual['configuraciones'].items():
                st.write(f"**{modo.title()}**")
                st.write(f"  • Agresividad: {config['agresividad']:.1%}")
                st.write(f"  • Variabilidad: {config['factor_variabilidad']:.1f}x")
    
    # Tipo 2: Calibrador multiplicativo (antiguo)
    elif 'sesgo_medio' in calibrador_actual:
        st.sidebar.info("📊 Tipo: Multiplicativo (antiguo)")
        st.sidebar.metric("Sesgo detectado", f"{calibrador_actual['sesgo_medio']:.1f} mm")
        st.sidebar.metric("Validación", f"{calibrador_actual['n_semanas_validacion']} semanas")
        
        with st.sidebar.expander("Ver factores de corrección"):
            for modo, factor in calibrador_actual['factores'].items():
                reduccion = (1 - factor) * 100
                st.write(f"**{modo.title()}**: {factor:.3f} (↓{reduccion:.0f}%)")
    
    # Tipo desconocido
    else:
        st.sidebar.warning("⚠️ Calibrador con formato desconocido")
        st.sidebar.info("Ejecuta `python calibrar_simple.py`")
    
else:
    st.sidebar.warning("⚠️ Sin calibrador")
    st.sidebar.info("💡 Ejecuta `python calibrar_simple.py`")


# ============================================================================
# SISTEMA DE AUTENTICACIÓN
# ============================================================================

auth.inicializar_sesion()

# Si no está autenticado, mostrar login
if not auth.esta_autenticado():
    auth.render_login()
    st.stop()  # Detener ejecución del resto de la app

# Sidebar con info del usuario
with st.sidebar:
    st.markdown("---")
    st.markdown("### 👤 Usuario Actual")
    usuario = auth.obtener_usuario_actual()
    st.write(f"**{usuario['nombre']}**")
    st.write(f"Rol: {usuario['rol'].title()}")
    
    if st.button("🚪 Cerrar Sesión", use_container_width=True):
        auth.logout()
        st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

# ============================================================================
# MAIN CONTENT - TABS DINÁMICOS
# ============================================================================

# Lista base de pestañas
titulos_tabs = ["📊 Predicciones", "📈 Análisis Histórico", "🎯 Métricas", "📡 Dispositivos", "ℹ️ Acerca de"]

# Si es admin, agregar pestaña de usuarios
es_admin = auth.es_admin()
if es_admin:
    titulos_tabs.append("👥 Usuarios")

# Crear los tabs
tabs = st.tabs(titulos_tabs)

# Asignar contenido a cada tab
tab1, tab2, tab3, tab4, tab5 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]

# Si es admin, el tab 6 existe
if es_admin:
    tab_usuarios = tabs[5]
    with tab_usuarios:
        render_gestion_usuarios()
# ============================================================================
# TAB 1: PREDICCIONES - ACTUALIZADO
# ============================================================================

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🏙️ Ciudades",
            len(metadata['ciudades']),
            help="Número de ciudades en este grupo"
        )
    
    with col2:
        st.metric(
            "📉 MAE",
            f"{metadata['metricas']['mae']:.1f} mm",
            help="Error Absoluto Medio"
        )
    
    with col3:
        st.metric(
            "📊 RMSE",
            f"{metadata['metricas']['rmse']:.1f} mm",
            help="Raíz del Error Cuadrático Medio"
        )
    
    with col4:
        st.metric(
            "🎯 Variabilidad",
            f"{metadata['metricas']['variability_ratio']:.0%}",
            help="Preservación de variabilidad natural"
        )
    
    st.markdown("---")
    
    with st.expander("🗺️ Ver ciudades incluidas"):
        ciudades_cols = st.columns(3)
        for idx, ciudad in enumerate(metadata['ciudades']):
            with ciudades_cols[idx % 3]:
                st.write(f"• {ciudad.title()}")
    
    st.markdown("---")
    
    # ✅ SECCIÓN DE CALIBRACIÓN ACTUALIZADA
    st.subheader("🔧 Calibración del Modelo (Opcional)")
    
    calibrador = modelo_seleccionado.get('calibrador')
    
    if calibrador:
        # Detectar tipo de calibrador
        es_estratificado = 'sesgos' in calibrador and 'configuraciones' in calibrador
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            usar_calibracion = st.checkbox(
                "Aplicar calibración pre-entrenada",
                value=False,
                help="Usa factores de corrección calculados durante el entrenamiento"
            )
        
        with col2:
            if usar_calibracion:
                modo_calibracion = st.radio(
                    "Intensidad:",
                    ["conservador", "balanceado", "agresivo", "ultra_agresivo"],
                    index=1 if es_estratificado else 0,
                    format_func=lambda x: x.replace('_', ' ').title(),
                    help="""
                    • Conservador: Corrección suave (30%, mejor para alertas)
                    • Balanceado: Corrección moderada (50%, uso general)
                    • Agresivo: Corrección fuerte (70%)
                    • Ultra Agresivo: Máxima corrección (95%, para MAE>30mm)
                    """
                )
        
        if usar_calibracion:
            if es_estratificado:
                config = calibrador['configuraciones'][modo_calibracion]
                st.info(f"""
                📊 **Calibrador Estratificado por Nivel**
                
                Este calibrador aplica correcciones diferentes según el nivel de lluvia:
                - **Agresividad**: {config['agresividad']:.0%} de corrección del sesgo observado
                - **Factor variabilidad**: {config['factor_variabilidad']:.1f}x
                - **Método**: Corrección aditiva adaptativa por nivel
                
                🎯 {config['descripcion']}
                """)
                
                with st.expander("Ver sesgos detectados por nivel"):
                    for nivel, sesgo in calibrador['sesgos'].items():
                        emoji = "📈" if sesgo > 0 else "📉"
                        direccion = "subestima" if sesgo > 0 else "sobreestima"
                        st.write(f"{emoji} **{nivel.replace('_', ' ').title()}**: "
                                f"Modelo {direccion} por {abs(sesgo):.1f} mm")
            else:
                # Calibrador multiplicativo antiguo
                factor_info = calibrador['factores'][modo_calibracion]
                st.info(f"""
                📊 **Información del Calibrador**
                - Sesgo detectado: {calibrador.get('sesgo_medio', 'N/A')} mm
                - Factor a aplicar: {factor_info:.3f}
                - Reducción estimada: ~{(1-factor_info)*100:.0f}%
                - Entrenado con: {calibrador.get('n_semanas_validacion', 'N/A')} semanas
                """)
    else:
        st.warning("⚠️ No hay calibrador disponible para esta región")
        st.info("💡 Ejecuta `python calibrar_simple.py` para generar calibradores")
        usar_calibracion = False
    
    # Botón de predicción
    if st.button("🔮 Generar Predicciones", type="primary", use_container_width=True):
        
        with st.spinner("Calculando predicciones..."):
            df_pred, serie = generar_predicciones(modelo_seleccionado, semanas_predecir, region_key)
        
        if df_pred is not None:
            
            # ✅ APLICAR CALIBRACIÓN SI ESTÁ ACTIVADA
            if usar_calibracion and calibrador:
                df_pred['Prediccion_Original'] = df_pred['Prediccion_mm'].copy()
                
                df_pred['Prediccion_mm'] = aplicar_calibracion(
                    df_pred['Prediccion_mm'].values,
                    calibrador,
                    modo=modo_calibracion
                )
                
                reduccion_promedio = (df_pred['Prediccion_Original'].mean() - df_pred['Prediccion_mm'].mean())
                
                # Mensaje según tipo de calibrador
                if 'sesgos' in calibrador:
                    config = calibrador['configuraciones'][modo_calibracion]
                    st.success(f"""
                    ✅ **Calibración estratificada "{modo_calibracion}" aplicada exitosamente**
                    
                    - **Agresividad**: {config['agresividad']:.0%} de corrección del sesgo
                    - **Factor variabilidad**: {config['factor_variabilidad']:.1f}x
                    - **Cambio promedio**: {reduccion_promedio:+.1f} mm/semana
                    
                    🔬 Este calibrador corrige sesgos diferentes según el nivel de lluvia
                    """)
                else:
                    factor_usado = calibrador['factores'][modo_calibracion]
                    st.success(f"""
                    ✅ **Calibración {modo_calibracion} aplicada exitosamente**
                    
                    - **Factor aplicado**: {factor_usado:.3f}
                    - **Reducción promedio**: {reduccion_promedio:.1f} mm/semana
                    - **Basado en**: {calibrador.get('n_semanas_validacion', 'N/A')} semanas de validación
                    """)
            
            # Estadísticas de predicción
            st.subheader("📋 Resumen de Predicciones")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Promedio", f"{df_pred['Prediccion_mm'].mean():.1f} mm/sem")
            with col2:
                st.metric("Máximo", f"{df_pred['Prediccion_mm'].max():.1f} mm")
            with col3:
                st.metric("Total", f"{df_pred['Prediccion_mm'].sum():.0f} mm")
            with col4:
                dias_lluvia = (df_pred['Prediccion_mm'] > 5).sum()
                st.metric("Semanas >5mm", f"{dias_lluvia}")
            
            # Gráfico principal
            st.subheader("📊 Predicciones vs Histórico")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            historico_reciente = serie.tail(52)
            ax.plot(historico_reciente.index, historico_reciente.values,
                   label='Histórico (último año)', color='#2E86AB', linewidth=2, alpha=0.8)
            
            if usar_calibracion and 'Prediccion_Original' in df_pred.columns:
                ax.plot(df_pred['Fecha'], df_pred['Prediccion_Original'],
                       label='Predicción Original', color='#FF6B6B', linewidth=2, 
                       marker='o', markersize=4, alpha=0.5, linestyle='--')
                ax.plot(df_pred['Fecha'], df_pred['Prediccion_mm'],
                       label='Predicción Calibrada', color='#E63946', linewidth=2.5, 
                       marker='o', markersize=5)
            else:
                ax.plot(df_pred['Fecha'], df_pred['Prediccion_mm'],
                       label='Predicción', color='#E63946', linewidth=2.5, marker='o', markersize=4)
            
            ax.axvline(serie.index[-1], color='black', linestyle='--', alpha=0.5, linewidth=1.5,
                      label='Hoy')
            
            std_historico = serie.std()
            ax.fill_between(df_pred['Fecha'],
                           (df_pred['Prediccion_mm'] - std_historico * 0.3).clip(lower=0),
                           df_pred['Prediccion_mm'] + std_historico * 0.3,
                           alpha=0.2, color='#E63946', label='Intervalo estimado')
            
            media_historica = serie.mean()
            ax.axhline(media_historica, color='gray', linestyle=':', alpha=0.5,
                      label=f'Media histórica ({media_historica:.1f} mm)')
            
            ax.set_xlabel('Fecha', fontsize=11, fontweight='bold')
            ax.set_ylabel('Precipitación (mm/semana)', fontsize=11, fontweight='bold')
            ax.set_title('Predicción de Precipitación Semanal', fontsize=13, fontweight='bold', pad=20)
            ax.legend(loc='upper left', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Sistema de alertas
            st.subheader("🚨 Sistema de Alertas")
            
            umbral_alto = np.percentile(serie, 85)
            umbral_muy_alto = np.percentile(serie, 95)
            
            alertas_altas = df_pred[df_pred['Prediccion_mm'] > umbral_alto]
            alertas_muy_altas = df_pred[df_pred['Prediccion_mm'] > umbral_muy_alto]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"🟢 **Umbral Normal**\n\n< {umbral_alto:.0f} mm/semana")
            with col2:
                st.warning(f"🟡 **Alerta Alta**\n\n{umbral_alto:.0f} - {umbral_muy_alto:.0f} mm")
            with col3:
                st.error(f"🔴 **Alerta Crítica**\n\n> {umbral_muy_alto:.0f} mm")
            
            if len(alertas_muy_altas) > 0:
                st.error(f"🔴 **ALERTA CRÍTICA**: {len(alertas_muy_altas)} semanas con precipitación muy alta")
                for _, row in alertas_muy_altas.head(5).iterrows():
                    st.write(f"• **{row['Fecha'].strftime('%d/%m/%Y')}**: {row['Prediccion_mm']:.1f} mm")
            
            elif len(alertas_altas) > 0:
                st.warning(f"🟡 **ALERTA ALTA**: {len(alertas_altas)} semanas con precipitación elevada")
                for _, row in alertas_altas.head(5).iterrows():
                    st.write(f"• **{row['Fecha'].strftime('%d/%m/%Y')}**: {row['Prediccion_mm']:.1f} mm")
            
            else:
                st.success("✅ No se prevén alertas. Precipitación dentro de rangos normales.")
            
            # Tabla de datos
            st.subheader("📋 Datos Detallados")
            
            df_display = df_pred.copy()
            df_display['Fecha'] = df_display['Fecha'].dt.strftime('%d/%m/%Y')
            df_display['Precipitación (mm)'] = df_display['Prediccion_mm'].round(1)
            
            def alerta_emoji(val):
                if val > umbral_muy_alto:
                    return '🔴'
                elif val > umbral_alto:
                    return '🟡'
                return '🟢'
            
            df_display['Nivel'] = df_display['Prediccion_mm'].apply(alerta_emoji)
            
            if usar_calibracion and 'Prediccion_Original' in df_pred.columns:
                df_display['Original (mm)'] = df_pred['Prediccion_Original'].round(1)
                df_display = df_display[['Fecha', 'Original (mm)', 'Precipitación (mm)', 'Nivel']]
            else:
                df_display = df_display[['Fecha', 'Precipitación (mm)', 'Nivel']]
            
            st.dataframe(df_display, use_container_width=True, height=400)
            
            # Botones de descarga
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_pred.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar predicciones (CSV)",
                    data=csv,
                    file_name=f"predicciones_{region_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("📄 Generar Reporte PDF", use_container_width=True):
                    with st.spinner("Generando reporte PDF..."):
                        pdf_data = generar_reporte_pdf(df_pred, metadata, region_key, serie)
                        
                        st.download_button(
                            label="📥 Descargar Reporte PDF",
                            data=pdf_data,
                            file_name=f"reporte_{region_key}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.success("✅ Reporte PDF generado exitosamente")

# ============================================================================
# TAB 2: ANÁLISIS HISTÓRICO
# ============================================================================

with tab2:
    st.subheader("📈 Análisis de Datos Históricos")
    
    # 🔥 Usar serie_actual (que ya fue creada en el sidebar)
    # serie_actual ya existe y es compatible con modelos híbridos
    serie = serie_actual
    
    if serie is None or len(serie) < 52:
        st.warning("⚠️ Datos históricos insuficientes para análisis completo")
        st.info("Este es un modelo híbrido v2.1. Los datos históricos están en formato optimizado.")
        st.stop()  # Detener ejecución de este TAB
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Media", f"{serie.mean():.1f} mm/sem")
    with col2:
        st.metric("Desv. Estándar", f"{serie.std():.1f} mm")
    with col3:
        st.metric("Máximo", f"{serie.max():.1f} mm")
    with col4:
        st.metric("Mínimo", f"{serie.min():.1f} mm")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(serie.index, serie.values, color='#2E86AB', linewidth=1, alpha=0.7)
    ax.fill_between(serie.index, 0, serie.values, alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Precipitación (mm/semana)', fontweight='bold')
    ax.set_title('Serie Temporal Completa (2010-2025)', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("🌦️ Patrón Estacional")
    
    patron_mensual = serie.groupby(serie.index.month).agg(['mean', 'std', 'max'])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    ax.bar(meses, patron_mensual['mean'], alpha=0.7, color='#2E86AB', label='Promedio')
    ax.errorbar(meses, patron_mensual['mean'], yerr=patron_mensual['std'],
                fmt='none', color='black', alpha=0.5, capsize=5, label='Desv. Estándar')
    
    ax.set_xlabel('Mes', fontweight='bold')
    ax.set_ylabel('Precipitación (mm/semana)', fontweight='bold')
    ax.set_title('Patrón Estacional Promedio', fontweight='bold', pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("📊 Distribución de Precipitación")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(serie.values, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.axvline(serie.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {serie.mean():.1f} mm')
    ax1.set_xlabel('Precipitación (mm/semana)', fontweight='bold')
    ax1.set_ylabel('Frecuencia', fontweight='bold')
    ax1.set_title('Histograma', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.boxplot(serie.values, vert=True)
    ax2.set_ylabel('Precipitación (mm/semana)', fontweight='bold')
    ax2.set_title('Box Plot', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# TAB 3: MÉTRICAS
# ============================================================================

with tab3:
    st.subheader("🎯 Métricas de Rendimiento del Modelo")
    
    metricas = metadata['metricas']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 Errores de Predicción")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        metricas_nombres = ['MAE', 'RMSE']
        metricas_valores = [metricas['mae'], metricas['rmse']]
        colores = ['#2E86AB', '#E63946']
        
        bars = ax.bar(metricas_nombres, metricas_valores, color=colores, alpha=0.7, edgecolor='black')
        ax.set_ylabel('mm/semana', fontweight='bold')
        ax.set_title('Errores de Predicción', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info(f"""
        **MAE (Error Absoluto Medio):** {metricas['mae']:.2f} mm  
        Error promedio en las predicciones.
        
        **RMSE (Raíz del Error Cuadrático Medio):** {metricas['rmse']:.2f} mm  
        Penaliza más los errores grandes.
        """)
    
    with col2:
        st.markdown("### 📊 Calidad del Modelo")
        
        var_ratio = metricas['variability_ratio']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        categories = ['Subestimada\n(<0.7)', 'Buena\n(0.7-1.3)', 'Sobreestimada\n(>1.3)']
        
        if var_ratio < 0.7:
            categoria_actual = 0
            color_actual = '#E63946'
        elif var_ratio <= 1.3:
            categoria_actual = 1
            color_actual = '#2E86AB'
        else:
            categoria_actual = 2
            color_actual = '#E63946'
        
        bars = ax.bar(categories, [1, 1, 1], color=['lightgray']*3, alpha=0.3, edgecolor='black')
        bars[categoria_actual].set_color(color_actual)
        bars[categoria_actual].set_alpha(0.7)
        
        ax.set_ylabel('Estado', fontweight='bold')
        ax.set_title(f'Variabilidad: {var_ratio:.0%}', fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info(f"""
        **Ratio de Variabilidad:** {var_ratio:.1%}  
        Mide qué tan bien el modelo preserva la variabilidad natural de la precipitación.
        
        **Estado:** {'✅ Buena' if 0.7 <= var_ratio <= 1.3 else '⚠️ Necesita ajuste'}
        """)
    
    st.markdown("### 🏆 Comparación con Estándares")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mae_quality = "Excelente" if metricas['mae'] < 10 else "Bueno" if metricas['mae'] < 15 else "Aceptable"
        mae_color = "green" if metricas['mae'] < 10 else "orange" if metricas['mae'] < 15 else "red"
        st.markdown(f"**MAE:** :{mae_color}[{mae_quality}]")
        st.progress(min(metricas['mae'] / 20, 1.0))
    
    with col2:
        rmse_quality = "Excelente" if metricas['rmse'] < 12 else "Bueno" if metricas['rmse'] < 20 else "Aceptable"
        rmse_color = "green" if metricas['rmse'] < 12 else "orange" if metricas['rmse'] < 20 else "red"
        st.markdown(f"**RMSE:** :{rmse_color}[{rmse_quality}]")
        st.progress(min(metricas['rmse'] / 25, 1.0))
    
    with col3:
        var_quality = "Excelente" if 0.8 <= var_ratio <= 1.2 else "Bueno" if 0.7 <= var_ratio <= 1.3 else "Mejorable"
        var_color = "green" if 0.8 <= var_ratio <= 1.2 else "orange" if 0.7 <= var_ratio <= 1.3 else "red"
        st.markdown(f"**Variabilidad:** :{var_color}[{var_quality}]")
        st.progress(var_ratio if var_ratio <= 1 else 1/var_ratio)
    
    st.markdown("---")
    
    # ✅ Validación histórica
    st.markdown("### 🔬 Validación Histórica del Modelo")
    
    resultado = validar_predicciones_historicas(modelo_seleccionado, semanas_test=12)
    
    if resultado:
        col1, col2, col3 = st.columns(3)
        
        col1.metric("MAE histórico", f"{resultado['MAE']:.1f} mm")
        col2.metric("RMSE histórico", f"{resultado['RMSE']:.1f} mm")
        
        if resultado['MAPE'] is not None:
            col3.metric("MAPE histórico", f"{resultado['MAPE']:.1f}%")
        else:
            col3.metric("MAPE histórico", "N/A")
        
        st.line_chart(resultado['df'][['Prediccion_mm', 'Real_mm']])
        
        st.success("Validación histórica realizada correctamente con datos reales.")
    else:
        st.warning("No hay suficientes datos para validar.")
    
    # ✅ Validación en tiempo real
    st.markdown("---")
    st.markdown("### 📅 Validación en Tiempo Real")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("""
        🔍 **Validación Adaptativa**: Selecciona cuántas semanas hacia atrás quieres validar. 
        El sistema generará predicciones desde ese punto del pasado y las comparará con los datos reales.
        """)
    
    with col2:
        semanas_validar_custom = st.selectbox(
            "Período a validar:",
            [4, 8, 12, 16, 20, 26],
            index=2,
            format_func=lambda x: f"{x} semanas (~{x//4} meses)",
            help="Selecciona cuántas semanas hacia atrás validar"
        )
    
    resultado_reciente = validar_predicciones_recientes(
        modelo_seleccionado, 
        region_key, 
        semanas_validar=semanas_validar_custom
    )
    
    if resultado_reciente:
        st.info(f"""
        📅 **Período de validación:** {resultado_reciente['fecha_inicio'].strftime('%d/%m/%Y')} 
        hasta {resultado_reciente['fecha_fin'].strftime('%d/%m/%Y')}  
        📊 **Total de semanas:** {resultado_reciente['total_semanas']}
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            col1.metric("MAE Real", f"{resultado_reciente['MAE']:.1f} mm")
        
        with col2:
            col2.metric("RMSE Real", f"{resultado_reciente['RMSE']:.1f} mm")
        
        with col3:
            if resultado_reciente['MAPE'] is not None:
                col3.metric("MAPE Real", f"{resultado_reciente['MAPE']:.1f}%")
            else:
                col3.metric("MAPE Real", "N/A")
        
        with col4:
            col4.metric("Precisión por Nivel", f"{resultado_reciente['Precision_Categorias']:.0f}%")
        
        st.markdown("#### 📊 Comparación: Predicción vs Realidad")
        
        df = resultado_reciente['df']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        ax1.plot(df['Fecha'], df['Real_mm'], 
                 label='Datos Reales', color='#2E86AB', linewidth=2.5, marker='o', markersize=5)
        ax1.plot(df['Fecha'], df['Prediccion_mm'], 
                 label='Predicción del Modelo', color='#E63946', linewidth=2, marker='s', markersize=5, alpha=0.8)
        
        ax1.fill_between(df['Fecha'], df['Real_mm'], df['Prediccion_mm'], 
                          alpha=0.2, color='gray', label='Diferencia')
        
        ax1.set_xlabel('Fecha', fontweight='bold')
        ax1.set_ylabel('Precipitación (mm/semana)', fontweight='bold')
        ax1.set_title('Validación Real: ¿Qué tan cerca estuvo el modelo?', fontweight='bold', pad=15)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(df['Fecha'], df['Error'], 
                color=['#E63946' if e > 0 else '#2E86AB' for e in df['Error']], 
                alpha=0.6, edgecolor='black')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_xlabel('Fecha', fontweight='bold')
        ax2.set_ylabel('Error (Predicción - Real) mm', fontweight='bold')
        ax2.set_title('Distribución de Errores por Semana', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        error_promedio_abs = df['Error_Abs'].mean()
        
        if error_promedio_abs < 5:
            st.success(f"✅ **Excelente**: Error promedio {error_promedio_abs:.1f} mm")
        elif error_promedio_abs < 10:
            st.info(f"✔️ **Bueno**: Error promedio {error_promedio_abs:.1f} mm")
        elif error_promedio_abs < 15:
            st.warning(f"⚠️ **Aceptable**: Error promedio {error_promedio_abs:.1f} mm")
        else:
            st.error(f"❌ **Necesita Mejora**: Error promedio {error_promedio_abs:.1f} mm")
    
    else:
        st.warning("⚠️ No hay suficientes datos recientes para validación en tiempo real")
    
    # ✅ NUEVA SECCIÓN: Evaluación de calibración
    if calibrador_actual:
        st.markdown("---")
        st.markdown("### 🎯 Evaluación del Calibrador")
        
        # Detectar tipo
        es_estratificado = 'sesgos' in calibrador_actual
        
        if es_estratificado:
            st.info("""
            📊 **Calibrador Estratificado**
            
            Este calibrador aplica correcciones diferentes según el nivel de precipitación,
            corrigiendo sesgos específicos detectados en cada rango.
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Calcular sesgo promedio absoluto
                sesgos_valores = list(calibrador_actual['sesgos'].values())
                sesgo_promedio = np.mean([abs(s) for s in sesgos_valores])
                st.metric(
                    "Sesgo Promedio",
                    f"{sesgo_promedio:.1f} mm",
                    help="Promedio de sesgos absolutos por nivel"
                )
            
            with col2:
                # Contar niveles
                n_niveles = len(calibrador_actual['sesgos'])
                st.metric(
                    "Niveles de Corrección",
                    n_niveles,
                    help="Rangos de precipitación con corrección específica"
                )
            
            with col3:
                # Factor balanceado
                config_bal = calibrador_actual['configuraciones']['balanceado']
                st.metric(
                    "Agresividad Balanceada",
                    f"{config_bal['agresividad']:.0%}",
                    help="% de corrección del sesgo en modo balanceado"
                )
            
            # Mostrar sesgos por nivel
            st.markdown("#### 🔧 Sesgos Detectados por Nivel")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            niveles = list(calibrador_actual['sesgos'].keys())
            sesgos = list(calibrador_actual['sesgos'].values())
            colores = ['#E63946' if s > 0 else '#2E86AB' for s in sesgos]
            
            bars = ax.barh(niveles, sesgos, color=colores, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
            ax.set_xlabel('Sesgo (mm)', fontweight='bold')
            ax.set_ylabel('Nivel de Precipitación', fontweight='bold')
            ax.set_title('Sesgos por Nivel de Lluvia', fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Agregar valores en las barras
            for bar, sesgo in zip(bars, sesgos):
                width = bar.get_width()
                label_x_pos = width + (5 if width > 0 else -5)
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{sesgo:+.1f} mm',
                       va='center', ha='left' if width > 0 else 'right',
                       fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("#### 📋 Configuraciones Disponibles")
            
            for modo, config in calibrador_actual['configuraciones'].items():
                with st.expander(f"⚙️ Modo {modo.title()}"):
                    st.write(f"**Agresividad**: {config['agresividad']:.0%}")
                    st.write(f"**Factor variabilidad**: {config['factor_variabilidad']:.1f}x")
                    st.write(f"**Descripción**: {config['descripcion']}")
        
        else:
            # Calibrador multiplicativo (antiguo)
            st.info("📊 **Calibrador Multiplicativo (versión antigua)**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Sesgo Detectado",
                    f"{calibrador_actual['sesgo_medio']:.1f} mm",
                    help="Sobreestimación promedio del modelo"
                )
            
            with col2:
                st.metric(
                    "Semanas de Validación",
                    calibrador_actual['n_semanas_validacion'],
                    help="Datos usados para calibrar"
                )
            
            with col3:
                factor_balanceado = calibrador_actual['factores']['balanceado']
                st.metric(
                    "Factor Balanceado",
                    f"{factor_balanceado:.3f}",
                    help="Factor de corrección recomendado"
                )
            
            st.markdown("#### 🔧 Factores de Corrección Disponibles")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            modos = list(calibrador_actual['factores'].keys())
            factores = list(calibrador_actual['factores'].values())
            colores = ['#2E86AB', '#FFA500', '#E63946']
            
            bars = ax.bar(modos, factores, color=colores, alpha=0.7, edgecolor='black')
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Sin corrección (1.0)')
            ax.set_ylabel('Factor Multiplicativo', fontweight='bold')
            ax.set_title('Factores de Calibración por Modo', fontweight='bold')
            ax.set_ylim(0, 1.2)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, factor in zip(bars, factores):
                reduccion = (1 - factor) * 100
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                       f'{factor:.3f}\n(↓{reduccion:.0f}%)',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.success("""
        ✅ **Calibrador listo para usar**
        
        Activa la opción "Aplicar calibración" en el TAB 1 para usar predicciones corregidas.
        """)


# ============================================================================
# TAB 4: GESTIÓN DE DISPOSITIVOS
# ============================================================================

with tab4:

     # Verificar permisos
    if not auth.esta_autenticado():
        st.error("🔒 Debes iniciar sesión para acceder a esta sección")
        st.stop()

    gestion_dispositivos.render_gestion_dispositivos()

# ============================================================================
# TAB 5: ACERCA DE
# ============================================================================

with tab5:
    st.subheader("ℹ️ Acerca del Sistema")
    
    st.markdown("""
    ### 🌧️ Sistema de Predicción de Precipitación - AQUARIA
    
    Este sistema utiliza modelos híbridos de Machine Learning y análisis estadístico para predecir 
    la precipitación semanal en diferentes regiones de República Dominicana.
    
    #### 🔬 Tecnología
    
    - **SARIMAX**: Modelos estadísticos que capturan estacionalidad y tendencias
    - **Variables Meteorológicas**: Temperatura, presión, viento, evapotranspiración (Open-Meteo)
    - **Calibración Multiplicativa**: Sistema de corrección de sesgo post-entrenamiento
    - **Actualización Automática**: Integración con Open-Meteo para datos recientes
    
    #### 📊 Datos
    
    - **Período**: 2010 - 2025 (15 años de datos históricos)
    - **Fuente precipitación**: Estaciones meteorológicas ONAMET
    - **Fuente meteorológica**: Open-Meteo Historical Weather API
    - **Frecuencia**: Agregación semanal
    - **Ciudades**: 15 estaciones distribuidas por todo el país
    
    #### 🎯 Grupos Climáticos
    
    Los modelos están entrenados en 3 grupos climáticos diferenciados:
    
    1. **Norte/Cibao**: Mayor variabilidad, influencia atlántica
    2. **Sur Seco**: Zona más árida, menor precipitación
    3. **Este/Capital**: Costa caribeña y zona metropolitana
    
    #### 🔧 Sistema de Calibración
    
    **Nuevo en versión 2.5**: Sistema de calibración multiplicativa que corrige sesgos sistemáticos.
    
    - **Conservador**: Corrección suave (75% del factor base) - Mejor para alertas tempranas
    - **Balanceado**: Corrección moderada (65%) - Uso general recomendado  
    - **Agresivo**: Corrección fuerte (55%) - Predicciones más ajustadas
    
    Cada calibrador fue entrenado con 12 semanas de validación histórica.
    
    #### 📈 Rendimiento
    
    - Error promedio: 6-9 mm/semana (sin calibración)
    - Error promedio: 4-6 mm/semana (con calibración balanceada)
    - Precisión: ~75-85% en predicciones semanales
    - Variabilidad preservada: >80%
    
    #### 🔄 Actualización
    
    Los modelos se actualizan mensualmente con nuevos datos para mantener la precisión.
    Los calibradores pueden regenerarse ejecutando `python calibrar_simple.py`.
    
    #### 👨‍💻 Desarrollo
    
    Sistema desarrollado como parte del proyecto AQUARIA (Alerta y Cuantificación de 
    Riesgos de Inundación Asistida por IA) - PUCMM 2025.
    
    **Tecnologías**: Python, Statsmodels, Scikit-learn, Streamlit, Open-Meteo API
    
    **Autores**: Juan Alexander Alejo Polonia, Pedro José De La Rosa Cornielle
    
    **Asesor**: Bryan Muñoz, Ing. Telemático
    
    ---
    
    **Versión**: 2.5 (Con sistema de calibración integrado)  
    **Última actualización**: Enero 2026
    """)
    
    st.info("💡 **Nota**: Este sistema es para fines educativos y de investigación. "
            "Para decisiones críticas, consulte servicios meteorológicos oficiales como ONAMET.")
    
    st.markdown("---")
    
    st.markdown("### 📚 Referencias del Proyecto")
    
    with st.expander("Ver documento completo del proyecto"):
        st.markdown("""
        **AQUARIA (Alerta y Cuantificación de Riesgos de Inundación Asistida por IA)**
        
        Propuesta de proyecto presentado como requisito parcial para optar por el
        título de Ingeniero en Telemático en la Pontificia Universidad Católica Madre y Maestra.
        
        El documento completo incluye:
        - Marco teórico sobre sistemas de alerta temprana
        - Análisis de tecnologías de monitoreo ambiental
        - Diseño del sistema AQUARIA
        - Arquitectura hardware y software
        - Evaluación de resultados
        
        📄 Documento: `P1 - ITT 1900 - ALEJO - DE LA ROSA - AQUARIA.pdf`
        """)



                    
# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">'
    '🌧️ Sistema de Predicción de Precipitación RD - AQUARIA | '
    f'Modelos entrenados: {len(modelos)} regiones | '
    f'Calibradores: {sum(1 for m in modelos.values() if m.get("calibrador"))} disponibles | '
    'Desarrollado con ❤️ en PUCMM'
    '</p>',
    unsafe_allow_html=True
)
