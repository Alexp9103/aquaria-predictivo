#!/usr/bin/env python3
"""
Sistema de alertas por Telegram
Gestiona suscripciones y envío de alertas
"""

import sqlite3
import pickle
import pandas as pd
from datetime import datetime
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE TELEGRAM BOT
# ============================================================================

TELEGRAM_BOT_TOKEN = "8515184062:AAHR1X6wboMelFCPEJzOlgAHRsX6h5TUnS0"  # Obtener de @BotFather
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ============================================================================
# BASE DE DATOS
# ============================================================================

def init_database():
    """Crea la base de datos de suscripciones"""
    conn = sqlite3.connect('suscripciones.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id TEXT UNIQUE NOT NULL,
            nombre TEXT,
            region TEXT NOT NULL,
            nivel_alerta TEXT DEFAULT 'alta',
            activo INTEGER DEFAULT 1,
            fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alertas_enviadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            region TEXT,
            nivel TEXT,
            precipitacion REAL,
            fecha_alerta TIMESTAMP,
            fecha_envio TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Base de datos inicializada")

def registrar_usuario(telegram_id, nombre, region, nivel_alerta='alta'):
    """Registra un nuevo usuario"""
    conn = sqlite3.connect('suscripciones.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO usuarios (telegram_id, nombre, region, nivel_alerta)
            VALUES (?, ?, ?, ?)
        ''', (telegram_id, nombre, region, nivel_alerta))
        
        conn.commit()
        logger.info(f"✅ Usuario {nombre} registrado para {region}")
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"Usuario {telegram_id} ya existe")
        return False
    finally:
        conn.close()

def obtener_suscriptores(region=None, nivel_alerta=None):
    """Obtiene lista de suscriptores activos"""
    conn = sqlite3.connect('suscripciones.db')
    cursor = conn.cursor()
    
    query = "SELECT telegram_id, nombre, region, nivel_alerta FROM usuarios WHERE activo = 1"
    params = []
    
    if region:
        query += " AND region = ?"
        params.append(region)
    
    if nivel_alerta:
        query += " AND nivel_alerta = ?"
        params.append(nivel_alerta)
    
    cursor.execute(query, params)
    usuarios = cursor.fetchall()
    conn.close()
    
    return usuarios

def registrar_alerta_enviada(usuario_id, region, nivel, precipitacion, fecha_alerta):
    """Registra que se envió una alerta"""
    conn = sqlite3.connect('suscripciones.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO alertas_enviadas (usuario_id, region, nivel, precipitacion, fecha_alerta)
        VALUES (?, ?, ?, ?, ?)
    ''', (usuario_id, region, nivel, precipitacion, fecha_alerta))
    
    conn.commit()
    conn.close()

# ============================================================================
# FUNCIONES DE TELEGRAM
# ============================================================================

def enviar_mensaje_telegram(chat_id, mensaje):
    """Envía mensaje por Telegram"""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": mensaje,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Error enviando mensaje a {chat_id}: {e}")
        return False

# ============================================================================
# GENERACIÓN Y ENVÍO DE ALERTAS
# ============================================================================

def generar_predicciones_para_alertas(region_key):
    """Genera predicciones para las próximas 2 semanas"""
    try:
        # Cargar modelo
        with open(f'modelos/metadata_{region_key}.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        with open(f'modelos/sarimax_{region_key}.pkl', 'rb') as f:
            sarimax = pickle.load(f)
        
        # Predicción 2 semanas
        pred = sarimax.forecast(steps=2)
        
        return pred.values
        
    except Exception as e:
        logger.error(f"Error generando predicción para {region_key}: {e}")
        return None

def determinar_nivel_alerta(precipitacion_mm, serie_historica):
    """Determina nivel de alerta basado en percentiles"""
    
    # 🔥 TEMPORAL: Umbrales muy bajos para testing
    umbral_alto = 20  # Normalmente: pd.Series(serie_historica).quantile(0.85)
    umbral_critico = 50  # Normalmente: pd.Series(serie_historica).quantile(0.95)
    
    if precipitacion_mm >= umbral_critico:
        return 'critica', '🔴'
    elif precipitacion_mm >= umbral_alto:
        return 'alta', '🟡'
    else:
        return 'normal', '🟢'

def enviar_alertas():
    """Proceso principal de envío de alertas"""
    
    logger.info("\n" + "="*70)
    logger.info("🚨 GENERANDO Y ENVIANDO ALERTAS")
    logger.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    regiones = {
        'grupo_1_norte_cibao': 'Norte/Cibao',
        'grupo_2_sur_seco': 'Sur Seco',
        'grupo_3_este_capital': 'Este/Capital'
    }
    
    alertas_enviadas = 0
    
    for region_key, region_nombre in regiones.items():
        logger.info(f"\n📍 Procesando {region_nombre}...")
        
        # Generar predicciones
        predicciones = generar_predicciones_para_alertas(region_key)
        
        if predicciones is None:
            continue
        
        # Cargar serie para umbrales
        with open(f'modelos/metadata_{region_key}.pkl', 'rb') as f:
            metadata = pickle.load(f)
        serie = metadata['serie']
        
        # Analizar próximas 2 semanas
        for i, precip in enumerate(predicciones):
            fecha = datetime.now() + pd.Timedelta(weeks=i+1)
            nivel, emoji = determinar_nivel_alerta(precip, serie)
            
            logger.info(f"  Semana {i+1}: {precip:.1f} mm → {nivel.upper()} {emoji}")
            
            # Solo enviar si es alerta alta o crítica
            if nivel in ['alta', 'critica']:
                
                # Obtener suscriptores de esta región
                suscriptores = obtener_suscriptores(region=region_key)
                
                # Preparar mensaje
                if nivel == 'critica':
                    titulo = "🔴 ALERTA CRÍTICA DE PRECIPITACIÓN"
                else:
                    titulo = "🟡 ALERTA ALTA DE PRECIPITACIÓN"
                
                mensaje = f"""
{titulo}

📍 <b>Región:</b> {region_nombre}
📅 <b>Semana:</b> {fecha.strftime('%d/%m/%Y')}
🌧️ <b>Precipitación esperada:</b> {precip:.1f} mm

<b>Recomendaciones:</b>
{'• Evacuar zonas de riesgo' if nivel == 'critica' else '• Preparar medidas preventivas'}
• Evitar zonas bajas e inundables
• Mantenerse informado

<i>Sistema de Predicción - RD</i>
                """
                
                # Enviar a cada suscriptor
                for telegram_id, nombre, _, _ in suscriptores:
                    if enviar_mensaje_telegram(telegram_id, mensaje):
                        alertas_enviadas += 1
                        logger.info(f"  ✅ Alerta enviada a {nombre}")
    
    logger.info(f"\n📊 Total alertas enviadas: {alertas_enviadas}")
    logger.info("="*70 + "\n")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def listar_usuarios():
    """Lista todos los usuarios registrados"""
    conn = sqlite3.connect('suscripciones.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM usuarios')
    usuarios = cursor.fetchall()
    conn.close()
    
    return usuarios

def desactivar_usuario(telegram_id):
    """Desactiva un usuario"""
    conn = sqlite3.connect('suscripciones.db')
    cursor = conn.cursor()
    
    cursor.execute('UPDATE usuarios SET activo = 0 WHERE telegram_id = ?', (telegram_id,))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Inicializar BD
    init_database()
    
    # Enviar alertas
    enviar_alertas()