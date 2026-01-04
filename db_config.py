"""
Módulo de conexión a Supabase para AQUARIA
"""
"""
Módulo de conexión a Supabase para AQUARIA
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import psycopg2
from psycopg2.extras import RealDictCursor

import streamlit as st

# Cargar variables de entorno
load_dotenv()

# ==================== DEBUG DE VARIABLES (BORRAR DESPUÉS) ====================
if not os.getenv("PGHOST"):
    print("❌ ALERTA: Las variables de entorno no se cargaron desde el archivo .env")
# ============================================================================

def get_db_secret(key):
    """Obtiene variables desde Streamlit Secrets o desde variables de entorno locales"""
    try:
        # Intenta primero con Streamlit Secrets (Nube)
        return st.secrets[key]
    except:
        # Si falla, busca en variables de entorno locales (PC)
        return os.getenv(key)
        
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL y SUPABASE_KEY deben estar en .env")
    return create_client(url, key)

supabase = get_supabase_client()

def get_pg_connection():
    try:
        conn = psycopg2.connect(
            host=get_db_secret("PGHOST"),
            port=get_db_secret("PGPORT"),
            database=get_db_secret("PGDATABASE"),
            user=get_db_secret("PGUSER"),
            password=get_db_secret("PGPASSWORD"),
            sslmode="require",
            # Añadimos esto para asegurar que no intente usar sentencias preparadas
            options="-c target_session_attrs=read-write" 
        )
        return conn
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None # Aquí devuelve None, por eso falla el cursor después
# ==================== FUNCIONES DE PRUEBA ====================

def test_connection():
    """Prueba conexión a Supabase"""
    try:
        # Test 1: Supabase client
        response = supabase.table('dispositivos').select('count').execute()
        print(f"✅ Supabase client OK")
        
        # Test 2: PostgreSQL directo
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM dispositivos")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        print(f"✅ PostgreSQL OK - {result['count']} dispositivos")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🔌 Probando conexión a Supabase...")
    test_connection()
