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
            host=st.secrets["PGHOST"],
            port=st.secrets["PGPORT"],
            database=st.secrets["PGDATABASE"],
            user=st.secrets["PGUSER"],
            password=st.secrets["PGPASSWORD"],
            sslmode="require",
            # Añadir esto para evitar que el pooler se confunda
            options="-c statement_timeout=30000" 
        )
        return conn
    except Exception as e:
        st.error(f"Error crítico: {e}")
        return None
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
