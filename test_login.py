#!/usr/bin/env python3
"""
Script de prueba para debugging del sistema de autenticación
"""

import hashlib
import sys
import os

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_config import get_pg_connection

def verificar_tabla():
    """Verifica si la tabla usuarios_auth existe buscando en el esquema público"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Consultamos directamente la existencia en el esquema 'public'
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'usuarios_auth'
            );
        """)
        
        existe = cursor.fetchone()['exists'] # Usamos nombre de columna por el RealDictCursor
        cursor.close()
        conn.close()
        return existe
    except Exception as e:
        print(f"❌ Error interno en el test: {e}")
        return False
    
def listar_usuarios():
    """Lista todos los usuarios"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id_usuario,
                email,
                nombre,
                rol,
                activo,
                password_hash
            FROM usuarios_auth
            ORDER BY id_usuario;
        """)
        
        usuarios = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return usuarios
    except Exception as e:
        print(f"❌ Error listando usuarios: {e}")
        return []

def probar_login(email, password):
    """Prueba login con email y password"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Generar hash
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        print(f"\n🔍 Intentando login:")
        print(f"   Email: {email}")
        print(f"   Password: {password}")
        print(f"   Hash generado: {password_hash}")
        
        # Buscar usuario
        cursor.execute("""
            SELECT 
                id_usuario,
                email,
                nombre,
                rol,
                activo,
                password_hash
            FROM usuarios_auth
            WHERE email = %s
        """, (email,))
        
        usuario = cursor.fetchone()
        
        if not usuario:
            print(f"\n❌ Usuario con email '{email}' no encontrado")
            cursor.close()
            conn.close()
            return False
        
        print(f"\n✅ Usuario encontrado:")
        print(f"   ID: {usuario['id_usuario']}")
        print(f"   Nombre: {usuario['nombre']}")
        print(f"   Rol: {usuario['rol']}")
        print(f"   Activo: {usuario['activo']}")
        print(f"   Hash almacenado: {usuario['password_hash']}")
        
        # Verificar password
        if usuario['password_hash'] == password_hash:
            print(f"\n✅ Password correcto")
            
            # Verificar activo
            if usuario['activo']:
                print(f"✅ Usuario activo")
                print(f"\n🎉 LOGIN EXITOSO")
                cursor.close()
                conn.close()
                return True
            else:
                print(f"❌ Usuario inactivo")
                cursor.close()
                conn.close()
                return False
        else:
            print(f"\n❌ Password incorrecto")
            print(f"   Hash esperado: {usuario['password_hash']}")
            print(f"   Hash recibido:  {password_hash}")
            cursor.close()
            conn.close()
            return False
        
    except Exception as e:
        print(f"❌ Error en login: {e}")
        return False

# ============================================================================
# MAIN
# ============================================================================

print("\n" + "="*70)
print("PRUEBA DE SISTEMA DE AUTENTICACIÓN")
print("="*70 + "\n")

# 1. Verificar tabla
print("1️⃣ Verificando tabla usuarios_auth...")
if verificar_tabla():
    print("   ✅ Tabla existe")
else:
    print("   ❌ Tabla NO existe")
    print("\n🚨 SOLUCIÓN: Ejecuta setup_auth_corregido.sql en Supabase")
    sys.exit(1)

# 2. Listar usuarios
print("\n2️⃣ Listando usuarios...")
usuarios = listar_usuarios()

if not usuarios:
    print("   ❌ No hay usuarios en la tabla")
    print("\n🚨 SOLUCIÓN: Ejecuta setup_auth_corregido.sql en Supabase")
    sys.exit(1)

print(f"   ✅ {len(usuarios)} usuario(s) encontrado(s):\n")

for u in usuarios:
    print(f"   • {u['email']} ({u['rol']}) - Activo: {u['activo']}")

# 3. Probar login con admin
print("\n3️⃣ Probando login con admin@aquaria.do...")
probar_login('admin@aquaria.do', 'Admin123!')

print("\n" + "="*70 + "\n")