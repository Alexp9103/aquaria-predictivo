"""
Sistema de Autenticación para AQUARIA
Roles: Admin y Usuario
"""

import streamlit as st
from db_config import supabase, get_pg_connection
from datetime import datetime
import hashlib

# ==================== GESTIÓN DE SESIÓN ====================

def inicializar_sesion():
    """Inicializa variables de sesión si no existen"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'role' not in st.session_state:
        st.session_state.role = None

def esta_autenticado():
    """Verifica si el usuario está autenticado"""
    return st.session_state.get('authenticated', False)

def es_admin():
    """Verifica si el usuario es admin"""
    return st.session_state.get('role') == 'admin'

def obtener_usuario_actual():
    """Retorna el usuario actual de la sesión"""
    return st.session_state.get('user')

# ==================== AUTENTICACIÓN ====================

def login(email, password):
    """
    Autentica usuario con email y password
    Retorna (success, user_data, error_message)
    """
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Hash del password (simple para demo - usar bcrypt en producción)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Buscar usuario
        cursor.execute("""
            SELECT 
                id_usuario,
                email,
                nombre,
                rol,
                activo
            FROM usuarios_auth
            WHERE email = %s AND password_hash = %s AND activo = TRUE
        """, (email, password_hash))
        
        usuario = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if usuario:
            # Guardar en sesión
            st.session_state.authenticated = True
            st.session_state.user = dict(usuario)
            st.session_state.role = usuario['rol']
            
            # Registrar login
            registrar_actividad(usuario['id_usuario'], 'login')
            
            return True, dict(usuario), None
        else:
            return False, None, "Email o contraseña incorrectos"
    
    except Exception as e:
        print(f"Error en login: {e}")
        return False, None, str(e)

def logout():
    """Cierra sesión del usuario"""
    if esta_autenticado():
        registrar_actividad(st.session_state.user['id_usuario'], 'logout')
    
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.role = None

def registrar_actividad(id_usuario, tipo):
    """Registra actividad del usuario"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO log_actividad (id_usuario, tipo_actividad, timestamp)
            VALUES (%s, %s, %s)
        """, (id_usuario, tipo, datetime.now()))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    except Exception as e:
        print(f"Error registrando actividad: {e}")

# ==================== GESTIÓN DE USUARIOS (SOLO ADMIN) ====================

def crear_usuario(email, nombre, password, rol='usuario'):
    """
    Crea un nuevo usuario (solo admin)
    Retorna (success, user_id, error_message)
    """
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Verificar email único
        cursor.execute(
            "SELECT id_usuario FROM usuarios_auth WHERE email = %s",
            (email,)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return False, None, "El email ya está registrado"
        
        # Hash del password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Crear usuario
        cursor.execute("""
            INSERT INTO usuarios_auth (email, nombre, password_hash, rol, activo)
            VALUES (%s, %s, %s, %s, TRUE)
            RETURNING id_usuario
        """, (email, nombre, password_hash, rol))
        
        resultado = cursor.fetchone()
        id_usuario = resultado['id_usuario']
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, id_usuario, None
    
    except Exception as e:
        print(f"Error creando usuario: {e}")
        if conn:
            conn.rollback()
        return False, None, str(e)

def listar_usuarios():
    """Lista todos los usuarios (solo admin)"""
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
                created_at
            FROM usuarios_auth
            ORDER BY created_at DESC
        """)
        
        usuarios = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(u) for u in usuarios]
    
    except Exception as e:
        print(f"Error listando usuarios: {e}")
        return []

def desactivar_usuario(id_usuario):
    """Desactiva un usuario (solo admin)"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE usuarios_auth SET activo = FALSE WHERE id_usuario = %s",
            (id_usuario,)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, None
    
    except Exception as e:
        print(f"Error desactivando usuario: {e}")
        if conn:
            conn.rollback()
        return False, str(e)

def activar_usuario(id_usuario):
    """Activa un usuario (solo admin)"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE usuarios_auth SET activo = TRUE WHERE id_usuario = %s",
            (id_usuario,)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, None
    
    except Exception as e:
        print(f"Error activando usuario: {e}")
        if conn:
            conn.rollback()
        return False, str(e)

# ==================== PERMISOS ====================

def puede_modificar_dispositivo(id_dispositivo):
    """
    Verifica si el usuario actual puede modificar un dispositivo
    - Admin: puede modificar todos
    - Usuario: solo los que él creó
    """
    if not esta_autenticado():
        return False
    
    if es_admin():
        return True
    
    # Verificar si el usuario creó este dispositivo
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT creado_por 
            FROM dispositivos 
            WHERE id_dispositivo = %s
        """, (id_dispositivo,))
        
        resultado = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if resultado:
            return resultado['creado_por'] == st.session_state.user['id_usuario']
        
        return False
    
    except Exception as e:
        print(f"Error verificando permisos: {e}")
        return False

# ==================== UI DE LOGIN ====================

def render_login():
    """Renderiza la página de login"""
    
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# 🌊 AQUARIA")
        st.markdown("### Sistema de Monitoreo")
        st.markdown("---")
        
        email = st.text_input("📧 Email", placeholder="usuario@ejemplo.com")
        password = st.text_input("🔒 Contraseña", type="password")
        
        st.markdown("")
        
        if st.button("Iniciar Sesión", use_container_width=True, type="primary"):
            if not email or not password:
                st.error("⚠️ Completa todos los campos")
            else:
                with st.spinner("Autenticando..."):
                    success, user, error = login(email, password)
                    
                    if success:
                        st.success(f"✅ Bienvenido {user['nombre']}")
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
        
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #666;'>"
            "Sistema de Alerta Temprana de Inundaciones<br>"
            "PUCMM 2026"
            "</p>",
            unsafe_allow_html=True
        )

# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n🔐 Probando sistema de autenticación...\n")
    
    # Listar usuarios
    usuarios = listar_usuarios()
    print(f"Total usuarios: {len(usuarios)}")
    for u in usuarios:
        print(f"  • {u['nombre']} ({u['email']}) - Rol: {u['rol']} - Activo: {u['activo']}")