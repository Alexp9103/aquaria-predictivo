"""
CRUD de Dispositivos para AQUARIA
"""

from db_config import supabase, get_pg_connection
from datetime import datetime

# ==================== DISPOSITIVOS ====================

def crear_dispositivo(codigo_hardware, nombre_codigo=None, nombre_rio=None, 
                     distancia_rio=300, distancia_orilla=250, 
                     estado='pendiente_configuracion', creado_por=None):
    """
    Crea un dispositivo manualmente
    
    Args:
        codigo_hardware: Chip ID o código único
        nombre_codigo: Código identificador (opcional)
        nombre_rio: Nombre del río (opcional)
        distancia_rio: Distancia dispositivo-río en cm
        distancia_orilla: Distancia crítica en cm
        estado: 'pendiente_configuracion' o 'activo'
    """
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Validar código hardware único
        cursor.execute(
            "SELECT id_dispositivo FROM dispositivos WHERE codigo_hardware = %s",
            (codigo_hardware,)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return None, "El código hardware ya existe"
        
        # Validar nombre_codigo único si se proporciona
        if nombre_codigo:
            cursor.execute(
                "SELECT id_dispositivo FROM dispositivos WHERE nombre_codigo = %s",
                (nombre_codigo,)
            )
            
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return None, "El código identificador ya existe"
        
        # Crear nombre personalizado
        nombre_personalizado = nombre_codigo.replace('_', ' ').title() if nombre_codigo else 'Dispositivo sin configurar'
        
        # Insertar
        cursor.execute("""
        INSERT INTO dispositivos (
            codigo_hardware,
            nombre_codigo,
            nombre_personalizado,
            nombre_rio,
            distancia_rio_dispositivo,
            distancia_dispositivo_orilla,
            estado,
            creado_por  -- ← AGREGAR
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)  -- ← Agregar %s
        RETURNING *
    """, (
        codigo_hardware,
        nombre_codigo,
        nombre_personalizado,
        nombre_rio or 'Pendiente',
        distancia_rio,
        distancia_orilla,
        estado,
        creado_por  
        ))
        
        resultado = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        return dict(resultado), None
    
    except Exception as e:
        print(f"Error creando dispositivo: {e}")
        if conn:
            conn.rollback()
        return None, str(e)

def listar_dispositivos(estado=None):
    """Lista dispositivos, opcionalmente filtrados por estado"""
    try:
        # Usar PostgreSQL directo para bypassear RLS
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        if estado:
            cursor.execute(
                "SELECT * FROM dispositivos WHERE estado = %s ORDER BY id_dispositivo",
                (estado,)
            )
        else:
            cursor.execute(
                "SELECT * FROM dispositivos ORDER BY id_dispositivo"
            )
        
        resultados = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(r) for r in resultados]
    
    except Exception as e:
        print(f"Error listando dispositivos: {e}")
        return []

def obtener_dispositivo(id_dispositivo):
    """Obtiene un dispositivo por ID"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM dispositivos WHERE id_dispositivo = %s",
            (id_dispositivo,)
        )
        
        resultado = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return dict(resultado) if resultado else None
    
    except Exception as e:
        print(f"Error obteniendo dispositivo: {e}")
        return None

def actualizar_dispositivo(id_dispositivo, datos):
    """
    Actualiza un dispositivo
    
    Args:
        id_dispositivo: ID del dispositivo
        datos: Dict con campos a actualizar
        
    Ejemplo:
        actualizar_dispositivo(1, {
            'nombre_codigo': 'RIO_YUNA',
            'nombre_rio': 'Río Yuna',
            'estado': 'activo'
        })
    """
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Construir query dinámicamente
        campos = ', '.join([f"{k} = %s" for k in datos.keys()])
        valores = list(datos.values()) + [id_dispositivo]
        
        query = f"UPDATE dispositivos SET {campos} WHERE id_dispositivo = %s RETURNING *"
        
        cursor.execute(query, valores)
        resultado = cursor.fetchone()
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return dict(resultado) if resultado else None
    
    except Exception as e:
        print(f"Error actualizando dispositivo: {e}")
        if conn:
            conn.rollback()
        return None

def activar_dispositivo(id_dispositivo, nombre_codigo, nombre_rio, 
                        distancia_rio=300, distancia_orilla=250):
    """
    Activa un dispositivo pendiente
    
    Args:
        id_dispositivo: ID del dispositivo
        nombre_codigo: Código legible (ej: RIO_YUNA)
        nombre_rio: Nombre del río (ej: Río Yuna)
        distancia_rio: Distancia del dispositivo al río en cm
        distancia_orilla: Distancia crítica en cm
    """
    datos = {
        'nombre_codigo': nombre_codigo,
        'nombre_personalizado': nombre_codigo.replace('_', ' ').title(),
        'nombre_rio': nombre_rio,
        'distancia_rio_dispositivo': distancia_rio,
        'distancia_dispositivo_orilla': distancia_orilla,
        'estado': 'activo'
    }
    
    return actualizar_dispositivo(id_dispositivo, datos)

def desactivar_dispositivo(id_dispositivo):
    """Desactiva un dispositivo"""
    return actualizar_dispositivo(id_dispositivo, {'estado': 'inactivo'})

# ==================== LECTURAS ====================

def obtener_ultimas_lecturas(id_dispositivo, limit=50):
    """Obtiene últimas N lecturas de un dispositivo"""
    try:
        response = supabase.table('lecturas')\
            .select('*')\
            .eq('id_dispositivo', id_dispositivo)\
            .order('timestamp', desc=True)\
            .limit(limit)\
            .execute()
        
        return response.data
    
    except Exception as e:
        print(f"Error obteniendo lecturas: {e}")
        return []

def obtener_estadisticas_dispositivo(id_dispositivo):
    """Obtiene estadísticas de un dispositivo usando PostgreSQL"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_lecturas,
                AVG(distancia_medida) as promedio_distancia,
                MIN(distancia_medida) as min_distancia,
                MAX(distancia_medida) as max_distancia,
                MAX(timestamp) as ultima_lectura,
                COUNT(CASE WHEN nivel_riesgo = 'critico' THEN 1 END) as alertas_criticas,
                COUNT(CASE WHEN nivel_riesgo = 'alerta' THEN 1 END) as alertas_altas
            FROM lecturas
            WHERE id_dispositivo = %s
        """, (id_dispositivo,))
        
        resultado = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return dict(resultado) if resultado else None
    
    except Exception as e:
        print(f"Error obteniendo estadísticas: {e}")
        return None

# ==================== USUARIOS ====================

def listar_usuarios_activos():
    """Lista usuarios activos"""
    try:
        response = supabase.table('usuarios')\
            .select('*')\
            .eq('activo', True)\
            .execute()
        
        return response.data
    
    except Exception as e:
        print(f"Error listando usuarios: {e}")
        return []

def obtener_suscripciones(id_dispositivo):
    """Obtiene usuarios suscritos a un dispositivo"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                u.id_usuario,
                u.nombre,
                u.telefono,
                u.email,
                ud.fecha_suscripcion
            FROM usuarios u
            JOIN usuarios_dispositivos ud ON u.id_usuario = ud.id_usuario
            WHERE ud.id_dispositivo = %s AND u.activo = TRUE
            ORDER BY u.nombre
        """, (id_dispositivo,))
        
        resultados = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(r) for r in resultados]
    
    except Exception as e:
        print(f"Error obteniendo suscripciones: {e}")
        return []

# ==================== ALERTAS ====================

def obtener_alertas_recientes(id_dispositivo=None, limit=20):
    """Obtiene alertas recientes"""
    try:
        query = supabase.table('historial_alertas')\
            .select('*, dispositivos(nombre_personalizado, nombre_rio)')\
            .order('timestamp', desc=True)\
            .limit(limit)
        
        if id_dispositivo:
            query = query.eq('id_dispositivo', id_dispositivo)
        
        response = query.execute()
        return response.data
    
    except Exception as e:
        print(f"Error obteniendo alertas: {e}")
        return []

# ==================== VALIDACIONES ====================

def validar_nombre_codigo_unico(nombre_codigo, excluir_id=None):
    """Verifica que un nombre_codigo no exista"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        if excluir_id:
            cursor.execute(
                "SELECT id_dispositivo FROM dispositivos WHERE nombre_codigo = %s AND id_dispositivo != %s",
                (nombre_codigo, excluir_id)
            )
        else:
            cursor.execute(
                "SELECT id_dispositivo FROM dispositivos WHERE nombre_codigo = %s",
                (nombre_codigo,)
            )
        
        resultado = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return resultado is None
    
    except Exception as e:
        print(f"Error validando nombre: {e}")
        return False

# ==================== FUNCIONES DE PRUEBA ====================

if __name__ == "__main__":
    print("\n🧪 Probando CRUD de dispositivos...\n")
    
    # Listar todos
    print("1. Dispositivos activos:")
    activos = listar_dispositivos('activo')
    for d in activos:
        print(f"   • {d['nombre_personalizado']} ({d['codigo_hardware']})")
    
    # Pendientes
    print("\n2. Dispositivos pendientes:")
    pendientes = listar_dispositivos('pendiente_configuracion')
    for d in pendientes:
        print(f"   • {d['codigo_hardware']} - ID: {d['id_dispositivo']}")
    
    # Estadísticas
    if activos:
        print(f"\n3. Estadísticas del primer dispositivo:")
        stats = obtener_estadisticas_dispositivo(activos[0]['id_dispositivo'])
        if stats:
            print(f"   • Total lecturas: {stats['total_lecturas']}")
            print(f"   • Promedio: {stats['promedio_distancia']:.1f} cm")
            print(f"   • Alertas críticas: {stats['alertas_criticas']}")
    
    print("\n✅ Pruebas completadas\n")