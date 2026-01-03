"""
Script de prueba para verificar conexión a Supabase
"""

from db_config import supabase, get_pg_connection

print("\n" + "="*60)
print("PRUEBA DE CONEXIÓN A SUPABASE")
print("="*60 + "\n")

# ==================== TEST 1: SUPABASE CLIENT ====================
print("1️⃣ Probando Supabase Client...")

try:
    # Listar dispositivos
    response = supabase.table('dispositivos').select('*').execute()
    
    print(f"   ✅ Conectado")
    print(f"   📊 Dispositivos encontrados: {len(response.data)}")
    
    if len(response.data) > 0:
        print(f"\n   Ejemplo de dispositivo:")
        disp = response.data[0]
        print(f"   • ID: {disp.get('id_dispositivo')}")
        print(f"   • Código: {disp.get('codigo_hardware')}")
        print(f"   • Estado: {disp.get('estado')}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# ==================== TEST 2: POSTGRESQL DIRECTO ====================
print("\n2️⃣ Probando PostgreSQL directo...")

try:
    conn = get_pg_connection()
    cursor = conn.cursor()
    
    # Query simple
    cursor.execute("""
        SELECT 
            estado,
            COUNT(*) as total
        FROM dispositivos
        GROUP BY estado
    """)
    
    resultados = cursor.fetchall()
    
    print(f"   ✅ Conectado")
    print(f"\n   Dispositivos por estado:")
    for row in resultados:
        print(f"   • {row['estado']}: {row['total']}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# ==================== TEST 3: QUERY COMPLEJO ====================
print("\n3️⃣ Probando query complejo...")

try:
    conn = get_pg_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            d.codigo_hardware,
            d.nombre_personalizado,
            d.estado,
            COUNT(l.id_lectura) as total_lecturas,
            MAX(l.timestamp) as ultima_lectura
        FROM dispositivos d
        LEFT JOIN lecturas l ON d.id_dispositivo = l.id_dispositivo
        GROUP BY d.id_dispositivo, d.codigo_hardware, d.nombre_personalizado, d.estado
        ORDER BY total_lecturas DESC
    """)
    
    resultados = cursor.fetchall()
    
    print(f"   ✅ Query ejecutado")
    print(f"\n   Top dispositivos con más lecturas:")
    
    for i, row in enumerate(resultados[:3], 1):
        print(f"\n   {i}. {row['codigo_hardware']}")
        print(f"      Estado: {row['estado']}")
        print(f"      Lecturas: {row['total_lecturas']}")
        if row['ultima_lectura']:
            print(f"      Última: {row['ultima_lectura']}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ PRUEBAS COMPLETADAS")
print("="*60 + "\n")