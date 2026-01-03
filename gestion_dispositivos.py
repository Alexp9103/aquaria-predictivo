"""
Tab de Gestión de Dispositivos para AQUARIA Streamlit
Importar en app.py y agregar como tab
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import crud_dispositivos as crud

import auth

def render_gestion_dispositivos():
    """Renderiza el tab completo de gestión de dispositivos"""
    
    st.header("📡 Gestión de Dispositivos AQUARIA")
    
    # Pestañas internas
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "➕ Crear Dispositivo",
        "🆕 Dispositivos Pendientes", 
        "✅ Dispositivos Activos",
        "📊 Estadísticas"
    ])
    
    # ==================== SUBTAB 1: CREAR DISPOSITIVO ====================
    with subtab1:
        st.subheader("➕ Crear Dispositivo Manualmente")
        
        st.info("""
        **Opciones para agregar dispositivos:**
        
        1️⃣ **Auto-registro (Recomendado):** El T-Call se conecta y se registra automáticamente
        
        2️⃣ **Creación manual:** Crear un dispositivo antes de tener el hardware físico
        """)
        
        with st.form("form_crear_dispositivo"):
            st.markdown("### 📝 Información del Dispositivo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                codigo_hardware = st.text_input(
                    "Código Hardware*",
                    placeholder="AQUARIA_CHIP_ABC123 o AQUARIA_RIO_OZAMA_02",
                    help="Identificador único del dispositivo (chip ID o código personalizado)"
                )
                
                nombre_codigo = st.text_input(
                    "Código Identificador",
                    placeholder="RIO_OZAMA",
                    help="Código corto para identificar el dispositivo (opcional)"
                )
            
            with col2:
                nombre_rio = st.text_input(
                    "Nombre del Río",
                    placeholder="Río Ozama",
                    help="Nombre completo del río a monitorear (opcional)"
                )
                
                estado_inicial = st.selectbox(
                    "Estado Inicial",
                    ["pendiente_configuracion", "activo"],
                    format_func=lambda x: "Pendiente de Configuración" if x == "pendiente_configuracion" else "Activo"
                )
            
            st.markdown("### 📏 Configuración de Medición")
            
            col1, col2 = st.columns(2)
            
            with col1:
                distancia_rio = st.number_input(
                    "Distancia Río-Dispositivo (cm)",
                    min_value=50,
                    max_value=1000,
                    value=300,
                    help="Altura del sensor sobre el lecho del río"
                )
            
            with col2:
                distancia_orilla = st.number_input(
                    "Distancia Crítica (cm)",
                    min_value=50,
                    max_value=1000,
                    value=250,
                    help="Distancia de orilla a nivel crítico"
                )
            
            st.markdown("---")
            
            crear = st.form_submit_button(
                "➕ Crear Dispositivo",
                type="primary",
                use_container_width=True
            )
        
        if crear:
            # Validaciones
            if not codigo_hardware:
                st.error("⚠️ El código hardware es obligatorio")
            
            elif not codigo_hardware.replace('_', '').replace('-', '').isalnum():
                st.error("⚠️ El código hardware solo puede contener letras, números, guiones y guiones bajos")
            
            elif nombre_codigo and not nombre_codigo.replace('_', '').isalnum():
                st.error("⚠️ El código identificador solo puede contener letras, números y guiones bajos")
            
            else:
                # Crear dispositivo
                resultado, error = crud.crear_dispositivo(
                codigo_hardware,
                nombre_codigo if nombre_codigo else None,
                nombre_rio if nombre_rio else None,
                distancia_rio,
                distancia_orilla,
                estado_inicial,
                creado_por=auth.obtener_usuario_actual()['id_usuario']  # ← AGREGAR
            )
                
                if resultado:
                    st.success(f"✅ Dispositivo '{codigo_hardware}' creado exitosamente")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ Error: {error}")
    
    # ==================== SUBTAB 2: PENDIENTES ====================
    with subtab2:
        st.subheader("🆕 Dispositivos Pendientes de Configuración")
        
        pendientes = crud.listar_dispositivos('pendiente_configuracion')
        
        if not pendientes:
            st.success("✅ No hay dispositivos pendientes de configuración")
        else:
            st.info(f"📋 {len(pendientes)} dispositivo(s) esperando configuración")
            
            for disp in pendientes:
                with st.expander(f"🔧 {disp['codigo_hardware']}", expanded=True):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **Información del Dispositivo:**
                        - **Chip ID:** `{disp['codigo_hardware']}`
                        - **Registrado:** {disp.get('created_at', 'N/A')[:10] if disp.get('created_at') else 'N/A'}
                        - **Estado:** {disp['estado']}
                        """)
                    
                    with col2:
                        # Mostrar estadísticas si tiene lecturas
                        stats = crud.obtener_estadisticas_dispositivo(disp['id_dispositivo'])
                        if stats and stats['total_lecturas'] > 0:
                            st.metric("Lecturas", stats['total_lecturas'])
                            st.metric("Última", 
                                     stats['ultima_lectura'].strftime('%d/%m/%Y %H:%M') 
                                     if stats['ultima_lectura'] else 'N/A')
                    
                    st.markdown("---")
                    st.markdown("### ⚙️ Configurar Dispositivo")
                    
                    # Formulario de activación
                    with st.form(f"form_{disp['id_dispositivo']}"):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            nombre_codigo = st.text_input(
                                "Código Identificador*",
                                placeholder="RIO_YUNA",
                                help="Código único para identificar el dispositivo (ej: RIO_YAQUE, RIO_OZAMA)",
                                key=f"cod_{disp['id_dispositivo']}"
                            )
                            
                            nombre_rio = st.text_input(
                                "Nombre del Río*",
                                placeholder="Río Yuna",
                                help="Nombre completo del río",
                                key=f"rio_{disp['id_dispositivo']}"
                            )
                        
                        with col2:
                            distancia_rio = st.number_input(
                                "Distancia Río-Dispositivo (cm)*",
                                min_value=50,
                                max_value=1000,
                                value=300,
                                help="Distancia del dispositivo al lecho del río",
                                key=f"dist_rio_{disp['id_dispositivo']}"
                            )
                            
                            distancia_orilla = st.number_input(
                                "Distancia Crítica (cm)*",
                                min_value=50,
                                max_value=1000,
                                value=250,
                                help="Distancia de la orilla al agua en nivel crítico",
                                key=f"dist_orilla_{disp['id_dispositivo']}"
                            )
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            activar = st.form_submit_button(
                                "✅ Activar Dispositivo",
                                type="primary",
                                use_container_width=True
                            )
                        
                        with col2:
                            cancelar = st.form_submit_button(
                                "❌ Cancelar",
                                use_container_width=True
                            )
                    
                    if activar:
                        # Validaciones
                        if not nombre_codigo or not nombre_rio:
                            st.error("⚠️ Completa todos los campos obligatorios")
                        
                        elif not nombre_codigo.replace('_', '').isalnum():
                            st.error("⚠️ El código solo puede contener letras, números y guiones bajos")
                        
                        elif not crud.validar_nombre_codigo_unico(nombre_codigo, disp['id_dispositivo']):
                            st.error(f"⚠️ El código '{nombre_codigo}' ya está en uso")
                        
                        else:
                            # Activar dispositivo
                            resultado = crud.activar_dispositivo(
                                disp['id_dispositivo'],
                                nombre_codigo,
                                nombre_rio,
                                distancia_rio,
                                distancia_orilla
                            )
                            
                            if resultado:
                                st.success(f"✅ Dispositivo '{nombre_codigo}' activado exitosamente")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Error activando dispositivo")
    
    # ==================== SUBTAB 3: ACTIVOS ====================
    with subtab3:
        st.subheader("✅ Dispositivos Activos")
        
        activos = crud.listar_dispositivos('activo')
        
        if not activos:
            st.warning("⚠️ No hay dispositivos activos")
        else:
            st.success(f"📊 {len(activos)} dispositivo(s) activo(s)")
            
            # Tabla resumen
            df_activos = pd.DataFrame([{
                'ID': d['id_dispositivo'],
                'Código': d.get('nombre_codigo') or 'N/A',
                'Nombre': d['nombre_personalizado'],
                'Río': d['nombre_rio'],
                'Estado': d['estado']
            } for d in activos])
            
            st.dataframe(df_activos, use_container_width=True, hide_index=True)
            
            # Detalles de cada dispositivo
            st.markdown("---")
            st.markdown("### 🔍 Detalles por Dispositivo")
            
            for disp in activos:
                with st.expander(f"📡 {disp['nombre_personalizado']} ({disp.get('nombre_codigo') or disp['codigo_hardware']})"):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Información General:**
                        - **ID:** {disp['id_dispositivo']}
                        - **Código Hardware:** `{disp['codigo_hardware']}`
                        - **Código Identificador:** {disp.get('nombre_codigo') or 'N/A'}
                        - **Río:** {disp['nombre_rio']}
                        - **Estado:** {disp['estado']}
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **Configuración:**
                        - **Distancia Río-Dispositivo:** {disp['distancia_rio_dispositivo']} cm
                        - **Distancia Crítica:** {disp['distancia_dispositivo_orilla']} cm
                        - **Umbral Normal:** {disp['umbral_normal_max']}%
                        - **Umbral Precaución:** {disp['umbral_precaucion_max']}%
                        - **Umbral Alerta:** {disp['umbral_alerta_max']}%
                        """)
                    
                    # Estadísticas
                    stats = crud.obtener_estadisticas_dispositivo(disp['id_dispositivo'])
                    
                    if stats:
                        st.markdown("---")
                        st.markdown("**📊 Estadísticas:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Lecturas", stats['total_lecturas'])
                        
                        with col2:
                            st.metric("Promedio", f"{stats['promedio_distancia']:.1f} cm")
                        
                        with col3:
                            st.metric("Alertas Críticas", stats['alertas_criticas'])
                        
                        with col4:
                            st.metric("Alertas Altas", stats['alertas_altas'])
                    
                    # Últimas lecturas
                    lecturas = crud.obtener_ultimas_lecturas(disp['id_dispositivo'], 10)
                    
                    if lecturas:
                        st.markdown("---")
                        st.markdown("**📈 Últimas 10 Lecturas:**")
                        
                        df_lecturas = pd.DataFrame([{
                            'Fecha': l['timestamp'][:19],
                            'Distancia (cm)': l['distancia_medida'],
                            'Riesgo (%)': f"{l['porcentaje_riesgo']:.1f}",
                            'Nivel': l['nivel_riesgo']
                        } for l in lecturas])
                        
                        st.dataframe(df_lecturas, use_container_width=True, hide_index=True)
                    
                    # Botón desactivar
                    st.markdown("---")
                    if auth.puede_modificar_dispositivo(disp['id_dispositivo']):
                        if st.button(f"🔴 Desactivar Dispositivo", key=f"deact_{disp['id_dispositivo']}"):
                            if crud.desactivar_dispositivo(disp['id_dispositivo']):
                                st.success("✅ Dispositivo desactivado")
                                st.rerun()
                            else:
                                st.error("❌ Error desactivando dispositivo")
                    else:
                        st.warning("🔒 No tienes permisos para modificar este dispositivo")

    
    # ==================== SUBTAB 4: ESTADÍSTICAS ====================
    with subtab4:
        st.subheader("📊 Estadísticas Generales")
        
        todos = crud.listar_dispositivos()
        activos = crud.listar_dispositivos('activo')
        pendientes = crud.listar_dispositivos('pendiente_configuracion')
        inactivos = crud.listar_dispositivos('inactivo')
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Dispositivos", len(todos))
        
        with col2:
            st.metric("Activos", len(activos), delta=f"+{len(activos)}")
        
        with col3:
            st.metric("Pendientes", len(pendientes), 
                     delta=f"-{len(pendientes)}" if len(pendientes) > 0 else "0")
        
        with col4:
            st.metric("Inactivos", len(inactivos))
        
        # Gráfico de distribución
        if todos:
            import plotly.express as px
            
            df_estados = pd.DataFrame([
                {'Estado': 'Activos', 'Cantidad': len(activos)},
                {'Estado': 'Pendientes', 'Cantidad': len(pendientes)},
                {'Estado': 'Inactivos', 'Cantidad': len(inactivos)}
            ])
            
            fig = px.pie(df_estados, values='Cantidad', names='Estado',
                        title='Distribución de Dispositivos por Estado',
                        color='Estado',
                        color_discrete_map={
                            'Activos': '#2E86AB',
                            'Pendientes': '#FFA500',
                            'Inactivos': '#E63946'
                        })
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Top dispositivos por lecturas
        st.markdown("---")
        st.markdown("### 🏆 Top Dispositivos por Lecturas")
        
        if activos:
            stats_todos = []
            for d in activos:
                stats = crud.obtener_estadisticas_dispositivo(d['id_dispositivo'])
                if stats:
                    stats_todos.append({
                        'Dispositivo': d['nombre_personalizado'],
                        'Total Lecturas': stats['total_lecturas'],
                        'Promedio (cm)': f"{stats['promedio_distancia']:.1f}",
                        'Alertas Críticas': stats['alertas_criticas']
                    })
            
            if stats_todos:
                df_stats = pd.DataFrame(stats_todos)
                df_stats = df_stats.sort_values('Total Lecturas', ascending=False)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)

# ==================== FUNCIÓN PARA INTEGRAR EN APP.PY ====================

def integrar_en_app():
    """
    Agregar esto en app.py:
    
    # Después de tus imports
    import gestion_dispositivos
    
    # En la sección de tabs, agregar:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Predicciones", 
        "📈 Análisis Histórico", 
        "🎯 Métricas", 
        "📡 Dispositivos",  # ← NUEVO
        "ℹ️ Acerca de"
    ])
    
    # Agregar el nuevo tab
    with tab5:  # O el número que corresponda
        gestion_dispositivos.render_gestion_dispositivos()
    """
    pass

if __name__ == "__main__":
    # Para probar standalone
    st.set_page_config(page_title="Gestión Dispositivos", layout="wide")
    render_gestion_dispositivos()