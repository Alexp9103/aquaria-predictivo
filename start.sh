#!/bin/bash
# start.sh - Script de inicio para Railway

echo "=========================================="
echo "🌊 INICIANDO AQUARIA EN RAILWAY"
echo "=========================================="

# Paso 1: Ejecutar inicialización (entrenar si es necesario)
echo ""
echo "📦 PASO 1: Inicialización"
python init_railway.py

# Verificar si init_railway.py tuvo éxito
if [ $? -ne 0 ]; then
    echo "❌ Error en inicialización"
    exit 1
fi

# Paso 2: Iniciar Streamlit
echo ""
echo "🚀 PASO 2: Iniciando Streamlit"
echo "=========================================="
streamlit run app-mejorada-debug.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false