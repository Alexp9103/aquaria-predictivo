"""
Actualiza series temporales - VERSION SIMPLIFICADA
Solo corta datos futuros, NO descarga de Open-Meteo
"""

import pandas as pd
from datetime import datetime

def actualizar_serie(serie_historica, region_key):
    """
    Corta datos futuros de la serie histórica
    NO descarga nada nuevo (para evitar datos incorrectos)
    
    Args:
        serie_historica: pandas.Series con datos históricos
        region_key: Nombre de la región
    
    Returns:
        pandas.Series solo con datos hasta HOY
    """
    
    # Fecha actual
    hoy = pd.Timestamp(datetime.now().date())
    
    # Cortar TODO lo que sea futuro (>= hoy)
    serie_limpia = serie_historica[serie_historica.index < hoy]
    
    # Eliminar semanas con valor 0 al final (datos incompletos)
    while len(serie_limpia) > 0 and serie_limpia.iloc[-1] == 0:
        serie_limpia = serie_limpia[:-1]
    
    # Retornar serie hasta ayer
    return serie_limpia