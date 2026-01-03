"""
Calibración estratificada AQUARIA - Con modo ULTRA-AGRESIVO
Para modelos con sobreestimación severa
"""

import pickle
import numpy as np
from pathlib import Path

# Sesgos observados (mm) - Extraído del análisis
SESGOS_ESTRATIFICADOS = {
    'grupo_1_norte_cibao': {
        'muy_bajo': +17.12,
        'bajo':     +9.65,
        'medio':    -0.27,
        'alto':     -16.32,
        'muy_alto': -66.89
    },
    'grupo_2_sur_seco': {
        'muy_bajo': +4.82,
        'bajo':     -0.23,
        'medio':    -11.80,
        'alto':     -33.98,
        'muy_alto': -61.36
    },
    'grupo_3_este_capital': {
        'muy_bajo': +10.29,
        'bajo':     +8.11,
        'medio':    -0.19,
        'alto':     -15.20,
        'muy_alto': -60.33
    }
}

UMBRALES = {
    'muy_bajo': (0, 5),
    'bajo': (5, 15),
    'medio': (15, 30),
    'alto': (30, 60),
    'muy_alto': (60, float('inf'))
}

def crear_calibrador(nombre_grupo, sesgos):
    """Crea calibrador con 4 modos (incluyendo ultra-agresivo)"""
    return {
        'grupo': nombre_grupo,
        'sesgos': sesgos,
        'umbrales': UMBRALES,
        'configuraciones': {
            'conservador': {
                'agresividad': 0.3,
                'factor_variabilidad': 1.1,
                'descripcion': 'Corrección mínima (mejor para alertas tempranas)'
            },
            'balanceado': {
                'agresividad': 0.5,
                'factor_variabilidad': 1.2,
                'descripcion': 'Balance MAE vs Variabilidad (uso general)'
            },
            'agresivo': {
                'agresividad': 0.7,
                'factor_variabilidad': 1.25,
                'descripcion': 'Corrección fuerte'
            },
            # 🔥 MODO ULTRA-AGRESIVO v3 - Solo corrección pura
            'ultra_agresivo': {
                'agresividad': 1.0,  # 100% de corrección del sesgo
                'factor_variabilidad': 1.0,  # Sin aumentar
                'factor_variabilidad_post': 1.0,  # Sin aumentar
                'aplicar_reduccion_extra': True,  # Flag para reducción adicional
                'reduccion_extra_porcentaje': 0.3,  # Reducir 30% extra del resultado
                'descripcion': 'Máxima corrección + reducción extra para sobreestimación severa'
            }
        },
        'version': '2.1_ultra_agresivo',
        'descripcion': 'Calibrador estratificado con modo ultra-agresivo'
    }

def main():
    print("="*70)
    print("🔧 CALIBRACIÓN ESTRATIFICADA – AQUARIA (CON ULTRA-AGRESIVO)")
    print("="*70)
    
    Path("modelos").mkdir(exist_ok=True)
    
    for grupo, sesgos in SESGOS_ESTRATIFICADOS.items():
        print(f"\n📦 {grupo}")
        
        for nivel, sesgo in sesgos.items():
            signo = "📈" if sesgo > 0 else "📉"
            print(f"   {signo} {nivel:12s}: {sesgo:+7.2f} mm")
        
        cal = crear_calibrador(grupo, sesgos)
        
        out = f"modelos/calibrador_{grupo}.pkl"
        with open(out, "wb") as f:
            pickle.dump(cal, f)
        
        print(f"   ✅ {out}")
    
    print(f"\n{'='*70}")
    print("✅ Calibradores con modo ULTRA-AGRESIVO listos")
    print("="*70)
    
    # Demo de corrección ultra-agresiva
    print("\n🧪 DEMO - Corrección Ultra-Agresiva vs Normal")
    print("="*70)
    
    with open("modelos/calibrador_grupo_1_norte_cibao.pkl", "rb") as f:
        cal = pickle.load(f)
    
    # Predicción típica problemática
    pred_original = 50.0  # mm (el modelo predice ~50mm constantemente)
    
    print(f"\n📊 Predicción original: {pred_original:.1f} mm")
    print(f"{'Modo':<20} {'Factor':<10} {'Predicción Corregida':<25} {'Reducción'}")
    print("-"*70)
    
    for modo in ['conservador', 'balanceado', 'agresivo', 'ultra_agresivo']:
        config = cal['configuraciones'][modo]
        agr = config['agresividad']
        var = config['factor_variabilidad']
        
        # Determinar nivel
        if pred_original < 5:
            nivel = 'muy_bajo'
        elif pred_original < 15:
            nivel = 'bajo'
        elif pred_original < 30:
            nivel = 'medio'
        elif pred_original < 60:
            nivel = 'alto'
        else:
            nivel = 'muy_alto'
        
        sesgo = cal['sesgos'][nivel]
        
        # Aplicar corrección
        pred_con_var = pred_original * var
        correccion = agr * sesgo
        pred_corregida = max(0.1, pred_con_var - correccion)
        
        reduccion = pred_original - pred_corregida
        
        print(f"{modo:<20} {agr:.2f} (x{var:.1f})  {pred_corregida:>20.1f} mm     {reduccion:+.1f} mm")
    
    print("\n" + "="*70)
    print("💡 Uso recomendado:")
    print("   • ULTRA-AGRESIVO: Para Grupo 1 Norte Cibao (MAE>30mm)")
    print("   • AGRESIVO: Para otros grupos con sobreestimación")
    print("   • BALANCEADO: Cuando el modelo funciona bien")
    print("="*70)

if __name__ == "__main__":
    main()