import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from pathlib import Path

# Verificar archivos
carpeta = '/home/juanpucmm/modelo predectivo/formatoadecuado'
archivos = list(Path(carpeta).glob("*_diario_*.csv"))
print(f"Archivos encontrados en {carpeta}: {len(archivos)}")
for archivo in archivos:
    print(f"  - {archivo.name}")

def analizar_y_agrupar_ciudades_rd(carpeta='datos'):
    """Analiza las 15 ciudades y las agrupa automáticamente"""
    
    # Tus 15 ciudades
    ciudades_esperadas = [
        'santo_domingo', 'la_sabana', 'punta_cana', 'montecristi',
        'las_americas', 'la_union', 'la_romana', 'jimani',
        'el_higuero', 'catey', 'cabrera', 'bayaguana',
        'barahona', 'arroyo_hondo', 'santiago'
    ]
    
    print("="*70)
    print("ANÁLISIS DE 15 CIUDADES - REPÚBLICA DOMINICANA")
    print("="*70)
    
    # Cargar datos
    datos_ciudades = {}
    archivos = list(Path(carpeta).glob("*.csv"))
    
    for archivo in archivos:
        ciudad = archivo.stem.replace(' ', '_').lower()
        
        try:
            df = pd.read_csv(archivo, parse_dates=['Fecha'])
            df = df.set_index('Fecha')
            
            # Resample a semanal
            semanal = df['Precip_mm'].resample('W-SUN').sum()
            
            # Filtrar datos válidos
            if len(semanal) < 52:  # Menos de 1 año
                print(f"⚠ {ciudad}: Datos insuficientes ({len(semanal)} semanas)")
                continue
            
            datos_ciudades[ciudad] = {
                'serie': semanal,
                'mean': semanal.mean(),
                'std': semanal.std(),
                'cv': semanal.std() / semanal.mean() if semanal.mean() > 0 else 0,
                'max': semanal.max(),
                'registros': len(semanal),
                'años': (semanal.index[-1] - semanal.index[0]).days / 365.25,
                'inicio': semanal.index[0],
                'fin': semanal.index[-1]
            }
            
            print(f"✓ {ciudad:20s}: {len(semanal):4d} semanas ({datos_ciudades[ciudad]['años']:.1f} años)")
            
        except Exception as e:
            print(f"✗ {ciudad}: Error - {str(e)}")
    
    if len(datos_ciudades) < 2:
        print("\nError: Se necesitan al menos 2 ciudades con datos válidos")
        return
    
    print(f"\n{len(datos_ciudades)} ciudades cargadas exitosamente")
    
    # Resumen estadístico
    print("\n" + "="*70)
    print("ESTADÍSTICAS DESCRIPTIVAS")
    print("="*70)
    print(f"{'Ciudad':<20} {'Media':<10} {'Std':<10} {'CV':<10} {'Max':<10}")
    print("-"*70)
    
    for ciudad in sorted(datos_ciudades.keys()):
        d = datos_ciudades[ciudad]
        print(f"{ciudad:<20} {d['mean']:>9.1f} {d['std']:>9.1f} {d['cv']:>9.2f} {d['max']:>9.1f}")
    
    # Alinear series temporales
    fechas_comunes = set(datos_ciudades[list(datos_ciudades.keys())[0]]['serie'].index)
    for ciudad in datos_ciudades:
        fechas_comunes &= set(datos_ciudades[ciudad]['serie'].index)
    
    fechas_comunes = sorted(list(fechas_comunes))
    print(f"\nFechas en común: {len(fechas_comunes)} semanas")
    
    if len(fechas_comunes) < 52:
        print("⚠ ADVERTENCIA: Pocas fechas comunes. Los datos pueden tener períodos diferentes.")
        # Usar todas las fechas disponibles con interpolación
        todas_fechas = set()
        for ciudad in datos_ciudades:
            todas_fechas |= set(datos_ciudades[ciudad]['serie'].index)
        fechas_comunes = sorted(list(todas_fechas))
    
    # Crear matriz alineada (con NaN donde falten datos)
    matriz_datos = pd.DataFrame({
        ciudad: datos_ciudades[ciudad]['serie'].reindex(fechas_comunes)
        for ciudad in datos_ciudades
    })
    
    # Rellenar NaN con interpolación lineal
    matriz_datos = matriz_datos.interpolate(method='linear', limit_direction='both')
    matriz_datos = matriz_datos.fillna(0)
    
    # Correlaciones
    correlaciones = matriz_datos.corr()
    
    print("\n" + "="*70)
    print("ANÁLISIS DE CORRELACIÓN")
    print("="*70)
    
    # Pares con alta correlación
    print("\nCiudades MUY SIMILARES (correlación > 0.70):")
    pares_altos = []
    for i in range(len(correlaciones)):
        for j in range(i+1, len(correlaciones)):
            corr = correlaciones.iloc[i, j]
            if corr > 0.70:
                ciudad1 = correlaciones.index[i]
                ciudad2 = correlaciones.columns[j]
                pares_altos.append((ciudad1, ciudad2, corr))
                print(f"  {ciudad1:<20} <-> {ciudad2:<20}: {corr:.3f}")
    
    # Pares con baja correlación
    print("\nCiudades MUY DIFERENTES (correlación < 0.40):")
    pares_bajos = []
    for i in range(len(correlaciones)):
        for j in range(i+1, len(correlaciones)):
            corr = correlaciones.iloc[i, j]
            if corr < 0.40:
                ciudad1 = correlaciones.index[i]
                ciudad2 = correlaciones.columns[j]
                pares_bajos.append((ciudad1, ciudad2, corr))
                print(f"  {ciudad1:<20} <-> {ciudad2:<20}: {corr:.3f}")
    
    # Clustering automático
    print("\n" + "="*70)
    print("CLUSTERING AUTOMÁTICO")
    print("="*70)
    
    distancias = pdist(matriz_datos.T, metric='correlation')
    linkage_matrix = linkage(distancias, method='ward')
    
    # Determinar número óptimo de clusters (2-5 grupos)
    clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
    
    grupos_automaticos = {}
    for i, ciudad in enumerate(correlaciones.columns):
        grupo = f"grupo_{clusters[i]}"
        if grupo not in grupos_automaticos:
            grupos_automaticos[grupo] = []
        grupos_automaticos[grupo].append(ciudad)
    
    print(f"\nGrupos detectados: {len(grupos_automaticos)}")
    for grupo, ciudades in sorted(grupos_automaticos.items()):
        print(f"\n{grupo.upper()}:")
        for ciudad in ciudades:
            d = datos_ciudades[ciudad]
            print(f"  - {ciudad:<20} (media: {d['mean']:.1f}mm, std: {d['std']:.1f}mm)")
    
    # Visualizaciones
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Mapa de calor
    ax1 = fig.add_subplot(gs[0, :2])
    sns.heatmap(correlaciones, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0.5, ax=ax1, cbar_kws={'label': 'Correlación'},
                square=True, linewidths=0.5)
    ax1.set_title('Matriz de Correlación - 15 Ciudades RD', fontsize=14, weight='bold')
    
    # 2. Dendrograma
    ax2 = fig.add_subplot(gs[0, 2])
    dendrogram(linkage_matrix, labels=correlaciones.columns, 
               orientation='right', ax=ax2)
    ax2.set_title('Clustering Jerárquico', fontsize=12, weight='bold')
    ax2.set_xlabel('Distancia')
    
    # 3. Distribuciones por grupo
    ax3 = fig.add_subplot(gs[1, :])
    for grupo, ciudades in grupos_automaticos.items():
        for ciudad in ciudades:
            serie = datos_ciudades[ciudad]['serie']
            ax3.hist(serie, bins=30, alpha=0.3, label=f"{ciudad} ({grupo})")
    ax3.set_xlabel('Precipitación semanal (mm)', fontsize=11)
    ax3.set_ylabel('Frecuencia', fontsize=11)
    ax3.set_title('Distribución de Precipitación por Ciudad y Grupo', fontsize=12, weight='bold')
    ax3.legend(fontsize=7, ncol=3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Patrones estacionales
    ax4 = fig.add_subplot(gs[2, 0])
    for ciudad, datos in datos_ciudades.items():
        patron = datos['serie'].groupby(datos['serie'].index.month).mean()
        ax4.plot(patron.index, patron.values, marker='o', label=ciudad, alpha=0.7)
    ax4.set_xlabel('Mes', fontsize=11)
    ax4.set_ylabel('Precipitación (mm/semana)', fontsize=11)
    ax4.set_title('Patrones Estacionales', fontsize=12, weight='bold')
    ax4.legend(fontsize=7, ncol=2)
    ax4.set_xticks(range(1,13))
    ax4.grid(True, alpha=0.3)
    
    # 5. Box plot por grupo
    ax5 = fig.add_subplot(gs[2, 1])
    datos_box = []
    labels_box = []
    for grupo, ciudades in grupos_automaticos.items():
        for ciudad in ciudades:
            datos_box.append(datos_ciudades[ciudad]['serie'].values)
            labels_box.append(f"{ciudad[:10]}")
    ax5.boxplot(datos_box, labels=labels_box)
    ax5.set_ylabel('Precipitación (mm/semana)', fontsize=11)
    ax5.set_title('Distribución por Ciudad', fontsize=12, weight='bold')
    ax5.tick_params(axis='x', rotation=90, labelsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Estadísticas resumen
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    corr_promedio = correlaciones.values[np.triu_indices_from(correlaciones.values, k=1)].mean()
    
    texto_resumen = f"""
RESUMEN ESTADÍSTICO

Total ciudades: {len(datos_ciudades)}
Grupos detectados: {len(grupos_automaticos)}

Correlación promedio: {corr_promedio:.3f}
Pares muy similares: {len(pares_altos)}
Pares muy diferentes: {len(pares_bajos)}

Precipitación promedio:
  Media: {np.mean([d['mean'] for d in datos_ciudades.values()]):.1f} mm/sem
  Rango: {min([d['mean'] for d in datos_ciudades.values()]):.1f} - {max([d['mean'] for d in datos_ciudades.values()]):.1f}

Variabilidad:
  CV promedio: {np.mean([d['cv'] for d in datos_ciudades.values()]):.2f}
"""
    
    ax6.text(0.1, 0.5, texto_resumen, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.savefig('analisis_15_ciudades_rd.png', dpi=300, bbox_inches='tight')
    print("\nGráfico guardado: analisis_15_ciudades_rd.png")
    
    # RECOMENDACIÓN FINAL
    print("\n" + "="*70)
    print("RECOMENDACIÓN ESTRATÉGICA")
    print("="*70)
    
    if corr_promedio > 0.65:
        print("\n✓ ESTRATEGIA: Modelo MULTI-CIUDAD único")
        print("  Las ciudades tienen patrones suficientemente similares")
        print("  Entrenar UN modelo con todas las ciudades y features geográficas")
        
    elif len(grupos_automaticos) <= 4:
        print(f"\n⚠ ESTRATEGIA: {len(grupos_automaticos)} modelos por GRUPO")
        print("  Hay variabilidad moderada entre ciudades")
        print(f"  Entrenar {len(grupos_automaticos)} modelos separados por grupo climático")
        print("\n  Grupos sugeridos:")
        for grupo, ciudades in grupos_automaticos.items():
            print(f"    {grupo}: {', '.join(ciudades)}")
            
    else:
        print("\n✗ ESTRATEGIA: Modelos INDIVIDUALES")
        print("  Las ciudades tienen patrones muy diferentes")
        print("  Entrenar un modelo por ciudad para máxima precisión")
    
    return datos_ciudades, correlaciones, grupos_automaticos

# Ejecutar análisis
datos, corr, grupos = analizar_y_agrupar_ciudades_rd(carpeta='/home/juanpucmm/modelo predectivo/formatoadecuado')
