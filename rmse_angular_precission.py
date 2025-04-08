import os
import numpy as np
import matplotlib.pyplot as plt
from utils.metric_utils import (
    compute_planet_metrics,  # el que devuelve estadísticos
    compute_planet_distributions  # hipotético: devuelve arrays de errores angulares
)

def boxplot_angular_distributions(distributions, planets):
    """
    Crea un diagrama de cajas (boxplot) comparando, para cada planeta,
    la distribución del error angular de Libnova y Siderust.
    
    Parameters:
        distributions (dict): 
            {
              'libnova':  { 'mercury': [vals...], 'venus': [vals...], ... },
              'siderust': { 'mercury': [vals...], 'venus': [vals...], ... }
            }
        planets (list): Ej. ["mercury", "venus", ...]
    """
    # Preparamos las estructuras de datos para el boxplot
    # Queremos que cada planeta tenga 2 cajas: Libnova y Siderust
    # Por ejemplo: [ [libnova mercurio], [siderust mercurio], [libnova venus], [siderust venus], ... ]
    data_for_boxplot = []
    labels = []

    for planet in planets:
        data_for_boxplot.append(distributions['libnova'][planet])
        labels.append(f"{planet[:3].capitalize()} - Lib")
        data_for_boxplot.append(distributions['siderust'][planet])
        labels.append(f"{planet[:3].capitalize()} - Sid")

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # boxplot => retornará un diagrama de cajas para cada elemento de data_for_boxplot
    bp = ax.boxplot(data_for_boxplot, patch_artist=True, showfliers=True)
    
    # Opcional: un poco de ajuste estético en las cajas
    # Por ejemplo, colorearlas de forma alternada
    for i, box in enumerate(bp['boxes']):
        if i % 2 == 0:
            box.set(facecolor="white")
        else:
            box.set(facecolor="lightgray")

    # Etiquetas en el eje X
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Error angular (arcosegundos)")
    ax.set_title("Distribución de Error Angular por Planeta: Libnova vs. Siderust")

    plt.tight_layout()
    plt.show()

def main():
    lab_root = os.environ["SIDERUST_LAB_ROOT"]
    astropy_path = os.path.join(lab_root, "astropy/dataset/")
    libnova_path = os.path.join(lab_root, "libnova/dataset/")
    siderust_path = os.path.join(lab_root, "siderust/dataset/")

    planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

    # 1) Aquí, compute_planet_metrics te da estadísticos agregados
    _, angular_results = compute_planet_metrics(planets, astropy_path, libnova_path, siderust_path)
    
    # 2) Supongamos que implementas algo parecido para obtener la distribución de errores
    distributions = compute_planet_distributions(planets, astropy_path, libnova_path, siderust_path)
    
    # 3) Dibujamos el boxplot
    boxplot_angular_distributions(distributions, planets)

if __name__ == "__main__":
    main()
