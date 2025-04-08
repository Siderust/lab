import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import rmse as rmse_utils

def main():
    # Definir rutas a los datasets (requiere que la variable de entorno SIDERUST_LAB_ROOT esté configurada)
    lab_root = os.environ["SIDERUST_LAB_ROOT"]
    astropy_path = os.path.join(lab_root, "astropy/dataset/")
    libnova_path = os.path.join(lab_root, "libnova/dataset/")
    siderust_path = os.path.join(lab_root, "siderust/dataset/")

    # Lista de planetas a analizar
    planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

    # Cargar los datasets para cada planeta y calcular RMSE
    results = {"libnova": {}, "siderust": {}}

    for planet in planets:
        ref = pd.read_csv(os.path.join(astropy_path, f"{planet}_heliocentric_ecliptic.csv"))
        libnova = pd.read_csv(os.path.join(libnova_path, f"{planet}_heliocentric_ecliptic.csv"))
        siderust = pd.read_csv(os.path.join(siderust_path, f"{planet}_heliocentric_ecliptic.csv"))
        
        results["libnova"][planet] = rmse_utils.calculate_details(ref, libnova)
        results["siderust"][planet] = rmse_utils.calculate_details(ref, siderust)

    # --------------------------------------------
    # Generación de gráficos para visualizar los resultados
    # --------------------------------------------

    # Preparar los datos para la gráfica
    planets_cap = [p.capitalize() for p in planets]
    x = np.arange(len(planets))
    width = 0.35  # ancho de las barras

    # Extraer valores para RMSE total y para cada eje
    libnova_total = [results["libnova"][p]['rmse_total'] for p in planets]
    siderust_total = [results["siderust"][p]['rmse_total'] for p in planets]

    libnova_x = [results["libnova"][p]['rmse_x'] for p in planets]
    siderust_x = [results["siderust"][p]['rmse_x'] for p in planets]

    libnova_y = [results["libnova"][p]['rmse_y'] for p in planets]
    siderust_y = [results["siderust"][p]['rmse_y'] for p in planets]

    libnova_z = [results["libnova"][p]['rmse_z'] for p in planets]
    siderust_z = [results["siderust"][p]['rmse_z'] for p in planets]

    # Crear figura con 4 gráficos individuales (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Gráfico de RMSE Total
    axs[0, 0].bar(x - width/2, libnova_total, width, label="Libnova")
    axs[0, 0].bar(x + width/2, siderust_total, width, label="Siderust")
    axs[0, 0].set_title("RMSE Total por planeta")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(planets_cap)
    axs[0, 0].set_ylabel("RMSE Total")
    axs[0, 0].legend()

    # Gráfico de RMSE en eje X
    axs[0, 1].bar(x - width/2, libnova_x, width, label="Libnova")
    axs[0, 1].bar(x + width/2, siderust_x, width, label="Siderust")
    axs[0, 1].set_title("RMSE en eje X por planeta")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(planets_cap)
    axs[0, 1].set_ylabel("RMSE X")
    axs[0, 1].legend()

    # Gráfico de RMSE en eje Y
    axs[1, 0].bar(x - width/2, libnova_y, width, label="Libnova")
    axs[1, 0].bar(x + width/2, siderust_y, width, label="Siderust")
    axs[1, 0].set_title("RMSE en eje Y por planeta")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(planets_cap)
    axs[1, 0].set_ylabel("RMSE Y")
    axs[1, 0].legend()

    # Gráfico de RMSE en eje Z
    axs[1, 1].bar(x - width/2, libnova_z, width, label="Libnova")
    axs[1, 1].bar(x + width/2, siderust_z, width, label="Siderust")
    axs[1, 1].set_title("RMSE en eje Z por planeta")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(planets_cap)
    axs[1, 1].set_ylabel("RMSE Z")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


# Para permitir la ejecución directa de este script:
if __name__ == "__main__":
    main()
