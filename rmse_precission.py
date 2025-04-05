import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_rmse_details(df_ref, df_lib, merge_on="jd"):
    """
    Une los datasets de referencia y de una librería, calcula las diferencias en las coordenadas
    y retorna el RMSE total y por cada eje.
    """
    # Unir los datasets utilizando 'jd' como clave
    df = pd.merge(df_ref, df_lib, on=merge_on, suffixes=('_ref', '_lib'))
    
    # Calcular las diferencias en cada componente
    df['dx'] = df['x_lib'] - df['x_ref']
    df['dy'] = df['y_lib'] - df['y_ref']
    df['dz'] = df['z_lib'] - df['z_ref']
    
    # Calcular el RMSE para cada eje
    rmse_x = np.sqrt(np.mean(df['dx']**2))
    rmse_y = np.sqrt(np.mean(df['dy']**2))
    rmse_z = np.sqrt(np.mean(df['dz']**2))
    
    # Calcular el error Euclidiano total para cada instante
    df['error'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    rmse_total = np.sqrt(np.mean(df['error']**2))
    
    return {
        'rmse_total': rmse_total,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_z': rmse_z
    }

# Definir rutas a los datasets
lab_root = os.environ["SIDERUST_LAB_ROOT"]
astropy_path = os.path.join(lab_root, "astropy/dataset/")
libnova_path = os.path.join(lab_root, "libnova/dataset/")
siderust_path = os.path.join(lab_root, "siderust/dataset/")

planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

# Cargar los datasets para cada planeta y calcular RMSE
results = {"libnova": {}, "siderust": {}}

for planet in planets:
    ref = pd.read_csv(os.path.join(astropy_path, f"{planet}_heliocentric_ecliptic.csv"))
    libnova = pd.read_csv(os.path.join(libnova_path, f"{planet}_heliocentric_ecliptic.csv"))
    siderust = pd.read_csv(os.path.join(siderust_path, f"{planet}_heliocentric_ecliptic.csv"))
    
    results["libnova"][planet] = calculate_rmse_details(ref, libnova)
    results["siderust"][planet] = calculate_rmse_details(ref, siderust)

# Imprimir resultados por planeta con 10 dígitos decimales
print("Comparación de RMSE por planeta (Total, X, Y, Z):")
for planet in planets:
    r_libnova = results["libnova"][planet]
    r_siderust = results["siderust"][planet]
    print(f"\n{planet.capitalize()}:")
    print(f"  Libnova - Total: {r_libnova['rmse_total']:.10f}, X: {r_libnova['rmse_x']:.10f}, Y: {r_libnova['rmse_y']:.10f}, Z: {r_libnova['rmse_z']:.10f}")
    print(f"  Siderust - Total: {r_siderust['rmse_total']:.10f}, X: {r_siderust['rmse_x']:.10f}, Y: {r_siderust['rmse_y']:.10f}, Z: {r_siderust['rmse_z']:.10f}")

# Calcular el RMSE promedio global para cada librería
avg_libnova = {
    'rmse_total': np.mean([results["libnova"][planet]['rmse_total'] for planet in planets]),
    'rmse_x': np.mean([results["libnova"][planet]['rmse_x'] for planet in planets]),
    'rmse_y': np.mean([results["libnova"][planet]['rmse_y'] for planet in planets]),
    'rmse_z': np.mean([results["libnova"][planet]['rmse_z'] for planet in planets])
}

avg_siderust = {
    'rmse_total': np.mean([results["siderust"][planet]['rmse_total'] for planet in planets]),
    'rmse_x': np.mean([results["siderust"][planet]['rmse_x'] for planet in planets]),
    'rmse_y': np.mean([results["siderust"][planet]['rmse_y'] for planet in planets]),
    'rmse_z': np.mean([results["siderust"][planet]['rmse_z'] for planet in planets])
}

print("\nRMSE Promedio Global:")
print("Libnova:")
print(f"  Total: {avg_libnova['rmse_total']:.10f}, X: {avg_libnova['rmse_x']:.10f}, Y: {avg_libnova['rmse_y']:.10f}, Z: {avg_libnova['rmse_z']:.10f}")
print("Siderust:")
print(f"  Total: {avg_siderust['rmse_total']:.10f}, X: {avg_siderust['rmse_x']:.10f}, Y: {avg_siderust['rmse_y']:.10f}, Z: {avg_siderust['rmse_z']:.10f}")

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

# Crear figura con subgráficos
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
