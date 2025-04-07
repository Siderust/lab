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

def calculate_angular_error_details(df_ref, df_lib, merge_on="jd"):
    """
    Une los datasets de referencia y de una librería, convierte las coordenadas cartesianas a angulares
    (longitud y latitud eclíptica) y calcula la separación angular en arcosegundos.
    """
    # Unir los datasets utilizando 'jd' como clave
    df = pd.merge(df_ref, df_lib, on=merge_on, suffixes=('_ref', '_lib'))
    
    # Calcular las distancias para cada conjunto (para evitar división por cero)
    df['r_ref'] = np.sqrt(df['x_ref']**2 + df['y_ref']**2 + df['z_ref']**2)
    df['r_lib'] = np.sqrt(df['x_lib']**2 + df['y_lib']**2 + df['z_lib']**2)
    
    # Convertir a coordenadas angulares: Longitud (λ) y latitud (β)
    # Usamos arctan2 para λ y arcsin(z/r) para β
    df['lambda_ref'] = np.arctan2(df['y_ref'], df['x_ref'])
    df['beta_ref']   = np.arcsin(df['z_ref'] / df['r_ref'])
    df['lambda_lib'] = np.arctan2(df['y_lib'], df['x_lib'])
    df['beta_lib']   = np.arcsin(df['z_lib'] / df['r_lib'])
    
    # Calcular la separación angular usando la fórmula del coseno esférico:
    # cos(Δ) = sin(β_ref) sin(β_lib) + cos(β_ref) cos(β_lib) cos(λ_ref - λ_lib)
    df['cos_sep'] = np.sin(df['beta_ref'])*np.sin(df['beta_lib']) + \
                    np.cos(df['beta_ref'])*np.cos(df['beta_lib'])*np.cos(df['lambda_ref'] - df['lambda_lib'])
    # Evitar posibles errores numéricos por valores fuera de [-1, 1]
    df['cos_sep'] = np.clip(df['cos_sep'], -1, 1)
    df['ang_sep_rad'] = np.arccos(df['cos_sep'])
    
    # Convertir la separación angular de radianes a arcosegundos:
    # 1 radian = (180/π)*3600 arcosegundos
    df['ang_sep_arcsec'] = df['ang_sep_rad'] * (180/np.pi) * 3600
    
    # Calcular el error medio angular y el RMSE de la separación angular (en arcosegundos)
    mean_ang_error = np.mean(df['ang_sep_arcsec'])
    rmse_ang_error = np.sqrt(np.mean(df['ang_sep_arcsec']**2))
    
    return {
        'mean_ang_error_arcsec': mean_ang_error,
        'rmse_ang_error_arcsec': rmse_ang_error
    }

# Definir rutas a los datasets
lab_root = os.environ["SIDERUST_LAB_ROOT"]
astropy_path = os.path.join(lab_root, "astropy/dataset/")
libnova_path = os.path.join(lab_root, "libnova/dataset/")
siderust_path = os.path.join(lab_root, "siderust/dataset/")

planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

# Cargar los datasets para cada planeta y calcular los errores
results = {"libnova": {}, "siderust": {}}
angular_results = {"libnova": {}, "siderust": {}}

for planet in planets:
    ref = pd.read_csv(os.path.join(astropy_path, f"{planet}_heliocentric_ecliptic.csv"))
    libnova = pd.read_csv(os.path.join(libnova_path, f"{planet}_heliocentric_ecliptic.csv"))
    siderust = pd.read_csv(os.path.join(siderust_path, f"{planet}_heliocentric_ecliptic.csv"))
    
    results["libnova"][planet] = calculate_rmse_details(ref, libnova)
    results["siderust"][planet] = calculate_rmse_details(ref, siderust)
    
    angular_results["libnova"][planet] = calculate_angular_error_details(ref, libnova)
    angular_results["siderust"][planet] = calculate_angular_error_details(ref, siderust)

# Imprimir resultados numéricos con 10 dígitos decimales
print("Comparación de RMSE por planeta (Total, X, Y, Z):")
for planet in planets:
    r_libnova = results["libnova"][planet]
    r_siderust = results["siderust"][planet]
    print(f"\n{planet.capitalize()}:")
    print(f"  Libnova - Total: {r_libnova['rmse_total']:.10f}, X: {r_libnova['rmse_x']:.10f}, Y: {r_libnova['rmse_y']:.10f}, Z: {r_libnova['rmse_z']:.10f}")
    print(f"  Siderust - Total: {r_siderust['rmse_total']:.10f}, X: {r_siderust['rmse_x']:.10f}, Y: {r_siderust['rmse_y']:.10f}, Z: {r_siderust['rmse_z']:.10f}")

print("\nComparación de Error Angular (arcosegundos) por planeta:")
for planet in planets:
    a_libnova = angular_results["libnova"][planet]
    a_siderust = angular_results["siderust"][planet]
    print(f"\n{planet.capitalize()}:")
    print(f"  Libnova - Error Medio: {a_libnova['mean_ang_error_arcsec']:.10f} arcsec, RMSE: {a_libnova['rmse_ang_error_arcsec']:.10f} arcsec")
    print(f"  Siderust - Error Medio: {a_siderust['mean_ang_error_arcsec']:.10f} arcsec, RMSE: {a_siderust['rmse_ang_error_arcsec']:.10f} arcsec")

# --------------------------------------------
# Ejemplo de gráfica para comparar el RMSE angular total por planeta
# --------------------------------------------
planets_cap = [p.capitalize() for p in planets]
x = np.arange(len(planets))
width = 0.35  # ancho de las barras

libnova_ang_rmse = [angular_results["libnova"][p]['rmse_ang_error_arcsec'] for p in planets]
siderust_ang_rmse = [angular_results["siderust"][p]['rmse_ang_error_arcsec'] for p in planets]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, libnova_ang_rmse, width, label="Libnova")
ax.bar(x + width/2, siderust_ang_rmse, width, label="Siderust")
ax.set_title("RMSE Angular (arcosegundos) por planeta")
ax.set_xticks(x)
ax.set_xticklabels(planets_cap)
ax.set_ylabel("RMSE Angular (arcosegundos)")
ax.legend()

plt.tight_layout()
plt.show()
