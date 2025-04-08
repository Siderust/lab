import pandas as pd
import numpy as np
import os
from utils.rmse import calculate_angular_error_details, calculate_details

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def get_file_path(base_path, planet):
    """Construct the file path for a planet's dataset."""
    file_name = f"{planet}_heliocentric_ecliptic.csv"
    return os.path.join(base_path, file_name)


def process_planet(planet, astropy_path, libnova_path, siderust_path):
    """
    Process a single planet:
    - Load reference, libnova, and siderust datasets.
    - Calculate rectangular and angular error details.
    """
    # Load reference data
    ref_file = get_file_path(astropy_path, planet)
    ref = load_csv(ref_file)
    
    # Load libnova and siderust datasets
    libnova_file = get_file_path(libnova_path, planet)
    siderust_file = get_file_path(siderust_path, planet)
    libnova_data = load_csv(libnova_file)
    siderust_data = load_csv(siderust_file)
    
    # Calculate details for rectangular errors
    rec_libnova = calculate_details(ref, libnova_data)
    rec_siderust = calculate_details(ref, siderust_data)
    
    # Calculate details for angular errors
    angular_libnova = calculate_angular_error_details(ref, libnova_data)
    angular_siderust = calculate_angular_error_details(ref, siderust_data)
    
    return rec_libnova, rec_siderust, angular_libnova, angular_siderust


def compute_planet_metrics(planets, astropy_path, libnova_path, siderust_path):
    """
    Compute RMSE and angular error details for all planets.
    Returns dictionaries with the results for libnova and siderust.
    """
    results = {"libnova": {}, "siderust": {}}
    angular_results = {"libnova": {}, "siderust": {}}
    
    for planet in planets:
        rec_libnova, rec_siderust, angular_libnova, angular_siderust = process_planet(
            planet, astropy_path, libnova_path, siderust_path
        )
        results["libnova"][planet] = rec_libnova
        results["siderust"][planet] = rec_siderust
        angular_results["libnova"][planet] = angular_libnova
        angular_results["siderust"][planet] = angular_siderust

    return results, angular_results

def calculate_angular_error_distribution(df_ref, df_lib, merge_on="jd"):
    """
    Similar a calculate_angular_error_details, pero retorna la lista
    (array) de errores angulares en arcsegundos para cada fila,
    en lugar de calcular estadísticos agregados.
    """
    df = pd.merge(df_ref, df_lib, on=merge_on, suffixes=('_ref', '_lib'))
    
    # Calculamos las distancias
    df['r_ref'] = np.sqrt(df['x_ref']**2 + df['y_ref']**2 + df['z_ref']**2)
    df['r_lib'] = np.sqrt(df['x_lib']**2 + df['y_lib']**2 + df['z_lib']**2)
    
    # Coordenadas eclípticas (arctan2 y arcsin)
    df['lambda_ref'] = np.arctan2(df['y_ref'], df['x_ref'])
    df['beta_ref']   = np.arcsin(df['z_ref'] / df['r_ref'])
    df['lambda_lib'] = np.arctan2(df['y_lib'], df['x_lib'])
    df['beta_lib']   = np.arcsin(df['z_lib'] / df['r_lib'])
    
    # Coseno de la separación (fórmula de la esfera)
    df['cos_sep'] = (np.sin(df['beta_ref']) * np.sin(df['beta_lib']) +
                     np.cos(df['beta_ref']) * np.cos(df['beta_lib']) *
                     np.cos(df['lambda_ref'] - df['lambda_lib']))
    df['cos_sep'] = np.clip(df['cos_sep'], -1, 1)
    
    # Separación angular en radianes
    df['ang_sep_rad'] = np.arccos(df['cos_sep'])
    
    # Pasar a arcosegundos
    df['ang_sep_arcsec'] = df['ang_sep_rad'] * (180/np.pi) * 3600
    
    # Retornamos la lista (array) de errores angulares para cada fila
    return df['ang_sep_arcsec'].values


def process_planet_distributions(planet, astropy_path, libnova_path, siderust_path):
    """
    Para un planeta específico, carga ref + libnova + siderust, y obtiene
    los arrays con la distribución de errores angulares (cada muestra).
    """
    # Archivos
    ref_file = get_file_path(astropy_path, planet)      # p. ej. "path/astropy/mercury_heliocentric_ecliptic.csv"
    libnova_file = get_file_path(libnova_path, planet)  
    siderust_file = get_file_path(siderust_path, planet)

    # Cargar DataFrames
    ref_df = load_csv(ref_file)
    libnova_df = load_csv(libnova_file)
    siderust_df = load_csv(siderust_file)
    
    # Calcular la distribución de errores (array) para cada librería
    libnova_errors = calculate_angular_error_distribution(ref_df, libnova_df)
    siderust_errors = calculate_angular_error_distribution(ref_df, siderust_df)
    
    return libnova_errors, siderust_errors


def compute_planet_distributions(planets, astropy_path, libnova_path, siderust_path):
    """
    Devuelve un diccionario con las distribuciones de error angular en arcsegundos
    para Libnova y Siderust, por cada planeta.
    
    Estructura de salida:
        {
          "libnova": {
             "mercury": [array de errores],
             "venus":   [array de errores],
             ...
          },
          "siderust": {
             "mercury": [array de errores],
             ...
          }
        }
    """
    distributions = {"libnova": {}, "siderust": {}}
    
    for planet in planets:
        libnova_errors, siderust_errors = process_planet_distributions(
            planet, astropy_path, libnova_path, siderust_path
        )
        
        # Almacenar en el diccionario final
        distributions["libnova"][planet] = libnova_errors
        distributions["siderust"][planet] = siderust_errors

    return distributions
