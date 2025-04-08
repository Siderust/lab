import pandas as pd
import numpy as np

def calculate_details(df_ref, df_lib, merge_on="jd"):
    """
    Une los datasets de referencia (df_ref) y de una librería (df_lib),
    calcula las diferencias en las coordenadas y retorna varias estadísticas:
    - RMSE total y por cada eje (x, y, z)
    - mediana del error total y por eje
    - desviación estándar del error total y por eje
    """
    # Unir los datasets usando 'jd' como clave
    df = pd.merge(df_ref, df_lib, on=merge_on, suffixes=('_ref', '_lib'))
    
    # Calcular diferencias en cada componente
    df['dx'] = df['x_lib'] - df['x_ref']
    df['dy'] = df['y_lib'] - df['y_ref']
    df['dz'] = df['z_lib'] - df['z_ref']
    
    # Calcular el error Euclidiano total para cada instante
    df['error'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    
    # RMSE por eje
    rmse_x = np.sqrt(np.mean(df['dx']**2))
    rmse_y = np.sqrt(np.mean(df['dy']**2))
    rmse_z = np.sqrt(np.mean(df['dz']**2))
    # RMSE total
    rmse_total = np.sqrt(np.mean(df['error']**2))
    
    # Mediana por eje
    median_x = np.median(df['dx'])
    median_y = np.median(df['dy'])
    median_z = np.median(df['dz'])
    # Mediana total
    median_total = np.median(df['error'])
    
    # Desviación estándar (ddof=1 para muestreo, puedes usar ddof=0 si lo prefieres)
    std_x = np.std(df['dx'], ddof=1)
    std_y = np.std(df['dy'], ddof=1)
    std_z = np.std(df['dz'], ddof=1)
    std_total = np.std(df['error'], ddof=1)
    
    return {
        'rmse_total': rmse_total,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_z': rmse_z,
        'median_total': median_total,
        'median_x': median_x,
        'median_y': median_y,
        'median_z': median_z,
        'std_total': std_total,
        'std_x': std_x,
        'std_y': std_y,
        'std_z': std_z
    }

def calculate_angular_error_details(df_ref, df_lib, merge_on="jd"):
    """
    Une los datasets de referencia (df_ref) y de una librería (df_lib),
    convierte las coordenadas cartesianas a ángulos eclípticos (λ, β) y
    calcula la separación angular respecto a la referencia (Astropy).
    
    Retorna varios estadísticos:
    - mean_ang_error_arcsec (media)
    - rmse_ang_error_arcsec
    - median_ang_error_arcsec
    - std_ang_error_arcsec
    """
    # Unir los datasets usando 'jd' como clave
    df = pd.merge(df_ref, df_lib, on=merge_on, suffixes=('_ref', '_lib'))
    
    # Distancia al origen para ref y librería
    df['r_ref'] = np.sqrt(df['x_ref']**2 + df['y_ref']**2 + df['z_ref']**2)
    df['r_lib'] = np.sqrt(df['x_lib']**2 + df['y_lib']**2 + df['z_lib']**2)
    
    # Convertir a coordenadas angulares (λ y β)
    df['lambda_ref'] = np.arctan2(df['y_ref'], df['x_ref'])
    df['beta_ref']   = np.arcsin(df['z_ref'] / df['r_ref'])
    df['lambda_lib'] = np.arctan2(df['y_lib'], df['x_lib'])
    df['beta_lib']   = np.arcsin(df['z_lib'] / df['r_lib'])
    
    # Fórmula del coseno esférico para la separación angular
    df['cos_sep'] = (np.sin(df['beta_ref']) * np.sin(df['beta_lib']) +
                     np.cos(df['beta_ref']) * np.cos(df['beta_lib']) *
                     np.cos(df['lambda_ref'] - df['lambda_lib']))
    
    # Ajustar posibles errores numéricos
    df['cos_sep'] = np.clip(df['cos_sep'], -1, 1)
    
    # Separación angular en radianes
    df['ang_sep_rad'] = np.arccos(df['cos_sep'])
    
    # Pasar a arcosegundos: 1 rad = (180/π) * 3600 arcsec
    df['ang_sep_arcsec'] = df['ang_sep_rad'] * (180/np.pi) * 3600
    
    # Calcular estadísticas
    mean_ang_error = np.mean(df['ang_sep_arcsec'])
    rmse_ang_error = np.sqrt(np.mean(df['ang_sep_arcsec']**2))
    median_ang_error = np.median(df['ang_sep_arcsec'])
    std_ang_error = np.std(df['ang_sep_arcsec'], ddof=1)
    
    return {
        'mean_ang_error_arcsec': mean_ang_error,
        'rmse_ang_error_arcsec': rmse_ang_error,
        'median_ang_error_arcsec': median_ang_error,
        'std_ang_error_arcsec': std_ang_error
    }
