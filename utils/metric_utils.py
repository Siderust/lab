import pandas as pd
import numpy as np
import os
from utils import consts, out
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
