import os
import numpy as np
import matplotlib.pyplot as plt
from utils.metric_utils import compute_planet_metrics

def plot_angular_rmse(angular_results, planets):
    """
    Plot the angular RMSE (arcseconds) per planet.
    
    Parameters:
        angular_results (dict): Dictionary with angular error details for each library and planet.
        planets (list): List of planet names.
    """
    # Capitalize planet names for display
    planets_cap = [p.capitalize() for p in planets]
    x = np.arange(len(planets))
    width = 0.35  # width of the bars

    # Extract RMSE angular error for each planet
    libnova_ang_rmse = [angular_results["libnova"][planet]['rmse_ang_error_arcsec'] for planet in planets]
    siderust_ang_rmse = [angular_results["siderust"][planet]['rmse_ang_error_arcsec'] for planet in planets]

    # Create the bar chart
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

def main():
    # Define the paths to the datasets using the environment variable.
    lab_root = os.environ["SIDERUST_LAB_ROOT"]
    astropy_path = os.path.join(lab_root, "astropy/dataset/")
    libnova_path = os.path.join(lab_root, "libnova/dataset/")
    siderust_path = os.path.join(lab_root, "siderust/dataset/")
    
    planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
    
    # Compute the metrics using the modularized function from the previous script.
    # This function returns both rectangular results and angular error details.
    _, angular_results = compute_planet_metrics(planets, astropy_path, libnova_path, siderust_path)
    
    # Plot the angular RMSE for each planet
    plot_angular_rmse(angular_results, planets)

if __name__ == '__main__':
    main()
