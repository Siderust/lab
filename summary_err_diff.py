import numpy as np
from utils import consts, out
from utils.metric_utils import compute_planet_metrics

def get_rec_gains(r_libnova, r_siderust):
    """Calculate rectangular RMSE gains between two sets of results."""
    total_improvement = r_libnova['rmse_total'] - r_siderust['rmse_total']
    x_improvement = r_libnova['rmse_x'] - r_siderust['rmse_x']
    y_improvement = r_libnova['rmse_y'] - r_siderust['rmse_y']
    z_improvement = r_libnova['rmse_z'] - r_siderust['rmse_z']
    return total_improvement, x_improvement, y_improvement, z_improvement

def get_sph_gains(a_libnova, a_siderust):
    """Calculate spherical angular RMSE gains between two sets of results."""
    mean_ang_error_arcsec = a_libnova['mean_ang_error_arcsec'] - a_siderust['mean_ang_error_arcsec']
    rmse_ang_error_arcsec = a_libnova['rmse_ang_error_arcsec'] - a_siderust['rmse_ang_error_arcsec']
    return mean_ang_error_arcsec, rmse_ang_error_arcsec

def calculate_global_average(results_dict, planets, keys):
    """
    Calculate the average values of the specified keys across all planets.
    """
    return {
        key: np.mean([results_dict[planet][key] for planet in planets])
        for key in keys
    }

def compute_gain_maps(planets, results, angular_results):
    """
    Compute both the rectangular and spherical gain maps.
    Processes per-planet gains as well as global averages.
    """
    rec_gain = {}
    sph_gain = {}
    
    # Calculate gains for each planet
    for planet in planets:
        rec_gain[planet] = get_rec_gains(results["libnova"][planet],
                                         results["siderust"][planet])
        sph_gain[planet] = get_sph_gains(angular_results["libnova"][planet],
                                         angular_results["siderust"][planet])
    
    # Calculate global average metrics for each library
    rec_avg_libnova = calculate_global_average(results["libnova"], planets,
                                               ['rmse_total', 'rmse_x', 'rmse_y', 'rmse_z'])
    rec_avg_siderust = calculate_global_average(results["siderust"], planets,
                                                ['rmse_total', 'rmse_x', 'rmse_y', 'rmse_z'])
    sph_avg_libnova = calculate_global_average(angular_results["libnova"], planets,
                                               ['mean_ang_error_arcsec', 'rmse_ang_error_arcsec'])
    sph_avg_siderust = calculate_global_average(angular_results["siderust"], planets,
                                                ['mean_ang_error_arcsec', 'rmse_ang_error_arcsec'])
    
    # Compute gains for global average results
    rec_gain["AVG"] = get_rec_gains(rec_avg_libnova, rec_avg_siderust)
    sph_gain["AVG"] = get_sph_gains(sph_avg_libnova, sph_avg_siderust)
    
    return rec_gain, sph_gain

def print_distance_to_astropy(planets, results, angular_results):
    """
    Print how far (in absolute terms) libnova and siderust are from astropy.
    """
    print("Distancia a Astropy (RMSE, Medidas Rectangulares en UA):")
    for planet in planets:
        r_lib = results["libnova"][planet]
        r_sid = results["siderust"][planet]
        print(f" {planet.capitalize()[:3]} - "
              f"Libnova => Total: {out.colorize(r_lib['rmse_total'])}, "
              f"X: {out.colorize(r_lib['rmse_x'])}, "
              f"Y: {out.colorize(r_lib['rmse_y'])}, "
              f"Z: {out.colorize(r_lib['rmse_z'])} | "
              f"Siderust => Total: {out.colorize(r_sid['rmse_total'])}, "
              f"X: {out.colorize(r_sid['rmse_x'])}, "
              f"Y: {out.colorize(r_sid['rmse_y'])}, "
              f"Z: {out.colorize(r_sid['rmse_z'])}")
    
    print("\nDistancia a Astropy (RMSE, Medidas Esféricas en ArcSec):")
    for planet in planets:
        a_lib = angular_results["libnova"][planet]
        a_sid = angular_results["siderust"][planet]
        print(f" {planet.capitalize()[:3]} - "
              f"Libnova => Mean: {out.colorize(a_lib['mean_ang_error_arcsec'])}, "
              f"RMSE: {out.colorize(a_lib['rmse_ang_error_arcsec'])} | "
              f"Siderust => Mean: {out.colorize(a_sid['mean_ang_error_arcsec'])}, "
              f"RMSE: {out.colorize(a_sid['rmse_ang_error_arcsec'])}")
    
def print_gain_results(rec_gain, sph_gain):
    """
    Print the computed gains in a formatted output.
    """
    print("Mejora en términos RMSE por planeta (Medidas Rectangulares en UA):")
    for planet, (total, x, y, z) in rec_gain.items():
        print(f" {planet.capitalize()[:3]} - "
              f"Total: {out.colorize(total)}, "
              f"X: {out.colorize(x)}, "
              f"Y: {out.colorize(y)}, "
              f"Z: {out.colorize(z)}")
    
    print("\nMejora en términos RMSE por planeta (Medidas Esféricas en ArcSec):")
    for planet, (mean, rmse) in sph_gain.items():
        print(f" {planet.capitalize()[:3]} - "
              f"Mean: {out.colorize(mean)}, "
              f"RMSE: {out.colorize(rmse)}")

def main():
    planets = consts.planets
    astropy_path = consts.astropy_path
    libnova_path = consts.libnova_path
    siderust_path = consts.siderust_path
    
    # Compute metrics for all planets
    results, angular_results = compute_planet_metrics(planets, astropy_path,
                                                      libnova_path, siderust_path)
    
    # Print absolute distances from astropy
    print_distance_to_astropy(planets, results, angular_results)
    print("\n" + "="*70 + "\n")
    
    # Compute gains (how much siderust improves over libnova)
    rec_gain, sph_gain = compute_gain_maps(planets, results, angular_results)
    
    # Print the gain results
    print_gain_results(rec_gain, sph_gain)

if __name__ == "__main__":
    main()
