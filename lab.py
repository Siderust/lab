import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_planet_helio(planet_name):
    astropy_data = "./astropy/dataset/"
    star_track_data = "./star-track/scripts/dataset/"
    libnova_data = "./libnova/dataset/"

    astropy_df    = pd.read_csv(astropy_data    + planet_name + "_heliocentric_ecliptic.csv")
    star_track_df = pd.read_csv(star_track_data + planet_name + "_heliocentric_ecliptic.csv")
    libnova_df    = pd.read_csv(libnova_data    + planet_name + "_heliocentric_ecliptic.csv")

    return (astropy_df, star_track_df, libnova_df)


def compute_distances(df, ref):
    x, y, z = df['x'], df['y'], df['z']
    distancias = np.sqrt((x - ref[0])**2 + (y - ref[1])**2 + (z - ref[2])**2)
    return distancias


def main():
    planet = "mercury"  # cambia esto si quieres otro planeta
    astropy_df, star_track_df, libnova_df = load_planet_helio(planet)

    # Usamos Astropy como punto de referencia
    punto_ref = (astropy_df['x'], astropy_df['y'], astropy_df['z'])

    # Calcular distancias punto a punto (requiere que los JDs estén alineados)
    dist_star_track = compute_distances(star_track_df, punto_ref)
    dist_libnova    = compute_distances(libnova_df, punto_ref)

    # Graficar
    jd = astropy_df['jd']
    plt.figure(figsize=(10, 6))
    plt.plot(jd, dist_star_track, label="Star-Track vs Astropy")
    plt.plot(jd, dist_libnova, label="Libnova vs Astropy")

    plt.xlabel("Julian Date (JD)")
    plt.ylabel("Distancia a Astropy (u.a.)")
    plt.title(f"Comparación de distancias heliocéntricas - {planet.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
