#!/usr/bin/env python3

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    get_body_barycentric,
    solar_system_ephemeris,
    HeliocentricMeanEcliptic,
    HeliocentricTrueEcliptic
)

def generate_heliocentric_ecliptic_csv(planet_name, start_jd, end_jd, step_days, filename):
    """
    For a given planet, generate heliocentric ecliptic coordinates (x, y, z)
    over JD times from start_jd to end_jd (inclusive) in increments of step_days,
    and write the results to filename as CSV.
    """
    times = np.arange(start_jd, end_jd + step_days, step_days)
    astropy_times = Time(times, format='jd')

    # Use a built-in ephemeris (can be changed to 'de441s' or others)
    with solar_system_ephemeris.set('de432s'):
        rows = []
        for t in astropy_times:

            # Get barycentric positions of planet and Sun
            planet_bary = get_body_barycentric(planet_name, t)

            # Convert that vector to an ICRS SkyCoord at time t
            planet_icrs_heliocentric = SkyCoord(planet_bary, frame='icrs', obstime=t)

            # Transform from ICRS to heliocentric ecliptic coordinates (J2000)
            planet_ecl = planet_icrs_heliocentric.transform_to(
                HeliocentricMeanEcliptic()
            )

            # Extract x, y, z in astronomical units (AU)
            x = planet_ecl.cartesian.x.to(u.AU).value
            y = planet_ecl.cartesian.y.to(u.AU).value
            z = planet_ecl.cartesian.z.to(u.AU).value

            rows.append((t.jd, x, y, z))

    # Write CSV
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("jd,x,y,z\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")


def main():
    """
    Generate heliocentric ecliptic coordinates (x, y, z) for each major planet
    from Jan 1, 2025 to Jan 1, 2026 in 6-hour increments, and store CSV files.
    """
    # Approximate Julian dates for 2025-01-01 and 2026-01-01
    start_jd = 2460676.5   # ~2025-Jan-01 00:00 UTC
    end_jd   = 2461041.5   # ~2026-Jan-01 00:00 UTC
    step_days = 0.25       # 6-hour increments

    # List of solar system bodies recognized by Astropy's get_body_barycentric
    # (use "earth" or "earth-moon-barycenter" as needed).
    planets = [
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
        # Uncomment if you want Pluto as well:
        # "pluto"
    ]

    for planet in planets:
        filename = f"{path}/{planet}_heliocentric_ecliptic.csv"
        generate_heliocentric_ecliptic_csv(
            planet, start_jd, end_jd, step_days, filename
        )
        print(f"Saved {filename} for planet: {planet}")

if __name__ == "__main__":
    import os
    path = "/home/user/src/astropy/dataset/"
    os.makedirs(path, exist_ok=True)
    main()
