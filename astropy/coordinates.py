import os
import numpy as np
import pandas as pd

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (
    get_body_barycentric,
    get_body,
    solar_system_ephemeris,
    SkyCoord,
    HeliocentricTrueEcliptic
)

# Root path for saving files
lab_root = os.environ.get("SIDERUST_LAB_ROOT", ".")

# List of solar system bodies
bodies = ['sun', 'moon', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

# Generate 6-hour time steps from 2025-01-01 to 2026-01-01
times_np = np.arange(
    np.datetime64('2025-01-01T00:00:00'),
    np.datetime64('2026-01-01T00:00:00'),
    np.timedelta64(6, 'h')
)
# Convert NumPy datetime64 array to an Astropy Time array
times = Time(times_np.astype(str), scale='utc')

# Use DE430 ephemeris (requires jplephem to be installed)
with solar_system_ephemeris.set('de430'):
    for body in bodies:
        data_icrs = []
        data_helio = []
        data_geo = []  # list to store geocentric-equatorial coordinates

        for t in times:
            jd = t.jd

            # Get barycentric positions of the body and the Sun
            barycentric_vector = get_body_barycentric(body, t)
            barycentric_vector = barycentric_vector.xyz.to(u.au)

            # Create the ICRS coordinate with proper length units
            planet_icrs = SkyCoord(
                barycentric_vector[0],
                barycentric_vector[1],
                barycentric_vector[2],
                frame='icrs',
                representation_type='cartesian',
                obstime=t
            )

            # Transform to heliocentric true ecliptic frame
            planet_ecl = planet_icrs.transform_to(HeliocentricTrueEcliptic(obstime=t))

            # Compute geocentric-equatorial coordinates using get_body.
            # This returns a SkyCoord in the GCRS frame, which is a standard geocentric equatorial frame.
            planet_geo = get_body(body, t)

            # Extract coordinates in AU
            x_icrs, y_icrs, z_icrs = planet_icrs.cartesian.xyz.to(u.AU).value
            x_ecl,  y_ecl,  z_ecl  = planet_ecl.cartesian.xyz.to(u.AU).value
            x_geo,  y_geo,  z_geo  = planet_geo.cartesian.xyz.to(u.AU).value

            data_icrs.append({
                'JD': f"{jd:.2f}",
                'X_AU': f"{x_icrs:.17f}",
                'Y_AU': f"{y_icrs:.17f}",
                'Z_AU': f"{z_icrs:.17f}"
            })

            if (body == 'sun'):
                data_helio.append({
                    'JD': f"{jd:.2f}",
                    'X_AU': f"{(0.0):.17f}",
                    'Y_AU': f"{(0.0):.17f}",
                    'Z_AU': f"{(0.0):.17f}"
                })
            else:
                data_helio.append({
                    'JD': f"{jd:.2f}",
                    'X_AU': f"{x_ecl:.17f}",
                    'Y_AU': f"{y_ecl:.17f}",
                    'Z_AU': f"{z_ecl:.17f}"
                })


            if (body == 'earth'):
                data_geo.append({
                    'JD': f"{jd:.2f}",
                    'X_AU': f"{(0.0):.17f}",
                    'Y_AU': f"{(0.0):.17f}",
                    'Z_AU': f"{(0.0):.17f}"
                })
            else:
                data_geo.append({
                    'JD': f"{jd:.2f}",
                    'X_AU': f"{x_geo:.17f}",
                    'Y_AU': f"{y_geo:.17f}",
                    'Z_AU': f"{z_geo:.17f}"
                })


        # Convert to DataFrames
        df_icrs = pd.DataFrame(data_icrs)
        df_helio = pd.DataFrame(data_helio)
        df_geo = pd.DataFrame(data_geo)

        # Define output directories
        icrs_path = os.path.join(lab_root, 'astropy', 'dataset', 'icrs')
        helio_path = os.path.join(lab_root, 'astropy', 'dataset', 'helio')
        geo_path   = os.path.join(lab_root, 'astropy', 'dataset', 'geo')

        os.makedirs(icrs_path, exist_ok=True)
        os.makedirs(helio_path, exist_ok=True)
        os.makedirs(geo_path, exist_ok=True)

        # Save CSV files
        df_icrs.to_csv(os.path.join(icrs_path, f'{body}_icrs.csv'), index=False)
        df_helio.to_csv(os.path.join(helio_path, f'{body}_helio.csv'), index=False)
        df_geo.to_csv(os.path.join(geo_path, f'{body}_geo.csv'), index=False)
