#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <libnova/mercury.h>
#include <libnova/venus.h>
#include <libnova/earth.h>
#include <libnova/mars.h>
#include <libnova/jupiter.h>
#include <libnova/saturn.h>
#include <libnova/uranus.h>
#include <libnova/neptune.h>
#include <libnova/ln_types.h>

static const std::string DATASET_ROOT = "/home/user/src/libnova/dataset/";

struct Planet {
    std::string name;
    void (*get_helio_coords)(double, ln_rect_posn*);
};

/**
 * Convierte coordenadas heliocéntricas rectangulares *ecuatoriales*
 * en coordenadas *eclípticas*, aplicando la rotación por la oblicuidad (J2000).
 */
void equatorial_to_ecliptic(double x_eq, double y_eq, double z_eq,
                            double &x_ecl, double &y_ecl, double &z_ecl)
{
    constexpr double eps_deg = 23.4392911;              // Oblicuidad J2000
    const double eps_rad = eps_deg * M_PI / 180.0;

    x_ecl = x_eq;
    y_ecl = std::cos(eps_rad) * y_eq + std::sin(eps_rad) * z_eq;
    z_ecl = -std::sin(eps_rad) * y_eq + std::cos(eps_rad) * z_eq;
}

std::vector<Planet> GetPlanetCoordinatesMap() {
    return {
        { "mercury", ln_get_mercury_rect_helio },
        { "venus",   ln_get_venus_rect_helio   },
        { "earth",   ln_get_earth_rect_helio   },
        { "mars",    ln_get_mars_rect_helio    },
        { "jupiter", ln_get_jupiter_rect_helio },
        { "saturn",  ln_get_saturn_rect_helio  },
        { "uranus",  ln_get_uranus_rect_helio  },
        { "neptune", ln_get_neptune_rect_helio }
    };
}
