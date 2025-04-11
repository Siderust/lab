#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <map>
#include <filesystem>
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

/**
 * Convert heliocentric rectangular *equatorial* coordinates
 * to *ecliptic* coordinates by applying the J2000 obliquity rotation.
 */
void equatorial_to_ecliptic(double x_eq, double y_eq, double z_eq,
                            double &x_ecl, double &y_ecl, double &z_ecl)
{
    constexpr double eps_deg = 23.4392911; // J2000 obliquity
    const double eps_rad = eps_deg * M_PI / 180.0;

    x_ecl = x_eq;
    y_ecl = std::cos(eps_rad) * y_eq + std::sin(eps_rad) * z_eq;
    z_ecl = -std::sin(eps_rad) * y_eq + std::cos(eps_rad) * z_eq;
}

// Define type aliases for function pointer types that match the library functions.
using RectHelioFunc = void(*)(double, ln_rect_posn*);
using SphHelioFunc  = void(*)(double, ln_helio_posn*);
using EquCoordsFunc = void(*)(double, ln_equ_posn*);

// Map from planet name to rectangular heliocentric function.
std::map<std::string, RectHelioFunc> GetPlanetRecHelioMap()
{
    std::map<std::string, RectHelioFunc> m;
    m["mercury"] = ln_get_mercury_rect_helio;
    m["venus"]   = ln_get_venus_rect_helio;
    m["earth"]   = ln_get_earth_rect_helio;
    m["mars"]    = ln_get_mars_rect_helio;
    m["jupiter"] = ln_get_jupiter_rect_helio;
    m["saturn"]  = ln_get_saturn_rect_helio;
    m["uranus"]  = ln_get_uranus_rect_helio;
    m["neptune"] = ln_get_neptune_rect_helio;
    return m;
}

// Map from planet name to spherical heliocentric function.
std::map<std::string, SphHelioFunc> GetPlanetSphHelioMap()
{
    std::map<std::string, SphHelioFunc> m;
    m["mercury"] = ln_get_mercury_helio_coords;
    m["venus"]   = ln_get_venus_helio_coords;
    m["earth"]   = ln_get_earth_helio_coords;
    m["mars"]    = ln_get_mars_helio_coords;
    m["jupiter"] = ln_get_jupiter_helio_coords;
    m["saturn"]  = ln_get_saturn_helio_coords;
    m["uranus"]  = ln_get_uranus_helio_coords;
    m["neptune"] = ln_get_neptune_helio_coords;
    return m;
}

// Map from planet name to spherical equatorial function.
std::map<std::string, EquCoordsFunc> GetPlanetSphEquMap()
{
    std::map<std::string, EquCoordsFunc> m;
    m["mercury"] = ln_get_mercury_equ_coords;
    m["venus"]   = ln_get_venus_equ_coords;
    m["mars"]    = ln_get_mars_equ_coords;
    m["jupiter"] = ln_get_jupiter_equ_coords;
    m["saturn"]  = ln_get_saturn_equ_coords;
    m["uranus"]  = ln_get_uranus_equ_coords;
    m["neptune"] = ln_get_neptune_equ_coords;
    return m;
}


#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <iomanip>

// Esta función genérica recibe:
// - nombre de archivo
// - cabecera CSV
// - una lambda/functor que calcula la posición (planetFunc)
// - una lambda/functor que formatea la cadena de salida a partir de (jd, pos)
// - el rango de fechas y step
template <typename PlanetFunc, typename ToCsvString>
void export_csv(
    const std::string &filename,
    const std::string &csv_header,
    PlanetFunc planetFunc,
    ToCsvString toCsvString,
    double jd_start,
    double jd_end,
    double step_days)
{
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "No se pudo crear el archivo: " << filename << std::endl;
        return;
    }

    // Escribir la cabecera
    ofs << csv_header << "\n";

    // Iterar en el rango de fechas julianas
    for (double jd = jd_start; jd <= jd_end + 1e-9; jd += step_days) {
        // Llamada al functor que devuelve la posición (o la escribe en un struct).
        //   planetFunc: double jd -> lo que necesites (un struct, un objeto, etc.)
        auto pos = planetFunc(jd);

        // Llamada al functor que formatea la salida a CSV.
        ofs << toCsvString(jd, pos) << "\n";
    }

    std::cout << "Archivo generado: " << filename << std::endl;
}
