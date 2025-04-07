#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
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

// Estructura para almacenar datos de cada planeta
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

/**
 * Genera un archivo CSV con las coordenadas heliocéntricas eclípticas
 * de un planeta, calculadas desde jd_start hasta jd_end en pasos de step_days.
 */
void export_heliocentric_coords_ecliptic(const Planet &planet,
                                         double jd_start,
                                         double jd_end,
                                         double step_days)
{
    // Nombre de archivo: "<DATASET_ROOT>/<planeta>_heliocentric_ecliptic.csv"
    const std::string filename = DATASET_ROOT + planet.name + "_heliocentric_ecliptic.csv";
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "No se pudo crear el archivo: " << filename << std::endl;
        return;
    }

    // Encabezado CSV
    ofs << "jd,x,y,z\n";

    for (double jd = jd_start; jd <= jd_end + 1e-9; jd += step_days) {
        // 1) Coordenadas heliocéntricas rectangulares *ecuatoriales*
        ln_rect_posn pos_equ;
        planet.get_helio_coords(jd, &pos_equ);

        // 2) Convertir a eclípticas
        double x_ecl, y_ecl, z_ecl;
        equatorial_to_ecliptic(pos_equ.X, pos_equ.Y, pos_equ.Z,
                               x_ecl, y_ecl, z_ecl);

        // Guardar en el CSV
        ofs << jd << ","
            << x_ecl << ","
            << y_ecl << ","
            << z_ecl << "\n";
    }

    std::cout << "Archivo generado: " << filename << std::endl;
}

int main()
{
    // Asegurarnos de que el directorio exista
    std::filesystem::create_directories(DATASET_ROOT);

    /*
     * Aproximaciones de fechas julianas:
     *   2025-01-01 ~ 2460676.5
     *   2026-01-01 ~ 2461041.5
     * Paso de 0.25 días => 6 horas
     */
    double jd_start  = 2460676.5;  // ~ 2025-01-01 00:00 UTC
    double jd_end    = 2461041.5;  // ~ 2026-01-01 00:00 UTC
    double step_days = 0.25;

    // Lista de planetas y función de libnova correspondiente
    std::vector<Planet> planets = {
        { "mercury", ln_get_mercury_rect_helio },
        { "venus",   ln_get_venus_rect_helio   },
        { "earth",   ln_get_earth_rect_helio   },
        { "mars",    ln_get_mars_rect_helio    },
        { "jupiter", ln_get_jupiter_rect_helio },
        { "saturn",  ln_get_saturn_rect_helio  },
        { "uranus",  ln_get_uranus_rect_helio  },
        { "neptune", ln_get_neptune_rect_helio }
    };

    // Generar CSV para cada planeta
    for (const auto &planet : planets) {
        export_heliocentric_coords_ecliptic(planet, jd_start, jd_end, step_days);
    }

    return 0;
}
