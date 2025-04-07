#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip> // Asegúrate de incluir este header

#include "utils.hpp"

void export_heliocentric_coords_ecliptic(const Planet &planet,
                                         double jd_start,
                                         double jd_end,
                                         double step_days)
{
    const std::string filename = DATASET_ROOT + planet.name + "_heliocentric_ecliptic.csv";
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "No se pudo crear el archivo: " << filename << std::endl;
        return;
    }

    // Encabezado CSV
    ofs << "jd,x,y,z\n";

    for (double jd = jd_start; jd <= jd_end + 1e-9; jd += step_days) {
        ln_rect_posn pos_equ;
        planet.get_helio_coords(jd, &pos_equ);

        double x_ecl, y_ecl, z_ecl;
        equatorial_to_ecliptic(pos_equ.X, pos_equ.Y, pos_equ.Z,
                               x_ecl, y_ecl, z_ecl);

        // jd con precisión de 2 decimales
        ofs << std::fixed << std::setprecision(2) << jd << ",";
        // Coordenadas con mayor precisión, por ejemplo, 17 decimales
        ofs << std::fixed << std::setprecision(17)
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
    std::vector<Planet> planets = GetPlanetCoordinatesMap();

    // Generar CSV para cada planeta
    for (const auto &planet : planets) {
        export_heliocentric_coords_ecliptic(planet, jd_start, jd_end, step_days);
    }

    return 0;
}
