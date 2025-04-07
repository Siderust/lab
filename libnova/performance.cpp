#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>

#include "utils.hpp"

/**
 * Genera un archivo CSV con las coordenadas heliocéntricas eclípticas
 * de un planeta, calculadas desde jd_start hasta jd_end en pasos de step_days.
 */
void compute_heliocentric_coords_ecliptic(const Planet &planet,
                                          double jd_start,
                                          double jd_end,
                                          double step_days)
{
    for (double jd = jd_start; jd <= jd_end + 1e-9; jd += step_days) {
        // 1) Coordenadas heliocéntricas rectangulares *ecuatoriales*
        ln_rect_posn pos_equ;
        planet.get_helio_coords(jd, &pos_equ);

        // 2) Convertir a eclípticas
        double x_ecl, y_ecl, z_ecl;
        equatorial_to_ecliptic(pos_equ.X, pos_equ.Y, pos_equ.Z,
                               x_ecl, y_ecl, z_ecl);
    }
}

int main()
{
    /*
     * Aproximaciones de fechas julianas:
     *   2025-01-01 ~ 2460676.5
     *   2026-01-01 ~ 2461041.5
     * Paso de 0.25 días => 6 horas
     */
    constexpr double jd_start  = 2460676.5;  // ~ 2025-01-01 00:00 UTC
    constexpr double jd_end    = 2461041.5;  // ~ 2026-01-01 00:00 UTC
    constexpr double step_days = 0.25;
    constexpr double n_steps = (jd_end - jd_start) / step_days;

    // Lista de planetas y función de libnova correspondiente
    std::vector<Planet> planets = GetPlanetCoordinatesMap();

    for (const auto &planet : planets)
    {
        auto start = std::chrono::high_resolution_clock::now();
        compute_heliocentric_coords_ecliptic(planet, jd_start, jd_end, step_days);
        auto end = std::chrono::high_resolution_clock::now();
    
        auto elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        std::cout << "Planet " << planet.name << " getHeliocentricEcliptic mean " << elapsed_us / n_steps << " µs.\n";
    }

    return 0;
}
