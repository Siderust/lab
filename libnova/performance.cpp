#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>

#include "utils.hpp"

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
    auto planets = GetPlanetRecHelioMap();

    for (const auto &[planet, func] : planets)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (double jd = jd_start; jd <= jd_end + 1e-9; jd += step_days) {
            ln_rect_posn pos_equ;
            func(jd, &pos_equ);
        }

        auto end = std::chrono::high_resolution_clock::now();
    
        auto elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        std::cout << "Planet " << planet << " getHeliocentricEcliptic mean " << elapsed_us / n_steps << " µs.\n";
    }

    return 0;
}
