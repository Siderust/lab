#include <filesystem>
#include "utils.hpp"

int main()
{
    // Crear el directorio (si no existe)
    std::filesystem::create_directories(DATASET_ROOT);

    // Rango y paso
    double jd_start  = 2460676.5; // ~2025-01-01
    double jd_end    = 2461041.5; // ~2026-01-01
    double step_days = 0.25;

    // 1) Exportar heliocéntricas en coordenadas eclípticas (rectangulares)
    auto planets = GetPlanetRecHelioMap();
    for (const auto &[planet, func] : planets) {
        std::string filename = DATASET_ROOT + planet + "_heliocentric_ecliptic.csv";
        std::string header   = "jd,x,y,z";

        // Lambda que encapsula la llamada:
        auto calcRectPos = [=](double jd) {
            ln_rect_posn pos_equ;
            func(jd, &pos_equ); // llama al functor de libnova
            return pos_equ;
        };

        // Lambda para formatear la salida CSV:
        auto rectPosToCsv = [&](double jd, const ln_rect_posn &pos_equ) {
            // Convertir a eclípticas
            double x_ecl, y_ecl, z_ecl;
            equatorial_to_ecliptic(pos_equ.X, pos_equ.Y, pos_equ.Z,
                                   x_ecl, y_ecl, z_ecl);

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << jd << ","
                << std::fixed << std::setprecision(17)
                << x_ecl << ","
                << y_ecl << ","
                << z_ecl;
            return oss.str();
        };

        export_csv(filename, header, calcRectPos, rectPosToCsv,
                   jd_start, jd_end, step_days);
    }

    // 2) Exportar heliocéntricas esféricas
    auto planets_sph = GetPlanetSphHelioMap();
    for (const auto &[planet, func] : planets_sph) {
        std::string filename = DATASET_ROOT + planet + "_heliocentric_spherical.csv";
        std::string header   = "jd,lon,lat,dist";

        auto calcHelioSph = [=](double jd) {
            ln_helio_posn pos_sph;
            func(jd, &pos_sph);
            return pos_sph;
        };

        auto helioSphToCsv = [&](double jd, const ln_helio_posn &pos_sph) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << jd << ","
                << std::fixed << std::setprecision(17)
                << pos_sph.L << ","
                << pos_sph.B << ","
                << pos_sph.R;
            return oss.str();
        };

        export_csv(filename, header, calcHelioSph, helioSphToCsv,
                   jd_start, jd_end, step_days);
    }

    // 3) Exportar ecuatoriales esféricas
    auto planets_eq = GetPlanetSphEquMap();
    for (const auto &[planet, func] : planets_eq) {
        std::string filename = DATASET_ROOT + planet + "_equatorial_spherical.csv";
        std::string header   = "jd,ra,dec";

        auto calcEquSph = [=](double jd) {
            ln_equ_posn pos_eq;
            func(jd, &pos_eq);
            return pos_eq;
        };

        auto equSphToCsv = [&](double jd, const ln_equ_posn &pos_eq) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << jd << ","
                << std::fixed << std::setprecision(17)
                << pos_eq.ra << ","
                << pos_eq.dec; 
            return oss.str();
        };

        export_csv(filename, header, calcEquSph, equSphToCsv,
                   jd_start, jd_end, step_days);
    }

    return 0;
}
