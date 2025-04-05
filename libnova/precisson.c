/******************************************************************************
 * planets.c
 *
 * Example code to generate heliocentric *ecliptic* coordinates for planets
 * using libnova. Each planet’s coordinates are saved to a CSV file:
 *   "mercury_helio.csv", "venus_helio.csv", etc.
 *
 * For more on libnova, see: https://libnova.sourceforge.net/
 ******************************************************************************/
#include <stdio.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include <libnova/mercury.h>
#include <libnova/venus.h>
#include <libnova/earth.h>
#include <libnova/mars.h>
#include <libnova/jupiter.h>
#include <libnova/saturn.h>
#include <libnova/uranus.h>
#include <libnova/neptune.h>
#include <libnova/pluto.h>
#include <libnova/ln_types.h>
//#include <libnova/ln_const.h>

/* A helper struct for each planet’s name and the libnova function
 * that returns heliocentric rectangular *equatorial* coords.
 * We'll convert them to heliocentric ecliptic afterwards.
 */
typedef struct {
    const char *name;
    void (*get_helio_coords)(double, struct ln_rect_posn *);
} PlanetFunc;

/* Rotation around X-axis by -ε (the negative of the obliquity),
 * converting Equatorial -> Ecliptic.
 * 
 * If (x_eq, y_eq, z_eq) are equatorial coordinates,
 * then (x_ecl, y_ecl, z_ecl) are ecliptic coordinates via:
 *
 *   x_ecl = x_eq
 *   y_ecl =  cos(ε) * y_eq + sin(ε) * z_eq
 *   z_ecl = -sin(ε) * y_eq + cos(ε) * z_eq
 */
static void equatorial_to_ecliptic(double x_eq, double y_eq, double z_eq,
                                   double *x_ecl, double *y_ecl, double *z_ecl)
{
    /* Obliquity of the ecliptic (J2000). 
       For more precise “of date” usage, 
       you could call ln_get_obliquity(jd, &eps_deg). */
    const double eps_deg = 23.4392911;
    const double eps_rad = eps_deg * M_PI / 180.0;

    *x_ecl = x_eq;
    *y_ecl = cos(eps_rad) * y_eq + sin(eps_rad) * z_eq;
    *z_ecl = -sin(eps_rad) * y_eq + cos(eps_rad) * z_eq;
}

/* 
 * Convert the planet's heliocentric rectangular *equatorial* coordinates 
 * to *ecliptic* coordinates and write them to a CSV for times from jd_start 
 * to jd_end in increments of step_days.
 */
void export_heliocentric_coords_ecliptic(
    const char *planet_name,
    void (*fn)(double, struct ln_rect_posn *),
    double jd_start,
    double jd_end,
    double step_days
) {
    char filename[1024];
    snprintf(filename, sizeof(filename), "/home/user/src/libnova/dataset/%s_heliocentric_ecliptic.csv", planet_name);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to create output file");
        return;
    }
    fprintf(fp, "jd,x,y,z\n");

    for (double jd = jd_start; jd <= jd_end + 1e-9; jd += step_days) {
        /* 1) Get heliocentric rectangular coords in the *equatorial* frame */
        struct ln_rect_posn pos_equ;
        fn(jd, &pos_equ);  /* pos_equ.X , pos_equ.Y, pos_equ.Z (AU) */

        /* 2) Rotate from equatorial -> ecliptic */
        double x_ecl, y_ecl, z_ecl;
        equatorial_to_ecliptic(
            pos_equ.X,
            pos_equ.Y,
            pos_equ.Z,
            &x_ecl,
            &y_ecl,
            &z_ecl
        );

        fprintf(fp, "%.9f,%.9f,%.9f,%.9f\n", jd, x_ecl, y_ecl, z_ecl);
    }

    fclose(fp);
    printf("Wrote %s\n", filename);
}

int create_dir_if_not_exists(const char *path) {
    struct stat st = {0};

    if (stat(path, &st) == -1) {
        // Directory does not exist
        if (mkdir(path, 0755) == 0) {
            return 0; // Success
        }
        else {
            perror("mkdir failed");
            return -1;
        }
    }
    return 0; // Already exists
}

int main(void) {

    create_dir_if_not_exists("/home/user/src/libnova/dataset/");

    /*
     * Approximate Julian Dates for:
     *   2025-01-01 ~ 2460676.5
     *   2026-01-01 ~ 2461041.5
     * Step of 0.25 days => 6 hours
     */
    double jd_start = 2460676.5;  /* ~ 2025-01-01 00:00 UTC */
    double jd_end   = 2461041.5;  /* ~ 2026-01-01 00:00 UTC */
    double step_days = 0.25;

    PlanetFunc planets[] = {
        { "mercury", ln_get_mercury_rect_helio },
        { "venus",   ln_get_venus_rect_helio   },
        { "earth",   ln_get_earth_rect_helio   },
        { "mars",    ln_get_mars_rect_helio    },
        { "jupiter", ln_get_jupiter_rect_helio },
        { "saturn",  ln_get_saturn_rect_helio  },
        { "uranus",  ln_get_uranus_rect_helio  },
        { "neptune", ln_get_neptune_rect_helio },
        /* Pluto if desired:
           { "pluto", ln_get_pluto_rect_helio },
        */
    };

    size_t num_planets = sizeof(planets) / sizeof(planets[0]);

    for (size_t i = 0; i < num_planets; i++) {
        export_heliocentric_coords_ecliptic(
            planets[i].name,
            planets[i].get_helio_coords,
            jd_start,
            jd_end,
            step_days
        );
    }

    return 0;
}
