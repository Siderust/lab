#define _POSIX_C_SOURCE 199309L

/*
 * libnova Adapter for the Siderust Lab
 *
 * Reads line-protocol from stdin, runs libnova transformations,
 * writes JSON results to stdout.
 *
 * Protocol (one experiment per invocation):
 *   Line 1: experiment name
 *   Line 2: N (number of test cases)
 *   Lines 3..N+2: space-separated values depending on experiment
 *
 * Output: JSON to stdout.
 *
 * Supported experiments:
 *   frame_rotation_bpn  — Precession + Nutation (Meeus / IAU 1980)
 *     Input per line:  jd_tt  vx vy vz
 *     Output: transformed direction (matrix is null — libnova has no BPN matrix)
 *
 *   gmst_era — Greenwich Mean Sidereal Time (no ERA — libnova has no ERA)
 *     Input per line:  jd_ut1  jd_tt
 *     Output: GMST (rad), GAST (rad)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <libnova/precession.h>
#include <libnova/nutation.h>
#include <libnova/sidereal_time.h>
#include <libnova/ln_types.h>
#include <libnova/utility.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static void normalize3(double v[3]) {
    double r = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (r > 0.0) { v[0] /= r; v[1] /= r; v[2] /= r; }
}

/* Convert Cartesian unit vector to RA/Dec in degrees */
static void cart_to_radec(const double v[3], double *ra_deg, double *dec_deg) {
    *dec_deg = asin(v[2]) * (180.0 / M_PI);
    *ra_deg  = atan2(v[1], v[0]) * (180.0 / M_PI);
    if (*ra_deg < 0.0) *ra_deg += 360.0;
}

/* Convert RA/Dec in degrees to Cartesian unit vector */
static void radec_to_cart(double ra_deg, double dec_deg, double v[3]) {
    double ra_r  = ra_deg  * (M_PI / 180.0);
    double dec_r = dec_deg * (M_PI / 180.0);
    v[0] = cos(dec_r) * cos(ra_r);
    v[1] = cos(dec_r) * sin(ra_r);
    v[2] = sin(dec_r);
}

/* Angular separation between two unit vectors (radians) */
static double ang_sep(const double a[3], const double b[3]) {
    double dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    if (dot >  1.0) dot =  1.0;
    if (dot < -1.0) dot = -1.0;
    return acos(dot);
}

/* ------------------------------------------------------------------ */
/* Forward transform: J2000 → epoch (precession then nutation)         */
/* ------------------------------------------------------------------ */

static void j2000_to_epoch(double jd_tt, const double vin[3], double vout[3]) {
    double ra_deg, dec_deg;
    cart_to_radec(vin, &ra_deg, &dec_deg);

    /* 1. Precession: J2000 → epoch */
    struct ln_equ_posn mean_pos = { .ra = ra_deg, .dec = dec_deg };
    struct ln_equ_posn prec_pos;
    ln_get_equ_prec(&mean_pos, jd_tt, &prec_pos);

    /* 2. Nutation: apply nutation correction */
    struct ln_equ_posn nut_pos;
    ln_get_equ_nut(&prec_pos, jd_tt, &nut_pos);

    radec_to_cart(nut_pos.ra, nut_pos.dec, vout);
    normalize3(vout);
}

/* ------------------------------------------------------------------ */
/* Reverse transform: epoch → J2000 (undo nutation then precession)    */
/* ------------------------------------------------------------------ */

static void epoch_to_j2000(double jd_tt, const double vin[3], double vout[3]) {
    double ra_deg, dec_deg;
    cart_to_radec(vin, &ra_deg, &dec_deg);

    /*
     * Undo nutation: ln_get_equ_nut applies additive RA/Dec corrections
     * using the nutation parameters. To reverse, we compute the same
     * corrections and subtract them.
     */
    struct ln_nutation nut;
    ln_get_nutation(jd_tt, &nut);

    double ra_r  = ra_deg * (M_PI / 180.0);
    double dec_r = dec_deg * (M_PI / 180.0);

    double nut_ecliptic = (nut.ecliptic + nut.obliquity) * (M_PI / 180.0);
    double sin_ecliptic = sin(nut_ecliptic);
    double sin_ra = sin(ra_r);
    double cos_ra = cos(ra_r);
    double tan_dec = tan(dec_r);

    /* Same formula as ln_get_equ_nut (Meeus Equ 22.1), but subtracted */
    double delta_ra = (cos(nut_ecliptic) + sin_ecliptic * sin_ra * tan_dec) * nut.longitude
                    - cos_ra * tan_dec * nut.obliquity;
    double delta_dec = (sin_ecliptic * cos_ra) * nut.longitude
                     + sin_ra * nut.obliquity;

    double unnut_ra  = ra_deg  - delta_ra;
    double unnut_dec = dec_deg - delta_dec;

    /* Undo precession: epoch → J2000 */
    struct ln_equ_posn epoch_pos = { .ra = unnut_ra, .dec = unnut_dec };
    struct ln_equ_posn j2000_pos;
    ln_get_equ_prec2(&epoch_pos, jd_tt, JD2000, &j2000_pos);

    radec_to_cart(j2000_pos.ra, j2000_pos.dec, vout);
    normalize3(vout);
}

/* ------------------------------------------------------------------ */
/* Experiment: frame_rotation_bpn                                      */
/* Applies Meeus precession + IAU 1980 nutation via libnova's RA/Dec   */
/* coordinate-level API. No rotation matrix is available.              */
/* ------------------------------------------------------------------ */

static void run_frame_rotation_bpn(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"frame_rotation_bpn\",\"library\":\"libnova\",");
    printf("\"model\":\"Meeus_prec_IAU1980_nut\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }

        double vin[3] = {vx, vy, vz};
        normalize3(vin);

        /* Forward: J2000 → epoch */
        double vout[3];
        j2000_to_epoch(jd_tt, vin, vout);

        /* Closure: epoch → J2000 */
        double vinv[3];
        epoch_to_j2000(jd_tt, vout, vinv);
        double closure_rad = ang_sep(vin, vinv);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],", vout[0], vout[1], vout[2]);
        printf("\"closure_rad\":%.17e,", closure_rad);
        printf("\"matrix\":null}");
    }

    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: gmst_era                                                */
/* Computes GMST (Meeus Formula 11.4) and GAST.                       */
/* libnova has no ERA function — era_rad is omitted.                   */
/* ------------------------------------------------------------------ */

static void run_gmst_era(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"gmst_era\",\"library\":\"libnova\",");
    printf("\"model\":\"GMST=Meeus_11.4, GAST=MST+nutation\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_ut1, jd_tt;
        if (scanf("%lf %lf", &jd_ut1, &jd_tt) != 2) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }

        /* ln_get_mean_sidereal_time returns hours */
        double mst_hours  = ln_get_mean_sidereal_time(jd_ut1);
        double gast_hours = ln_get_apparent_sidereal_time(jd_ut1);

        /* Convert hours → radians */
        double gmst_rad = mst_hours  * (M_PI / 12.0);
        double gast_rad = gast_hours * (M_PI / 12.0);

        if (i > 0) printf(",\n");
        printf("{\"jd_ut1\":%.15f,\"jd_tt\":%.15f,", jd_ut1, jd_tt);
        printf("\"gmst_rad\":%.17e,\"gast_rad\":%.17e}", gmst_rad, gast_rad);
    }

    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Performance timing                                                  */
/* ------------------------------------------------------------------ */

static void run_frame_rotation_bpn_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    /* Read all inputs */
    double *jds  = malloc(n * sizeof(double));
    double *vecs = malloc(n * 3 * sizeof(double));
    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }
        jds[i] = jd_tt;
        vecs[3*i+0] = vx; vecs[3*i+1] = vy; vecs[3*i+2] = vz;
    }

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double vin[3] = {vecs[3*i], vecs[3*i+1], vecs[3*i+2]};
        normalize3(vin);
        double vout[3];
        j2000_to_epoch(jds[i], vin, vout);
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double vout[3];
    for (int i = 0; i < n; i++) {
        double vin[3] = {vecs[3*i], vecs[3*i+1], vecs[3*i+2]};
        normalize3(vin);
        j2000_to_epoch(jds[i], vin, vout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"frame_rotation_bpn_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,", (double)n / (elapsed_ns * 1e-9));
    /* Dummy use of vout to prevent optimization */
    printf("\"_sink\":%.17e}\n", vout[0]);

    free(jds);
    free(vecs);
}

/* ------------------------------------------------------------------ */
/* Main dispatcher                                                     */
/* ------------------------------------------------------------------ */

int main(void) {
    char experiment[256];
    if (scanf("%255s", experiment) != 1) {
        fprintf(stderr, "Usage: echo 'experiment_name\\nN\\n...' | ./libnova_adapter\n");
        return 1;
    }

    if (strcmp(experiment, "frame_rotation_bpn") == 0) {
        run_frame_rotation_bpn();
    } else if (strcmp(experiment, "gmst_era") == 0) {
        run_gmst_era();
    } else if (strcmp(experiment, "frame_rotation_bpn_perf") == 0) {
        run_frame_rotation_bpn_perf();
    } else {
        fprintf(stderr, "Unknown experiment: %s\n", experiment);
        return 1;
    }

    return 0;
}
