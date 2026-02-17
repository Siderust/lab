#define _POSIX_C_SOURCE 199309L

/*
 * libnova Adapter for the Siderust Lab
 *
 * Supported experiments:
 *   frame_rotation_bpn  — Precession + Nutation (Meeus / IAU 1980)
 *   gmst_era            — GMST (Meeus 11.4) & GAST
 *   equ_ecl             — Equatorial ↔ Ecliptic
 *   equ_horizontal      — Equatorial → Horizontal (AltAz)
 *   solar_position      — Sun geocentric RA/Dec (VSOP87)
 *   lunar_position      — Moon geocentric RA/Dec (ELP 2000-82B)
 *   kepler_solver       — Kepler equation (Sinnott bisection)
 *   frame_rotation_bpn_perf — BPN performance timing
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
#include <libnova/transform.h>
#include <libnova/solar.h>
#include <libnova/lunar.h>
#include <libnova/elliptic_motion.h>

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
/* Experiment: equ_ecl                                                 */
/* Equatorial ↔ Ecliptic via libnova transform API                     */
/* libnova uses degrees: RA in hours→deg, Dec in deg, lon/lat in deg   */
/* Input per line: jd_tt  ra_rad  dec_rad                              */
/* ------------------------------------------------------------------ */

static void run_equ_ecl(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"equ_ecl\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_transform\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt, ra_rad, dec_rad;
        if (scanf("%lf %lf %lf", &jd_tt, &ra_rad, &dec_rad) != 3) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }

        /* Convert input radians → libnova degrees */
        /* libnova equ_posn.ra is in degrees (0..360), dec in degrees */
        double ra_deg  = ra_rad  * (180.0 / M_PI);
        double dec_deg = dec_rad * (180.0 / M_PI);

        struct ln_equ_posn equ = { .ra = ra_deg, .dec = dec_deg };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, jd_tt, &ecl);

        double ecl_lon = ecl.lng * (M_PI / 180.0);
        double ecl_lat = ecl.lat * (M_PI / 180.0);

        /* Closure: ecliptic → equatorial */
        struct ln_lnlat_posn ecl2 = { .lng = ecl.lng, .lat = ecl.lat };
        struct ln_equ_posn equ_back;
        ln_get_equ_from_ecl(&ecl2, jd_tt, &equ_back);

        double ra_back  = equ_back.ra  * (M_PI / 180.0);
        double dec_back = equ_back.dec * (M_PI / 180.0);

        double v_in[3]  = { cos(dec_rad)*cos(ra_rad), cos(dec_rad)*sin(ra_rad), sin(dec_rad) };
        double v_bk[3]  = { cos(dec_back)*cos(ra_back), cos(dec_back)*sin(ra_back), sin(dec_back) };
        double closure_rad = ang_sep(v_in, v_bk);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"ra_rad\":%.17e,\"dec_rad\":%.17e,", jd_tt, ra_rad, dec_rad);
        printf("\"ecl_lon_rad\":%.17e,\"ecl_lat_rad\":%.17e,", ecl_lon, ecl_lat);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: equ_horizontal                                          */
/* Equatorial → Horizontal via libnova                                 */
/* libnova azimuth: 0°=South, increasing westward → convert to         */
/* 0°=North, increasing eastward to match ERFA convention.             */
/* Input per line: jd_ut1 jd_tt ra_rad dec_rad obs_lon_rad obs_lat_rad */
/* ------------------------------------------------------------------ */

static void run_equ_horizontal(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"equ_horizontal\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_transform\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_ut1, jd_tt, ra_rad, dec_rad, obs_lon, obs_lat;
        if (scanf("%lf %lf %lf %lf %lf %lf", &jd_ut1, &jd_tt, &ra_rad, &dec_rad,
                  &obs_lon, &obs_lat) != 6) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }

        double ra_deg  = ra_rad  * (180.0 / M_PI);
        double dec_deg = dec_rad * (180.0 / M_PI);
        double lon_deg = obs_lon * (180.0 / M_PI);
        double lat_deg = obs_lat * (180.0 / M_PI);

        struct ln_equ_posn equ = { .ra = ra_deg, .dec = dec_deg };
        struct ln_lnlat_posn observer = { .lng = lon_deg, .lat = lat_deg };
        struct ln_hrz_posn hrz;
        ln_get_hrz_from_equ(&equ, &observer, jd_ut1, &hrz);

        /* libnova: az 0=South, increasing toward West (CW from above).
         * ERFA:    az 0=North, increasing toward East (CW from above).
         * Convert: az_erfa = (az_libnova + 180°) mod 360° */
        double az_erfa_deg = fmod(hrz.az + 180.0, 360.0);
        double az_rad = az_erfa_deg * (M_PI / 180.0);
        double alt_rad = hrz.alt * (M_PI / 180.0);

        /* Closure via reverse */
        struct ln_hrz_posn hrz2 = { .az = hrz.az, .alt = hrz.alt };
        struct ln_equ_posn equ_back;
        ln_get_equ_from_hrz(&hrz2, &observer, jd_ut1, &equ_back);

        double ra_back  = equ_back.ra  * (M_PI / 180.0);
        double dec_back = equ_back.dec * (M_PI / 180.0);
        double v_in[3]  = { cos(dec_rad)*cos(ra_rad), cos(dec_rad)*sin(ra_rad), sin(dec_rad) };
        double v_bk[3]  = { cos(dec_back)*cos(ra_back), cos(dec_back)*sin(ra_back), sin(dec_back) };
        double closure_rad = ang_sep(v_in, v_bk);

        if (i > 0) printf(",\n");
        printf("{\"jd_ut1\":%.15f,\"jd_tt\":%.15f,", jd_ut1, jd_tt);
        printf("\"ra_rad\":%.17e,\"dec_rad\":%.17e,", ra_rad, dec_rad);
        printf("\"obs_lon_rad\":%.17e,\"obs_lat_rad\":%.17e,", obs_lon, obs_lat);
        printf("\"az_rad\":%.17e,\"alt_rad\":%.17e,", az_rad, alt_rad);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: solar_position                                          */
/* Sun geocentric RA/Dec via libnova VSOP87                            */
/* Input per line: jd_tt                                               */
/* ------------------------------------------------------------------ */

static void run_solar_position(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"solar_position\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_VSOP87\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt;
        if (scanf("%lf", &jd_tt) != 1) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }

        struct ln_equ_posn posn;
        ln_get_solar_equ_coords(jd_tt, &posn);

        /* libnova: ra in degrees (0..360), dec in degrees */
        double ra_rad  = posn.ra  * (M_PI / 180.0);
        double dec_rad = posn.dec * (M_PI / 180.0);

        /* Get distance */
        double dist_au = ln_get_earth_solar_dist(jd_tt);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,", jd_tt);
        printf("\"ra_rad\":%.17e,\"dec_rad\":%.17e,\"dist_au\":%.17e}", ra_rad, dec_rad, dist_au);
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: lunar_position                                          */
/* Moon geocentric RA/Dec via libnova ELP 2000-82B                     */
/* Input per line: jd_tt                                               */
/* ------------------------------------------------------------------ */

static void run_lunar_position(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"lunar_position\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_ELP2000\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt;
        if (scanf("%lf", &jd_tt) != 1) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }

        struct ln_equ_posn posn;
        ln_get_lunar_equ_coords(jd_tt, &posn);

        double ra_rad  = posn.ra  * (M_PI / 180.0);
        double dec_rad = posn.dec * (M_PI / 180.0);

        double dist_km = ln_get_lunar_earth_dist(jd_tt);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,", jd_tt);
        printf("\"ra_rad\":%.17e,\"dec_rad\":%.17e,\"dist_km\":%.17e}", ra_rad, dec_rad, dist_km);
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: kepler_solver                                           */
/* Kepler equation via libnova's Sinnott bisection method              */
/* Input per line: M_rad  e                                            */
/* ------------------------------------------------------------------ */

static void run_kepler_solver(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"kepler_solver\",\"library\":\"libnova\",");
    printf("\"model\":\"Sinnott_bisection\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double M_rad, e;
        if (scanf("%lf %lf", &M_rad, &e) != 2) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }

        /* libnova uses degrees */
        double M_deg = M_rad * (180.0 / M_PI);
        double E_deg = ln_solve_kepler(e, M_deg);
        double nu_deg = ln_get_ell_true_anomaly(e, E_deg);

        double E_rad  = E_deg  * (M_PI / 180.0);
        double nu_rad = nu_deg * (M_PI / 180.0);

        /* Self-consistency: recompute M from E */
        double residual = fabs(E_rad - e * sin(E_rad) - M_rad);

        if (i > 0) printf(",\n");
        printf("{\"M_rad\":%.17e,\"e\":%.17e,", M_rad, e);
        printf("\"E_rad\":%.17e,\"nu_rad\":%.17e,", E_rad, nu_rad);
        printf("\"residual_rad\":%.17e,\"iters\":-1,\"converged\":%s}",
               residual, residual < 1e-6 ? "true" : "false");
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

static void run_gmst_era_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *jd_ut1_arr = malloc(n * sizeof(double));
    double *jd_tt_arr = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        if (scanf("%lf %lf", &jd_ut1_arr[i], &jd_tt_arr[i]) != 2) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }
    }

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double gmst = ln_get_mean_sidereal_time(jd_ut1_arr[i]);
        (void)gmst;
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double gmst = ln_get_mean_sidereal_time(jd_ut1_arr[i]);
        sink += gmst;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"gmst_era_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(jd_ut1_arr);
    free(jd_tt_arr);
}

static void run_equ_ecl_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *jds = malloc(n * sizeof(double));
    double *ras = malloc(n * sizeof(double));
    double *decs = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        if (scanf("%lf %lf %lf", &jds[i], &ras[i], &decs[i]) != 3) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }
    }

    /* Pre-convert to libnova API units outside timed loops for fairness. */
    double *ras_deg = malloc(n * sizeof(double));
    double *decs_deg = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        ras_deg[i] = ras[i] * (180.0 / M_PI);
        decs_deg[i] = decs[i] * (180.0 / M_PI);
    }

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        struct ln_equ_posn equ = { ras_deg[i], decs_deg[i] };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, jds[i], &ecl);
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        struct ln_equ_posn equ = { ras_deg[i], decs_deg[i] };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, jds[i], &ecl);
        sink += ecl.lng;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"equ_ecl_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(jds);
    free(ras);
    free(decs);
    free(ras_deg);
    free(decs_deg);
}

static void run_equ_horizontal_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *params = malloc(n * 6 * sizeof(double));

    for (int i = 0; i < n; i++) {
        if (scanf("%lf %lf %lf %lf %lf %lf",
                  &params[6*i], &params[6*i+1], &params[6*i+2],
                  &params[6*i+3], &params[6*i+4], &params[6*i+5]) != 6) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }
    }

    /* Pre-convert to libnova degree-based coordinates outside timing. */
    double *params_deg = malloc(n * 5 * sizeof(double));
    for (int i = 0; i < n; i++) {
        params_deg[5*i] = params[6*i];  /* jd_ut1 */
        params_deg[5*i+1] = params[6*i+2] * (180.0 / M_PI); /* ra */
        params_deg[5*i+2] = params[6*i+3] * (180.0 / M_PI); /* dec */
        params_deg[5*i+3] = params[6*i+4] * (180.0 / M_PI); /* lon */
        params_deg[5*i+4] = params[6*i+5] * (180.0 / M_PI); /* lat */
    }

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double jd_ut1 = params_deg[5*i];
        double ra = params_deg[5*i+1];
        double dec = params_deg[5*i+2];
        double lon = params_deg[5*i+3];
        double lat = params_deg[5*i+4];

        struct ln_equ_posn object = { ra, dec };
        struct ln_lnlat_posn observer = { lon, lat };
        struct ln_hrz_posn hrz;
        ln_get_hrz_from_equ(&object, &observer, jd_ut1, &hrz);
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double jd_ut1 = params_deg[5*i];
        double ra = params_deg[5*i+1];
        double dec = params_deg[5*i+2];
        double lon = params_deg[5*i+3];
        double lat = params_deg[5*i+4];

        struct ln_equ_posn object = { ra, dec };
        struct ln_lnlat_posn observer = { lon, lat };
        struct ln_hrz_posn hrz;
        ln_get_hrz_from_equ(&object, &observer, jd_ut1, &hrz);
        sink += hrz.az;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"equ_horizontal_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(params);
    free(params_deg);
}

static void run_solar_position_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *jds = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        if (scanf("%lf", &jds[i]) != 1) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }
    }

    /* Warm-up: match functional scope (RA/Dec plus distance). */
    for (int i = 0; i < n && i < 100; i++) {
        struct ln_equ_posn equ;
        ln_get_solar_equ_coords(jds[i], &equ);
        double dist_au = ln_get_earth_solar_dist(jds[i]);
        (void)dist_au;
    }

    /* Timed run — perf contract: compute RA + Dec + distance */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        struct ln_equ_posn equ;
        ln_get_solar_equ_coords(jds[i], &equ);
        double dist_au = ln_get_earth_solar_dist(jds[i]);
        sink += equ.ra + equ.dec + dist_au;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"solar_position_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(jds);
}

static void run_lunar_position_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *jds = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        if (scanf("%lf", &jds[i]) != 1) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }
    }

    /* Warm-up: match functional scope (RA/Dec plus distance). */
    for (int i = 0; i < n && i < 100; i++) {
        struct ln_equ_posn equ;
        ln_get_lunar_equ_coords(jds[i], &equ);
        double dist_km = ln_get_lunar_earth_dist(jds[i]);
        (void)dist_km;
    }

    /* Timed run — perf contract: compute RA + Dec + distance */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        struct ln_equ_posn equ;
        ln_get_lunar_equ_coords(jds[i], &equ);
        double dist_km = ln_get_lunar_earth_dist(jds[i]);
        sink += equ.ra + equ.dec + dist_km;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"lunar_position_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(jds);
}

static void run_kepler_solver_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *m_arr = malloc(n * sizeof(double));
    double *e_arr = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        if (scanf("%lf %lf", &m_arr[i], &e_arr[i]) != 2) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }
    }

    /* Pre-convert M radians -> degrees for ln_solve_kepler API parity. */
    double *m_deg_arr = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        m_deg_arr[i] = m_arr[i] * (180.0 / M_PI);
    }

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double E = ln_solve_kepler(e_arr[i], m_deg_arr[i]);
        (void)E;
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double E = ln_solve_kepler(e_arr[i], m_deg_arr[i]);
        sink += E;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"kepler_solver_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(m_arr);
    free(e_arr);
    free(m_deg_arr);
}

/* ================================================================== */
/* NEW COORDINATE-TRANSFORM EXPERIMENTS                                */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* Experiment: frame_bias                                              */
/* libnova has no frame bias concept — output skipped JSON.            */
/* ------------------------------------------------------------------ */

static void run_frame_bias(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    /* Consume input lines */
    for (int i = 0; i < n; i++) {
        double a, b, c, d;
        scanf("%lf %lf %lf %lf", &a, &b, &c, &d);
    }
    printf("{\"experiment\":\"frame_bias\",\"library\":\"libnova\",\"skipped\":true,");
    printf("\"reason\":\"libnova has no frame bias concept\"}\n");
}

static void run_frame_bias_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    for (int i = 0; i < n; i++) {
        double a, b, c, d;
        scanf("%lf %lf %lf %lf", &a, &b, &c, &d);
    }
    printf("{\"experiment\":\"frame_bias_perf\",\"library\":\"libnova\",\"skipped\":true,");
    printf("\"reason\":\"libnova has no frame bias concept\"}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: precession                                              */
/* J2000 → MeanOfDate via ln_get_equ_prec                              */
/* Input per line: jd_tt vx vy vz                                      */
/* ------------------------------------------------------------------ */

static void run_precession(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"precession\",\"library\":\"libnova\",");
    printf("\"model\":\"Meeus_precession\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double vin[3] = {vx, vy, vz};
        normalize3(vin);

        double ra_deg, dec_deg;
        cart_to_radec(vin, &ra_deg, &dec_deg);

        struct ln_equ_posn mean_pos = { .ra = ra_deg, .dec = dec_deg };
        struct ln_equ_posn prec_pos;
        ln_get_equ_prec(&mean_pos, jd_tt, &prec_pos);

        double vout[3];
        radec_to_cart(prec_pos.ra, prec_pos.dec, vout);
        normalize3(vout);

        /* Closure via reverse precession */
        struct ln_equ_posn back_pos;
        ln_get_equ_prec2(&prec_pos, jd_tt, JD2000, &back_pos);
        double vback[3];
        radec_to_cart(back_pos.ra, back_pos.dec, vback);
        normalize3(vback);
        double closure_rad = ang_sep(vin, vback);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],", vout[0], vout[1], vout[2]);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

static void run_precession_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *jds = malloc(n * sizeof(double));
    double *ras_deg = malloc(n * sizeof(double));
    double *decs_deg = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        jds[i] = jd_tt;
        double vin[3] = {vx, vy, vz};
        normalize3(vin);
        cart_to_radec(vin, &ras_deg[i], &decs_deg[i]);
    }

    for (int i = 0; i < n && i < 100; i++) {
        struct ln_equ_posn mean_pos = { ras_deg[i], decs_deg[i] };
        struct ln_equ_posn prec_pos;
        ln_get_equ_prec(&mean_pos, jds[i], &prec_pos);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        struct ln_equ_posn mean_pos = { ras_deg[i], decs_deg[i] };
        struct ln_equ_posn prec_pos;
        ln_get_equ_prec(&mean_pos, jds[i], &prec_pos);
        sink += prec_pos.ra;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);

    printf("{\"experiment\":\"precession_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, elapsed_ns / n);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);
    free(jds); free(ras_deg); free(decs_deg);
}

/* ------------------------------------------------------------------ */
/* Experiment: nutation                                                */
/* MeanOfDate → TrueOfDate via ln_get_equ_nut                         */
/* Input per line: jd_tt vx vy vz                                      */
/* ------------------------------------------------------------------ */

static void run_nutation(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"nutation\",\"library\":\"libnova\",");
    printf("\"model\":\"IAU_1980_nutation\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double vin[3] = {vx, vy, vz};
        normalize3(vin);

        double ra_deg, dec_deg;
        cart_to_radec(vin, &ra_deg, &dec_deg);

        struct ln_equ_posn mean_of_date = { .ra = ra_deg, .dec = dec_deg };
        struct ln_equ_posn true_of_date;
        ln_get_equ_nut(&mean_of_date, jd_tt, &true_of_date);

        double vout[3];
        radec_to_cart(true_of_date.ra, true_of_date.dec, vout);
        normalize3(vout);

        /* Closure: undo nutation */
        struct ln_nutation nut;
        ln_get_nutation(jd_tt, &nut);

        double ra_r  = true_of_date.ra * (M_PI / 180.0);
        double dec_r = true_of_date.dec * (M_PI / 180.0);
        double nut_ecliptic = (nut.ecliptic + nut.obliquity) * (M_PI / 180.0);
        double delta_ra = (cos(nut_ecliptic) + sin(nut_ecliptic) * sin(ra_r) * tan(dec_r)) * nut.longitude
                        - cos(ra_r) * tan(dec_r) * nut.obliquity;
        double delta_dec = (sin(nut_ecliptic) * cos(ra_r)) * nut.longitude
                         + sin(ra_r) * nut.obliquity;
        double back_ra  = true_of_date.ra  - delta_ra;
        double back_dec = true_of_date.dec - delta_dec;

        double vback[3];
        radec_to_cart(back_ra, back_dec, vback);
        normalize3(vback);
        double closure_rad = ang_sep(vin, vback);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],", vout[0], vout[1], vout[2]);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

static void run_nutation_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *jds = malloc(n * sizeof(double));
    double *ras_deg = malloc(n * sizeof(double));
    double *decs_deg = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        jds[i] = jd_tt;
        double vin[3] = {vx, vy, vz};
        normalize3(vin);
        cart_to_radec(vin, &ras_deg[i], &decs_deg[i]);
    }

    for (int i = 0; i < n && i < 100; i++) {
        struct ln_equ_posn pos = { ras_deg[i], decs_deg[i] };
        struct ln_equ_posn out;
        ln_get_equ_nut(&pos, jds[i], &out);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        struct ln_equ_posn pos = { ras_deg[i], decs_deg[i] };
        struct ln_equ_posn out;
        ln_get_equ_nut(&pos, jds[i], &out);
        sink += out.ra;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);

    printf("{\"experiment\":\"nutation_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, elapsed_ns / n);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);
    free(jds); free(ras_deg); free(decs_deg);
}

/* ------------------------------------------------------------------ */
/* Experiment: icrs_ecl_j2000                                          */
/* Equatorial → Ecliptic via ln_get_ecl_from_equ at J2000              */
/* Input per line: jd_tt vx vy vz                                      */
/* ------------------------------------------------------------------ */

static void run_icrs_ecl_j2000(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"icrs_ecl_j2000\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_transform_J2000\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double vin[3] = {vx, vy, vz};
        normalize3(vin);

        double ra_deg, dec_deg;
        cart_to_radec(vin, &ra_deg, &dec_deg);

        struct ln_equ_posn equ = { .ra = ra_deg, .dec = dec_deg };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, JD2000, &ecl);

        double ecl_lon = ecl.lng * (M_PI / 180.0);
        double ecl_lat = ecl.lat * (M_PI / 180.0);

        /* Build output unit vector in ecliptic frame */
        double vout[3] = { cos(ecl_lat)*cos(ecl_lon), cos(ecl_lat)*sin(ecl_lon), sin(ecl_lat) };
        normalize3(vout);

        /* Closure */
        struct ln_equ_posn equ_back;
        ln_get_equ_from_ecl(&ecl, JD2000, &equ_back);
        double vback[3];
        radec_to_cart(equ_back.ra, equ_back.dec, vback);
        normalize3(vback);
        double closure_rad = ang_sep(vin, vback);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],", vout[0], vout[1], vout[2]);
        printf("\"ecl_lon_rad\":%.17e,\"ecl_lat_rad\":%.17e,", ecl_lon, ecl_lat);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

static void run_icrs_ecl_j2000_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *ras_deg = malloc(n * sizeof(double));
    double *decs_deg = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double vin[3] = {vx, vy, vz};
        normalize3(vin);
        cart_to_radec(vin, &ras_deg[i], &decs_deg[i]);
    }

    for (int i = 0; i < n && i < 100; i++) {
        struct ln_equ_posn equ = { ras_deg[i], decs_deg[i] };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, JD2000, &ecl);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        struct ln_equ_posn equ = { ras_deg[i], decs_deg[i] };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, JD2000, &ecl);
        sink += ecl.lng;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);

    printf("{\"experiment\":\"icrs_ecl_j2000_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, elapsed_ns / n);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);
    free(ras_deg); free(decs_deg);
}

/* ------------------------------------------------------------------ */
/* Experiment: icrs_ecl_tod                                            */
/* Equatorial → Ecliptic of date via ln_get_ecl_from_equ               */
/* Input per line: jd_tt ra_rad dec_rad                                */
/* ------------------------------------------------------------------ */

static void run_icrs_ecl_tod(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"icrs_ecl_tod\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_transform_of_date\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt, ra_rad, dec_rad;
        if (scanf("%lf %lf %lf", &jd_tt, &ra_rad, &dec_rad) != 3) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double ra_deg  = ra_rad  * (180.0 / M_PI);
        double dec_deg = dec_rad * (180.0 / M_PI);

        struct ln_equ_posn equ = { .ra = ra_deg, .dec = dec_deg };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, jd_tt, &ecl);

        double ecl_lon = ecl.lng * (M_PI / 180.0);
        double ecl_lat = ecl.lat * (M_PI / 180.0);

        /* Closure */
        struct ln_equ_posn equ_back;
        ln_get_equ_from_ecl(&ecl, jd_tt, &equ_back);
        double ra_back  = equ_back.ra  * (M_PI / 180.0);
        double dec_back = equ_back.dec * (M_PI / 180.0);
        double v_in[3]  = { cos(dec_rad)*cos(ra_rad), cos(dec_rad)*sin(ra_rad), sin(dec_rad) };
        double v_bk[3]  = { cos(dec_back)*cos(ra_back), cos(dec_back)*sin(ra_back), sin(dec_back) };
        double closure_rad = ang_sep(v_in, v_bk);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"ra_rad\":%.17e,\"dec_rad\":%.17e,", jd_tt, ra_rad, dec_rad);
        printf("\"ecl_lon_rad\":%.17e,\"ecl_lat_rad\":%.17e,", ecl_lon, ecl_lat);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

static void run_icrs_ecl_tod_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *jds = malloc(n * sizeof(double));
    double *ras_deg = malloc(n * sizeof(double));
    double *decs_deg = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double jd_tt, ra_rad, dec_rad;
        if (scanf("%lf %lf %lf", &jd_tt, &ra_rad, &dec_rad) != 3) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        jds[i] = jd_tt;
        ras_deg[i] = ra_rad * (180.0 / M_PI);
        decs_deg[i] = dec_rad * (180.0 / M_PI);
    }

    for (int i = 0; i < n && i < 100; i++) {
        struct ln_equ_posn equ = { ras_deg[i], decs_deg[i] };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, jds[i], &ecl);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        struct ln_equ_posn equ = { ras_deg[i], decs_deg[i] };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, jds[i], &ecl);
        sink += ecl.lng + ecl.lat;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);

    printf("{\"experiment\":\"icrs_ecl_tod_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, elapsed_ns / n);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);
    free(jds); free(ras_deg); free(decs_deg);
}

/* ------------------------------------------------------------------ */
/* Experiment: horiz_to_equ                                            */
/* Horizontal → Equatorial via ln_get_equ_from_hrz                     */
/* Input: jd_ut1 jd_tt az_rad alt_rad obs_lon_rad obs_lat_rad         */
/* ------------------------------------------------------------------ */

static void run_horiz_to_equ(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"horiz_to_equ\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_transform\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_ut1, jd_tt, az_rad, alt_rad, obs_lon, obs_lat;
        if (scanf("%lf %lf %lf %lf %lf %lf", &jd_ut1, &jd_tt, &az_rad, &alt_rad,
                  &obs_lon, &obs_lat) != 6) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        /* Convert ERFA convention (az: 0=N, CW) → libnova (az: 0=S, CW):
         * az_libnova = (az_erfa - 180) mod 360 */
        double az_deg_ln = fmod((az_rad * (180.0 / M_PI)) - 180.0 + 360.0, 360.0);
        double alt_deg = alt_rad * (180.0 / M_PI);
        double lon_deg = obs_lon * (180.0 / M_PI);
        double lat_deg = obs_lat * (180.0 / M_PI);

        struct ln_hrz_posn hrz = { .az = az_deg_ln, .alt = alt_deg };
        struct ln_lnlat_posn observer = { .lng = lon_deg, .lat = lat_deg };
        struct ln_equ_posn equ;
        ln_get_equ_from_hrz(&hrz, &observer, jd_ut1, &equ);

        double ra  = equ.ra  * (M_PI / 180.0);
        double dec = equ.dec * (M_PI / 180.0);

        /* Closure: equ → hrz → equ */
        struct ln_hrz_posn hrz2;
        ln_get_hrz_from_equ(&equ, &observer, jd_ut1, &hrz2);
        struct ln_equ_posn equ_back;
        ln_get_equ_from_hrz(&hrz2, &observer, jd_ut1, &equ_back);

        double ra_back  = equ_back.ra  * (M_PI / 180.0);
        double dec_back = equ_back.dec * (M_PI / 180.0);
        double v_in[3]  = { cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec) };
        double v_bk[3]  = { cos(dec_back)*cos(ra_back), cos(dec_back)*sin(ra_back), sin(dec_back) };
        double closure_rad = ang_sep(v_in, v_bk);

        if (i > 0) printf(",\n");
        printf("{\"jd_ut1\":%.15f,\"jd_tt\":%.15f,", jd_ut1, jd_tt);
        printf("\"az_rad\":%.17e,\"alt_rad\":%.17e,", az_rad, alt_rad);
        printf("\"obs_lon_rad\":%.17e,\"obs_lat_rad\":%.17e,", obs_lon, obs_lat);
        printf("\"ra_rad\":%.17e,\"dec_rad\":%.17e,", ra, dec);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

static void run_horiz_to_equ_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    double *params = malloc(n * 6 * sizeof(double));
    for (int i = 0; i < n; i++) {
        if (scanf("%lf %lf %lf %lf %lf %lf",
                  &params[6*i], &params[6*i+1], &params[6*i+2],
                  &params[6*i+3], &params[6*i+4], &params[6*i+5]) != 6) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
    }

    /* Pre-convert to libnova degree-based coordinates */
    double *p_deg = malloc(n * 5 * sizeof(double));
    for (int i = 0; i < n; i++) {
        p_deg[5*i]   = params[6*i]; /* jd_ut1 */
        double az_deg_ln = fmod((params[6*i+2] * (180.0 / M_PI)) - 180.0 + 360.0, 360.0);
        p_deg[5*i+1] = az_deg_ln;
        p_deg[5*i+2] = params[6*i+3] * (180.0 / M_PI); /* alt */
        p_deg[5*i+3] = params[6*i+4] * (180.0 / M_PI); /* lon */
        p_deg[5*i+4] = params[6*i+5] * (180.0 / M_PI); /* lat */
    }

    for (int i = 0; i < n && i < 100; i++) {
        struct ln_hrz_posn hrz = { p_deg[5*i+1], p_deg[5*i+2] };
        struct ln_lnlat_posn obs = { p_deg[5*i+3], p_deg[5*i+4] };
        struct ln_equ_posn equ;
        ln_get_equ_from_hrz(&hrz, &obs, p_deg[5*i], &equ);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        struct ln_hrz_posn hrz = { p_deg[5*i+1], p_deg[5*i+2] };
        struct ln_lnlat_posn obs = { p_deg[5*i+3], p_deg[5*i+4] };
        struct ln_equ_posn equ;
        ln_get_equ_from_hrz(&hrz, &obs, p_deg[5*i], &equ);
        sink += equ.ra + equ.dec;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);

    printf("{\"experiment\":\"horiz_to_equ_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, elapsed_ns / n);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);
    free(params); free(p_deg);
}

/* ================================================================== */
/* 13 NEW DIRECTION-VECTOR TRANSFORM EXPERIMENTS                       */
/* ================================================================== */

/* Helper: skip an experiment (consume N direction-vector inputs) */
static void _skip_dir(const char *exp_name, const char *reason) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    for (int i = 0; i < n; i++) { double a,b,c,d; scanf("%lf %lf %lf %lf", &a,&b,&c,&d); }
    printf("{\"experiment\":\"%s\",\"library\":\"libnova\",\"skipped\":true,", exp_name);
    printf("\"reason\":\"%s\"}\n", reason);
}

/* --- inv_frame_bias: skipped --- */
static void run_inv_frame_bias(void)      { _skip_dir("inv_frame_bias",      "libnova has no frame bias concept"); }
static void run_inv_frame_bias_perf(void) { _skip_dir("inv_frame_bias_perf", "libnova has no frame bias concept"); }

/* --- inv_bpn: skipped (no ICRS/frame bias) --- */
static void run_inv_bpn(void)      { _skip_dir("inv_bpn",      "libnova has no ICRS/frame bias concept"); }
static void run_inv_bpn_perf(void) { _skip_dir("inv_bpn_perf", "libnova has no ICRS/frame bias concept"); }

/* --- bias_precession: skipped --- */
static void run_bias_precession(void)      { _skip_dir("bias_precession",      "libnova has no frame bias concept"); }
static void run_bias_precession_perf(void) { _skip_dir("bias_precession_perf", "libnova has no frame bias concept"); }

/* --- inv_bias_precession: skipped --- */
static void run_inv_bias_precession(void)      { _skip_dir("inv_bias_precession",      "libnova has no frame bias concept"); }
static void run_inv_bias_precession_perf(void) { _skip_dir("inv_bias_precession_perf", "libnova has no frame bias concept"); }

/* --- inv_precession: MeanOfDate → J2000 via ln_get_equ_prec2 --- */
static void run_inv_precession(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"inv_precession\",\"library\":\"libnova\",");
    printf("\"model\":\"Meeus_inv_precession\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3] = {vx, vy, vz}; normalize3(vin);
        double ra_deg, dec_deg;
        cart_to_radec(vin, &ra_deg, &dec_deg);
        struct ln_equ_posn pos_date = { ra_deg, dec_deg };
        struct ln_equ_posn pos_j2000;
        ln_get_equ_prec2(&pos_date, jd_tt, JD2000, &pos_j2000);
        double vout[3]; radec_to_cart(pos_j2000.ra, pos_j2000.dec, vout); normalize3(vout);
        /* Closure: precess forward again */
        struct ln_equ_posn check;
        ln_get_equ_prec(&pos_j2000, jd_tt, &check);
        double vback[3]; radec_to_cart(check.ra, check.dec, vback); normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_inv_precession_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *jds = malloc(n * sizeof(double));
    double *ras = malloc(n * sizeof(double));
    double *decs = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz);
        jds[i] = jd_tt; double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v, &ras[i], &decs[i]);
    }
    for (int i = 0; i < n && i < 100; i++) {
        struct ln_equ_posn p={ras[i],decs[i]}, o;
        ln_get_equ_prec2(&p, jds[i], JD2000, &o);
    }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_equ_posn p={ras[i],decs[i]}, o;
        ln_get_equ_prec2(&p, jds[i], JD2000, &o);
        sink += o.ra;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"inv_precession_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(jds); free(ras); free(decs);
}

/* --- inv_nutation: TrueOfDate → MeanOfDate (approximate ΔRA/ΔDec subtraction) --- */
static void run_inv_nutation(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"inv_nutation\",\"library\":\"libnova\",");
    printf("\"model\":\"IAU_1980_inv_nutation_approx\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3]={vx,vy,vz}; normalize3(vin);
        double ra_deg, dec_deg;
        cart_to_radec(vin, &ra_deg, &dec_deg);
        /* Compute nutation corrections and subtract (approximate inverse) */
        struct ln_nutation nut;
        ln_get_nutation(jd_tt, &nut);
        double ra_r = ra_deg * (M_PI/180.0);
        double dec_r = dec_deg * (M_PI/180.0);
        double nut_ecl = (nut.ecliptic + nut.obliquity) * (M_PI/180.0);
        double delta_ra = (cos(nut_ecl) + sin(nut_ecl)*sin(ra_r)*tan(dec_r)) * nut.longitude
                        - cos(ra_r)*tan(dec_r) * nut.obliquity;
        double delta_dec = (sin(nut_ecl)*cos(ra_r)) * nut.longitude + sin(ra_r) * nut.obliquity;
        double back_ra = ra_deg - delta_ra;
        double back_dec = dec_deg - delta_dec;
        double vout[3]; radec_to_cart(back_ra, back_dec, vout); normalize3(vout);
        /* Closure: apply nutation forward */
        struct ln_equ_posn mean_pos = { back_ra, back_dec };
        struct ln_equ_posn true_check;
        ln_get_equ_nut(&mean_pos, jd_tt, &true_check);
        double vback[3]; radec_to_cart(true_check.ra, true_check.dec, vback); normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_inv_nutation_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *jds=malloc(n*sizeof(double)), *ras=malloc(n*sizeof(double)), *decs=malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
        double jd,vx,vy,vz; scanf("%lf %lf %lf %lf",&jd,&vx,&vy,&vz);
        jds[i]=jd; double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v,&ras[i],&decs[i]);
    }
    for (int i=0;i<n&&i<100;i++) { struct ln_nutation nut; ln_get_nutation(jds[i],&nut); }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_nutation nut; ln_get_nutation(jds[i],&nut);
        double ra_r=ras[i]*(M_PI/180.0), dec_r=decs[i]*(M_PI/180.0);
        double nut_ecl=(nut.ecliptic+nut.obliquity)*(M_PI/180.0);
        double delta_ra = (cos(nut_ecl)+sin(nut_ecl)*sin(ra_r)*tan(dec_r))*nut.longitude
                        - cos(ra_r)*tan(dec_r)*nut.obliquity;
        sink += ras[i]-delta_ra;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"inv_nutation_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(jds); free(ras); free(decs);
}

/* --- inv_icrs_ecl_j2000: EclMeanJ2000 → ICRS (≈EqJ2000) via ln_get_equ_from_ecl at J2000 --- */
static void run_inv_icrs_ecl_j2000(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"inv_icrs_ecl_j2000\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_inv_ecl_j2000\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3]={vx,vy,vz}; normalize3(vin);
        /* Input is ecliptic direction vector → convert to lon/lat */
        double lon_deg, lat_deg;
        cart_to_radec(vin, &lon_deg, &lat_deg); /* same math for spherical coords */
        struct ln_lnlat_posn ecl = { lon_deg, lat_deg };
        struct ln_equ_posn equ;
        ln_get_equ_from_ecl(&ecl, JD2000, &equ);
        double vout[3]; radec_to_cart(equ.ra, equ.dec, vout); normalize3(vout);
        /* Closure: convert back */
        struct ln_lnlat_posn ecl_back;
        ln_get_ecl_from_equ(&equ, JD2000, &ecl_back);
        double vback[3]; radec_to_cart(ecl_back.lng, ecl_back.lat, vback); normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_inv_icrs_ecl_j2000_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *lons=malloc(n*sizeof(double)), *lats=malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
        double jd,vx,vy,vz; scanf("%lf %lf %lf %lf",&jd,&vx,&vy,&vz);
        double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v,&lons[i],&lats[i]);
    }
    for (int i=0;i<n&&i<100;i++) {
        struct ln_lnlat_posn e={lons[i],lats[i]}; struct ln_equ_posn q;
        ln_get_equ_from_ecl(&e, JD2000, &q);
    }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_lnlat_posn e={lons[i],lats[i]}; struct ln_equ_posn q;
        ln_get_equ_from_ecl(&e, JD2000, &q);
        sink+=q.ra;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"inv_icrs_ecl_j2000_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(lons); free(lats);
}

/* --- obliquity: EclMeanJ2000 → EqMeanJ2000 via ln_get_equ_from_ecl at J2000 --- */
static void run_obliquity(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"obliquity\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_obliquity_j2000\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3]={vx,vy,vz}; normalize3(vin);
        double lon_deg, lat_deg;
        cart_to_radec(vin, &lon_deg, &lat_deg);
        struct ln_lnlat_posn ecl = { lon_deg, lat_deg };
        struct ln_equ_posn equ;
        ln_get_equ_from_ecl(&ecl, JD2000, &equ);
        double vout[3]; radec_to_cart(equ.ra, equ.dec, vout); normalize3(vout);
        struct ln_lnlat_posn ecl_back;
        ln_get_ecl_from_equ(&equ, JD2000, &ecl_back);
        double vback[3]; radec_to_cart(ecl_back.lng, ecl_back.lat, vback); normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_obliquity_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *lons=malloc(n*sizeof(double)), *lats=malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
        double jd,vx,vy,vz; scanf("%lf %lf %lf %lf",&jd,&vx,&vy,&vz);
        double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v,&lons[i],&lats[i]);
    }
    for (int i=0;i<n&&i<100;i++) {
        struct ln_lnlat_posn e={lons[i],lats[i]}; struct ln_equ_posn q;
        ln_get_equ_from_ecl(&e, JD2000, &q);
    }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_lnlat_posn e={lons[i],lats[i]}; struct ln_equ_posn q;
        ln_get_equ_from_ecl(&e, JD2000, &q); sink+=q.ra;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"obliquity_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(lons); free(lats);
}

/* --- inv_obliquity: EqMeanJ2000 → EclMeanJ2000 via ln_get_ecl_from_equ at J2000 --- */
static void run_inv_obliquity(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"inv_obliquity\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_inv_obliquity_j2000\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3]={vx,vy,vz}; normalize3(vin);
        double ra_deg, dec_deg;
        cart_to_radec(vin, &ra_deg, &dec_deg);
        struct ln_equ_posn equ = { ra_deg, dec_deg };
        struct ln_lnlat_posn ecl;
        ln_get_ecl_from_equ(&equ, JD2000, &ecl);
        double ecl_lon = ecl.lng * (M_PI/180.0);
        double ecl_lat = ecl.lat * (M_PI/180.0);
        double vout[3] = { cos(ecl_lat)*cos(ecl_lon), cos(ecl_lat)*sin(ecl_lon), sin(ecl_lat) };
        normalize3(vout);
        /* Closure */
        struct ln_equ_posn equ_back;
        ln_get_equ_from_ecl(&ecl, JD2000, &equ_back);
        double vback[3]; radec_to_cart(equ_back.ra, equ_back.dec, vback); normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_inv_obliquity_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *ras=malloc(n*sizeof(double)), *decs=malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
        double jd,vx,vy,vz; scanf("%lf %lf %lf %lf",&jd,&vx,&vy,&vz);
        double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v,&ras[i],&decs[i]);
    }
    for (int i=0;i<n&&i<100;i++) {
        struct ln_equ_posn e={ras[i],decs[i]}; struct ln_lnlat_posn q;
        ln_get_ecl_from_equ(&e, JD2000, &q);
    }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_equ_posn e={ras[i],decs[i]}; struct ln_lnlat_posn q;
        ln_get_ecl_from_equ(&e, JD2000, &q); sink+=q.lng;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"inv_obliquity_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(ras); free(decs);
}

/* --- precession_nutation: EqMeanJ2000 → EqTrueOfDate via prec then nut --- */
static void run_precession_nutation(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"precession_nutation\",\"library\":\"libnova\",");
    printf("\"model\":\"Meeus_prec_IAU1980_nut\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3]={vx,vy,vz}; normalize3(vin);
        double ra_deg, dec_deg;
        cart_to_radec(vin, &ra_deg, &dec_deg);
        /* Precession: J2000 → date */
        struct ln_equ_posn j2000 = { ra_deg, dec_deg };
        struct ln_equ_posn mean_date;
        ln_get_equ_prec(&j2000, jd_tt, &mean_date);
        /* Nutation: mean → true */
        struct ln_equ_posn true_date;
        ln_get_equ_nut(&mean_date, jd_tt, &true_date);
        double vout[3]; radec_to_cart(true_date.ra, true_date.dec, vout); normalize3(vout);
        /* Closure: reverse */
        struct ln_nutation nut; ln_get_nutation(jd_tt, &nut);
        double ra_r=true_date.ra*(M_PI/180.0), dec_r=true_date.dec*(M_PI/180.0);
        double nut_ecl=(nut.ecliptic+nut.obliquity)*(M_PI/180.0);
        double dra = (cos(nut_ecl)+sin(nut_ecl)*sin(ra_r)*tan(dec_r))*nut.longitude - cos(ra_r)*tan(dec_r)*nut.obliquity;
        double ddec = (sin(nut_ecl)*cos(ra_r))*nut.longitude + sin(ra_r)*nut.obliquity;
        struct ln_equ_posn back_mean = { true_date.ra - dra, true_date.dec - ddec };
        struct ln_equ_posn back_j2000;
        ln_get_equ_prec2(&back_mean, jd_tt, JD2000, &back_j2000);
        double vback[3]; radec_to_cart(back_j2000.ra, back_j2000.dec, vback); normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_precession_nutation_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *jds=malloc(n*sizeof(double)), *ras=malloc(n*sizeof(double)), *decs=malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
        double jd,vx,vy,vz; scanf("%lf %lf %lf %lf",&jd,&vx,&vy,&vz);
        jds[i]=jd; double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v,&ras[i],&decs[i]);
    }
    for (int i=0;i<n&&i<100;i++) {
        struct ln_equ_posn j={ras[i],decs[i]}, m, t;
        ln_get_equ_prec(&j,jds[i],&m); ln_get_equ_nut(&m,jds[i],&t);
    }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_equ_posn j={ras[i],decs[i]}, m, t;
        ln_get_equ_prec(&j,jds[i],&m); ln_get_equ_nut(&m,jds[i],&t); sink+=t.ra;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"precession_nutation_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(jds); free(ras); free(decs);
}

/* --- inv_precession_nutation: EqTrueOfDate → EqMeanJ2000 --- */
static void run_inv_precession_nutation(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"inv_precession_nutation\",\"library\":\"libnova\",");
    printf("\"model\":\"IAU1980_inv_nut_Meeus_inv_prec\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3]={vx,vy,vz}; normalize3(vin);
        double ra_deg, dec_deg;
        cart_to_radec(vin, &ra_deg, &dec_deg);
        /* Inverse nutation */
        struct ln_nutation nut; ln_get_nutation(jd_tt, &nut);
        double ra_r=ra_deg*(M_PI/180.0), dec_r=dec_deg*(M_PI/180.0);
        double nut_ecl=(nut.ecliptic+nut.obliquity)*(M_PI/180.0);
        double dra = (cos(nut_ecl)+sin(nut_ecl)*sin(ra_r)*tan(dec_r))*nut.longitude - cos(ra_r)*tan(dec_r)*nut.obliquity;
        double ddec = (sin(nut_ecl)*cos(ra_r))*nut.longitude + sin(ra_r)*nut.obliquity;
        double mean_ra = ra_deg - dra, mean_dec = dec_deg - ddec;
        /* Inverse precession */
        struct ln_equ_posn mean_pos = { mean_ra, mean_dec };
        struct ln_equ_posn j2000_pos;
        ln_get_equ_prec2(&mean_pos, jd_tt, JD2000, &j2000_pos);
        double vout[3]; radec_to_cart(j2000_pos.ra, j2000_pos.dec, vout); normalize3(vout);
        /* Closure: forward */
        struct ln_equ_posn fwd_mean, fwd_true;
        ln_get_equ_prec(&j2000_pos, jd_tt, &fwd_mean);
        ln_get_equ_nut(&fwd_mean, jd_tt, &fwd_true);
        double vback[3]; radec_to_cart(fwd_true.ra, fwd_true.dec, vback); normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_inv_precession_nutation_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *jds=malloc(n*sizeof(double)), *ras=malloc(n*sizeof(double)), *decs=malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
        double jd,vx,vy,vz; scanf("%lf %lf %lf %lf",&jd,&vx,&vy,&vz);
        jds[i]=jd; double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v,&ras[i],&decs[i]);
    }
    for (int i=0;i<n&&i<100;i++) {
        struct ln_nutation nut; ln_get_nutation(jds[i], &nut);
        struct ln_equ_posn m={ras[i]-0.001,decs[i]}, j;
        ln_get_equ_prec2(&m, jds[i], JD2000, &j);
    }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_nutation nut; ln_get_nutation(jds[i],&nut);
        double ra_r=ras[i]*(M_PI/180.0), dec_r=decs[i]*(M_PI/180.0);
        double nut_ecl=(nut.ecliptic+nut.obliquity)*(M_PI/180.0);
        double dra=(cos(nut_ecl)+sin(nut_ecl)*sin(ra_r)*tan(dec_r))*nut.longitude-cos(ra_r)*tan(dec_r)*nut.obliquity;
        double ddec=(sin(nut_ecl)*cos(ra_r))*nut.longitude+sin(ra_r)*nut.obliquity;
        struct ln_equ_posn m={ras[i]-dra,decs[i]-ddec}, j;
        ln_get_equ_prec2(&m,jds[i],JD2000,&j); sink+=j.ra;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"inv_precession_nutation_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(jds); free(ras); free(decs);
}

/* --- inv_icrs_ecl_tod: EclTrueOfDate → ICRS via ecl→eq(date) then prec→J2000 --- */
static void run_inv_icrs_ecl_tod(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"inv_icrs_ecl_tod\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_inv_ecl_tod\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3]={vx,vy,vz}; normalize3(vin);
        double lon_deg, lat_deg;
        cart_to_radec(vin, &lon_deg, &lat_deg);
        struct ln_lnlat_posn ecl = { lon_deg, lat_deg };
        struct ln_equ_posn equ_date;
        ln_get_equ_from_ecl(&ecl, jd_tt, &equ_date);
        /* Precess back to J2000 (≈ICRS) */
        struct ln_equ_posn equ_j2000;
        ln_get_equ_prec2(&equ_date, jd_tt, JD2000, &equ_j2000);
        double vout[3]; radec_to_cart(equ_j2000.ra, equ_j2000.dec, vout); normalize3(vout);
        /* Closure */
        struct ln_equ_posn fwd_date;
        ln_get_equ_prec(&equ_j2000, jd_tt, &fwd_date);
        struct ln_lnlat_posn ecl_back;
        ln_get_ecl_from_equ(&fwd_date, jd_tt, &ecl_back);
        double vback[3];
        double blon=ecl_back.lng*(M_PI/180.0), blat=ecl_back.lat*(M_PI/180.0);
        vback[0]=cos(blat)*cos(blon); vback[1]=cos(blat)*sin(blon); vback[2]=sin(blat);
        normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_inv_icrs_ecl_tod_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *jds=malloc(n*sizeof(double)), *lons=malloc(n*sizeof(double)), *lats=malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
        double jd,vx,vy,vz; scanf("%lf %lf %lf %lf",&jd,&vx,&vy,&vz);
        jds[i]=jd; double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v,&lons[i],&lats[i]);
    }
    for (int i=0;i<n&&i<100;i++) {
        struct ln_lnlat_posn e={lons[i],lats[i]}; struct ln_equ_posn q, j;
        ln_get_equ_from_ecl(&e,jds[i],&q); ln_get_equ_prec2(&q,jds[i],JD2000,&j);
    }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_lnlat_posn e={lons[i],lats[i]}; struct ln_equ_posn q, j;
        ln_get_equ_from_ecl(&e,jds[i],&q); ln_get_equ_prec2(&q,jds[i],JD2000,&j); sink+=j.ra;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"inv_icrs_ecl_tod_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(jds); free(lons); free(lats);
}

/* --- inv_equ_ecl: EclTrueOfDate → EqMeanOfDate via ln_get_equ_from_ecl(date) --- */
static void run_inv_equ_ecl(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    printf("{\"experiment\":\"inv_equ_ecl\",\"library\":\"libnova\",");
    printf("\"model\":\"libnova_inv_equ_ecl\",");
    printf("\"count\":%d,\"cases\":[\n", n);
    for (int i = 0; i < n; i++) {
        double jd_tt,vx,vy,vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) { fprintf(stderr,"bad\n"); exit(1); }
        double vin[3]={vx,vy,vz}; normalize3(vin);
        double lon_deg, lat_deg;
        cart_to_radec(vin, &lon_deg, &lat_deg);
        struct ln_lnlat_posn ecl = { lon_deg, lat_deg };
        struct ln_equ_posn equ_date;
        ln_get_equ_from_ecl(&ecl, jd_tt, &equ_date);
        double vout[3]; radec_to_cart(equ_date.ra, equ_date.dec, vout); normalize3(vout);
        /* Closure */
        struct ln_lnlat_posn ecl_back;
        ln_get_ecl_from_equ(&equ_date, jd_tt, &ecl_back);
        double vback[3];
        double blon=ecl_back.lng*(M_PI/180.0), blat=ecl_back.lat*(M_PI/180.0);
        vback[0]=cos(blat)*cos(blon); vback[1]=cos(blat)*sin(blon); vback[2]=sin(blat);
        normalize3(vback);
        double closure_rad = ang_sep(vin, vback);
        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],\"closure_rad\":%.17e}", vout[0], vout[1], vout[2], closure_rad);
    }
    printf("\n]}\n");
}

static void run_inv_equ_ecl_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }
    double *jds=malloc(n*sizeof(double)), *lons=malloc(n*sizeof(double)), *lats=malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
        double jd,vx,vy,vz; scanf("%lf %lf %lf %lf",&jd,&vx,&vy,&vz);
        jds[i]=jd; double v[3]={vx,vy,vz}; normalize3(v); cart_to_radec(v,&lons[i],&lats[i]);
    }
    for (int i=0;i<n&&i<100;i++) {
        struct ln_lnlat_posn e={lons[i],lats[i]}; struct ln_equ_posn q;
        ln_get_equ_from_ecl(&e,jds[i],&q);
    }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double sink=0;
    for (int i=0;i<n;i++) {
        struct ln_lnlat_posn e={lons[i],lats[i]}; struct ln_equ_posn q;
        ln_get_equ_from_ecl(&e,jds[i],&q); sink+=q.ra;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    printf("{\"experiment\":\"inv_equ_ecl_perf\",\"library\":\"libnova\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           n, ns, ns/n, (double)n/(ns*1e-9), sink);
    free(jds); free(lons); free(lats);
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
    } else if (strcmp(experiment, "equ_ecl") == 0) {
        run_equ_ecl();
    } else if (strcmp(experiment, "equ_horizontal") == 0) {
        run_equ_horizontal();
    } else if (strcmp(experiment, "solar_position") == 0) {
        run_solar_position();
    } else if (strcmp(experiment, "lunar_position") == 0) {
        run_lunar_position();
    } else if (strcmp(experiment, "kepler_solver") == 0) {
        run_kepler_solver();
    } else if (strcmp(experiment, "frame_rotation_bpn_perf") == 0) {
        run_frame_rotation_bpn_perf();
    } else if (strcmp(experiment, "gmst_era_perf") == 0) {
        run_gmst_era_perf();
    } else if (strcmp(experiment, "equ_ecl_perf") == 0) {
        run_equ_ecl_perf();
    } else if (strcmp(experiment, "equ_horizontal_perf") == 0) {
        run_equ_horizontal_perf();
    } else if (strcmp(experiment, "solar_position_perf") == 0) {
        run_solar_position_perf();
    } else if (strcmp(experiment, "lunar_position_perf") == 0) {
        run_lunar_position_perf();
    } else if (strcmp(experiment, "kepler_solver_perf") == 0) {
        run_kepler_solver_perf();
    } else if (strcmp(experiment, "frame_bias") == 0) {
        run_frame_bias();
    } else if (strcmp(experiment, "frame_bias_perf") == 0) {
        run_frame_bias_perf();
    } else if (strcmp(experiment, "precession") == 0) {
        run_precession();
    } else if (strcmp(experiment, "precession_perf") == 0) {
        run_precession_perf();
    } else if (strcmp(experiment, "nutation") == 0) {
        run_nutation();
    } else if (strcmp(experiment, "nutation_perf") == 0) {
        run_nutation_perf();
    } else if (strcmp(experiment, "icrs_ecl_j2000") == 0) {
        run_icrs_ecl_j2000();
    } else if (strcmp(experiment, "icrs_ecl_j2000_perf") == 0) {
        run_icrs_ecl_j2000_perf();
    } else if (strcmp(experiment, "icrs_ecl_tod") == 0) {
        run_icrs_ecl_tod();
    } else if (strcmp(experiment, "icrs_ecl_tod_perf") == 0) {
        run_icrs_ecl_tod_perf();
    } else if (strcmp(experiment, "horiz_to_equ") == 0) {
        run_horiz_to_equ();
    } else if (strcmp(experiment, "horiz_to_equ_perf") == 0) {
        run_horiz_to_equ_perf();
    /* 13 new matrix experiments */
    } else if (strcmp(experiment, "inv_frame_bias") == 0) {
        run_inv_frame_bias();
    } else if (strcmp(experiment, "inv_frame_bias_perf") == 0) {
        run_inv_frame_bias_perf();
    } else if (strcmp(experiment, "inv_precession") == 0) {
        run_inv_precession();
    } else if (strcmp(experiment, "inv_precession_perf") == 0) {
        run_inv_precession_perf();
    } else if (strcmp(experiment, "inv_nutation") == 0) {
        run_inv_nutation();
    } else if (strcmp(experiment, "inv_nutation_perf") == 0) {
        run_inv_nutation_perf();
    } else if (strcmp(experiment, "inv_bpn") == 0) {
        run_inv_bpn();
    } else if (strcmp(experiment, "inv_bpn_perf") == 0) {
        run_inv_bpn_perf();
    } else if (strcmp(experiment, "inv_icrs_ecl_j2000") == 0) {
        run_inv_icrs_ecl_j2000();
    } else if (strcmp(experiment, "inv_icrs_ecl_j2000_perf") == 0) {
        run_inv_icrs_ecl_j2000_perf();
    } else if (strcmp(experiment, "obliquity") == 0) {
        run_obliquity();
    } else if (strcmp(experiment, "obliquity_perf") == 0) {
        run_obliquity_perf();
    } else if (strcmp(experiment, "inv_obliquity") == 0) {
        run_inv_obliquity();
    } else if (strcmp(experiment, "inv_obliquity_perf") == 0) {
        run_inv_obliquity_perf();
    } else if (strcmp(experiment, "bias_precession") == 0) {
        run_bias_precession();
    } else if (strcmp(experiment, "bias_precession_perf") == 0) {
        run_bias_precession_perf();
    } else if (strcmp(experiment, "inv_bias_precession") == 0) {
        run_inv_bias_precession();
    } else if (strcmp(experiment, "inv_bias_precession_perf") == 0) {
        run_inv_bias_precession_perf();
    } else if (strcmp(experiment, "precession_nutation") == 0) {
        run_precession_nutation();
    } else if (strcmp(experiment, "precession_nutation_perf") == 0) {
        run_precession_nutation_perf();
    } else if (strcmp(experiment, "inv_precession_nutation") == 0) {
        run_inv_precession_nutation();
    } else if (strcmp(experiment, "inv_precession_nutation_perf") == 0) {
        run_inv_precession_nutation_perf();
    } else if (strcmp(experiment, "inv_icrs_ecl_tod") == 0) {
        run_inv_icrs_ecl_tod();
    } else if (strcmp(experiment, "inv_icrs_ecl_tod_perf") == 0) {
        run_inv_icrs_ecl_tod_perf();
    } else if (strcmp(experiment, "inv_equ_ecl") == 0) {
        run_inv_equ_ecl();
    } else if (strcmp(experiment, "inv_equ_ecl_perf") == 0) {
        run_inv_equ_ecl_perf();
    } else {
        fprintf(stderr, "Unknown experiment: %s\n", experiment);
        return 1;
    }

    return 0;
}
