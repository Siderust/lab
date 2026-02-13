#define _POSIX_C_SOURCE 199309L

/*
 * ERFA Adapter for the Siderust Lab
 *
 * Reads line-protocol from stdin, runs ERFA transformations,
 * writes JSON results to stdout.
 *
 * Protocol (one experiment per invocation):
 *   Line 1: experiment name
 *   Line 2: N (number of test cases)
 *   Lines 3..N+2: space-separated values depending on experiment
 *
 * Supported experiments:
 *   frame_rotation_bpn      — BPN matrix (GCRS→CIRS)
 *   gmst_era                — GMST & ERA
 *   equ_ecl                 — Equatorial ↔ Ecliptic
 *   equ_horizontal          — Equatorial → Horizontal (AltAz)
 *   solar_position          — Sun geocentric RA/Dec
 *   lunar_position          — Moon geocentric RA/Dec (simplified Meeus)
 *   kepler_solver           — Kepler equation M→E→ν
 *   frame_rotation_bpn_perf — BPN performance timing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "erfa.h"

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

/* Multiply 3x3 matrix by 3-vector: out = m * v */
static void mv3(double m[3][3], const double v[3], double out[3]) {
    for (int i = 0; i < 3; i++) {
        out[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2];
    }
}

/* Angular separation between two unit vectors (radians) */
static double ang_sep(const double a[3], const double b[3]) {
    double dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    if (dot >  1.0) dot =  1.0;
    if (dot < -1.0) dot = -1.0;
    return acos(dot);
}

/* Normalize angle to [0, 2π) */
static double normalize_angle(double a) {
    a = fmod(a, 2.0 * M_PI);
    if (a < 0.0) a += 2.0 * M_PI;
    return a;
}

/* ------------------------------------------------------------------ */
/* Experiment: frame_rotation_bpn                                      */
/* Computes the Bias-Precession-Nutation matrix using IAU 2006/2000A   */
/* and applies it to input direction vectors.                          */
/* ------------------------------------------------------------------ */

static void run_frame_rotation_bpn(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    /* Header */
    printf("{\"experiment\":\"frame_rotation_bpn\",\"library\":\"erfa\",");
    printf("\"model\":\"IAU_2006_2000A\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt, vx, vy, vz;
        if (scanf("%lf %lf %lf %lf", &jd_tt, &vx, &vy, &vz) != 4) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }

        /* Split JD into two-part form for best precision */
        double date1 = 2451545.0;           /* J2000.0 */
        double date2 = jd_tt - 2451545.0;

        /* Compute BPN matrix (GCRS → CIRS), IAU 2006/2000A */
        double rnpb[3][3];
        eraPnm06a(date1, date2, rnpb);

        /* Apply BPN to input direction vector */
        double vin[3] = {vx, vy, vz};
        normalize3(vin);
        double vout[3];
        mv3(rnpb, vin, vout);
        normalize3(vout);

        /* Also compute inverse (CIRS → GCRS) for closure test */
        double vinv[3];
        /* Transpose of rnpb */
        double rnpb_t[3][3];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                rnpb_t[r][c] = rnpb[c][r];
        mv3(rnpb_t, vout, vinv);
        normalize3(vinv);

        double closure_rad = ang_sep(vin, vinv);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"input\":[%.17e,%.17e,%.17e],", jd_tt, vin[0], vin[1], vin[2]);
        printf("\"output\":[%.17e,%.17e,%.17e],", vout[0], vout[1], vout[2]);
        printf("\"closure_rad\":%.17e,", closure_rad);
        printf("\"matrix\":[[%.17e,%.17e,%.17e],[%.17e,%.17e,%.17e],[%.17e,%.17e,%.17e]]}",
            rnpb[0][0], rnpb[0][1], rnpb[0][2],
            rnpb[1][0], rnpb[1][1], rnpb[1][2],
            rnpb[2][0], rnpb[2][1], rnpb[2][2]);
    }

    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: gmst_era                                                */
/* Computes GMST (IAU 2006) and ERA (IAU 2000) at given epochs.        */
/* ------------------------------------------------------------------ */

static void run_gmst_era(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"gmst_era\",\"library\":\"erfa\",");
    printf("\"model\":\"GMST=IAU2006, ERA=IAU2000\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_ut1, jd_tt;
        if (scanf("%lf %lf", &jd_ut1, &jd_tt) != 2) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }

        double ut1_hi = 2451545.0;
        double ut1_lo = jd_ut1 - 2451545.0;
        double tt_hi  = 2451545.0;
        double tt_lo  = jd_tt - 2451545.0;

        double gmst = eraGmst06(ut1_hi, ut1_lo, tt_hi, tt_lo);
        double era  = eraEra00(ut1_hi, ut1_lo);
        double gast = eraGst06a(ut1_hi, ut1_lo, tt_hi, tt_lo);

        if (i > 0) printf(",\n");
        printf("{\"jd_ut1\":%.15f,\"jd_tt\":%.15f,", jd_ut1, jd_tt);
        printf("\"gmst_rad\":%.17e,\"era_rad\":%.17e,\"gast_rad\":%.17e}", gmst, era, gast);
    }

    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: equ_ecl                                                 */
/* Equatorial (ICRS RA/Dec) ↔ Ecliptic (lon/lat), IAU 2006             */
/* Input per line: jd_tt  ra_rad  dec_rad                              */
/* ------------------------------------------------------------------ */

static void run_equ_ecl(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"equ_ecl\",\"library\":\"erfa\",");
    printf("\"model\":\"IAU_2006_ecliptic\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt, ra_rad, dec_rad;
        if (scanf("%lf %lf %lf", &jd_tt, &ra_rad, &dec_rad) != 3) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double date1 = 2451545.0, date2 = jd_tt - 2451545.0;

        /* Forward: Equatorial → Ecliptic */
        double ecl_lon, ecl_lat;
        eraEqec06(date1, date2, ra_rad, dec_rad, &ecl_lon, &ecl_lat);

        /* Closure: Ecliptic → Equatorial */
        double ra_back, dec_back;
        eraEceq06(date1, date2, ecl_lon, ecl_lat, &ra_back, &dec_back);

        double v_in[3]   = { cos(dec_rad)*cos(ra_rad), cos(dec_rad)*sin(ra_rad), sin(dec_rad) };
        double v_back[3]  = { cos(dec_back)*cos(ra_back), cos(dec_back)*sin(ra_back), sin(dec_back) };
        double closure_rad = ang_sep(v_in, v_back);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,\"ra_rad\":%.17e,\"dec_rad\":%.17e,", jd_tt, ra_rad, dec_rad);
        printf("\"ecl_lon_rad\":%.17e,\"ecl_lat_rad\":%.17e,", ecl_lon, ecl_lat);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: equ_horizontal                                          */
/* Equatorial (RA/Dec) → Horizontal (Az/Alt) via GAST + eraHd2ae       */
/* Input per line: jd_ut1 jd_tt ra_rad dec_rad obs_lon_rad obs_lat_rad */
/* ------------------------------------------------------------------ */

static void run_equ_horizontal(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"equ_horizontal\",\"library\":\"erfa\",");
    printf("\"model\":\"eraHd2ae_GAST\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_ut1, jd_tt, ra_rad, dec_rad, obs_lon, obs_lat;
        if (scanf("%lf %lf %lf %lf %lf %lf", &jd_ut1, &jd_tt, &ra_rad, &dec_rad,
                  &obs_lon, &obs_lat) != 6) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double ut1_hi = 2451545.0, ut1_lo = jd_ut1 - 2451545.0;
        double tt_hi  = 2451545.0, tt_lo  = jd_tt  - 2451545.0;

        double gast = eraGst06a(ut1_hi, ut1_lo, tt_hi, tt_lo);
        double last = gast + obs_lon;
        double ha   = last - ra_rad;

        double az, alt;
        eraHd2ae(ha, dec_rad, obs_lat, &az, &alt);

        /* Closure */
        double ha_back, dec_back;
        eraAe2hd(az, alt, obs_lat, &ha_back, &dec_back);
        double ra_back = last - ha_back;
        double v_in[3]  = { cos(dec_rad)*cos(ra_rad), cos(dec_rad)*sin(ra_rad), sin(dec_rad) };
        double v_bk[3]  = { cos(dec_back)*cos(ra_back), cos(dec_back)*sin(ra_back), sin(dec_back) };
        double closure_rad = ang_sep(v_in, v_bk);

        if (i > 0) printf(",\n");
        printf("{\"jd_ut1\":%.15f,\"jd_tt\":%.15f,", jd_ut1, jd_tt);
        printf("\"ra_rad\":%.17e,\"dec_rad\":%.17e,", ra_rad, dec_rad);
        printf("\"obs_lon_rad\":%.17e,\"obs_lat_rad\":%.17e,", obs_lon, obs_lat);
        printf("\"az_rad\":%.17e,\"alt_rad\":%.17e,", az, alt);
        printf("\"closure_rad\":%.17e}", closure_rad);
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: solar_position                                          */
/* Apparent geocentric Sun RA/Dec using ERFA epv00                     */
/* (epv00 returns BCRS-aligned equatorial vectors, not ecliptic)       */
/* Input per line: jd_tt                                               */
/* ------------------------------------------------------------------ */

static void run_solar_position(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"solar_position\",\"library\":\"erfa\",");
    printf("\"model\":\"ERFA_epv00_analytic\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt;
        if (scanf("%lf", &jd_tt) != 1) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double date1 = 2451545.0, date2 = jd_tt - 2451545.0;

        double pvh[2][3], pvb[2][3];
        eraEpv00(date1, date2, pvh, pvb);

        /* Sun geocentric ≈ –Earth heliocentric (BCRS equatorial) */
        double sx = -pvh[0][0], sy = -pvh[0][1], sz = -pvh[0][2];
        double dist_au = sqrt(sx*sx + sy*sy + sz*sz);
        double ra  = normalize_angle(atan2(sy, sx));
        double dec = asin(sz / dist_au);

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,", jd_tt);
        printf("\"ra_rad\":%.17e,\"dec_rad\":%.17e,\"dist_au\":%.17e}", ra, dec, dist_au);
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: lunar_position                                          */
/* Geocentric Moon RA/Dec — simplified Meeus Ch.47 (major terms).      */
/* ERFA has no dedicated Moon function; this is approximate (~10').     */
/* Input per line: jd_tt                                               */
/* ------------------------------------------------------------------ */

static void run_lunar_position(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"lunar_position\",\"library\":\"erfa\",");
    printf("\"model\":\"Meeus_Ch47_simplified\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double jd_tt;
        if (scanf("%lf", &jd_tt) != 1) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }
        double date1 = 2451545.0, date2 = jd_tt - 2451545.0;
        double T = date2 / 36525.0;

        /* Mean elements (degrees) — Meeus Ch 47 */
        double Lp = fmod(218.3164477 + 481267.88123421*T
                    - 0.0015786*T*T + T*T*T/538841.0
                    - T*T*T*T/65194000.0, 360.0);
        double D  = fmod(297.8501921 + 445267.1114034*T
                    - 0.0018819*T*T + T*T*T/545868.0
                    - T*T*T*T/113065000.0, 360.0);
        double M  = fmod(357.5291092 + 35999.0502909*T
                    - 0.0001536*T*T + T*T*T/24490000.0, 360.0);
        double Mp = fmod(134.9633964 + 477198.8675055*T
                    + 0.0087414*T*T + T*T*T/69699.0
                    - T*T*T*T/14712000.0, 360.0);
        double F  = fmod(93.2720950 + 483202.0175233*T
                    - 0.0036539*T*T - T*T*T/3526000.0
                    + T*T*T*T/863310000.0, 360.0);

        double Lp_r = Lp*M_PI/180.0, D_r = D*M_PI/180.0;
        double M_r = M*M_PI/180.0, Mp_r = Mp*M_PI/180.0, F_r = F*M_PI/180.0;

        /* Major longitude terms (×1e-6 deg) */
        double sum_l = 6288774.0*sin(Mp_r)
                     + 1274027.0*sin(2.0*D_r - Mp_r)
                     +  658314.0*sin(2.0*D_r)
                     +  213618.0*sin(2.0*Mp_r)
                     -  185116.0*sin(M_r)
                     -  114332.0*sin(2.0*F_r);

        /* Major latitude terms */
        double sum_b = 5128122.0*sin(F_r)
                     +  280602.0*sin(Mp_r + F_r)
                     +  277693.0*sin(Mp_r - F_r)
                     +  173237.0*sin(2.0*D_r - F_r);

        /* Major distance terms (km) */
        double sum_r = -20905355.0*cos(Mp_r)
                     -  3699111.0*cos(2.0*D_r - Mp_r)
                     -  2955968.0*cos(2.0*D_r)
                     -   569925.0*cos(2.0*Mp_r);

        double ecl_lon = (Lp + sum_l / 1000000.0) * M_PI / 180.0;
        double ecl_lat = (sum_b / 1000000.0) * M_PI / 180.0;
        double dist_km = 385000.56 + sum_r / 1000.0;

        /* Ecliptic → equatorial using mean obliquity */
        double eps = eraObl06(date1, date2);  /* radians */
        double ce = cos(eps), se = sin(eps);
        double ra  = normalize_angle(atan2(sin(ecl_lon)*ce - tan(ecl_lat)*se, cos(ecl_lon)));
        double dec = asin(sin(ecl_lat)*ce + cos(ecl_lat)*se*sin(ecl_lon));

        if (i > 0) printf(",\n");
        printf("{\"jd_tt\":%.15f,", jd_tt);
        printf("\"ra_rad\":%.17e,\"dec_rad\":%.17e,\"dist_km\":%.17e}", ra, dec, dist_km);
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Experiment: kepler_solver                                           */
/* Kepler equation M→E→ν self-consistency                              */
/* Input per line: M_rad  e                                            */
/* ------------------------------------------------------------------ */

static void run_kepler_solver(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    printf("{\"experiment\":\"kepler_solver\",\"library\":\"erfa\",");
    printf("\"model\":\"Newton_Raphson\",");
    printf("\"count\":%d,\"cases\":[\n", n);

    for (int i = 0; i < n; i++) {
        double M_rad, e;
        if (scanf("%lf %lf", &M_rad, &e) != 2) {
            fprintf(stderr, "bad input line %d\n", i); exit(1);
        }

        /* Newton-Raphson: solve M = E - e·sin(E) */
        double E = M_rad;
        int converged = 1, iters;
        for (iters = 0; iters < 100; iters++) {
            double f  = E - e * sin(E) - M_rad;
            double fp = 1.0 - e * cos(E);
            if (fabs(fp) < 1e-30) { converged = 0; break; }
            double dE = f / fp;
            E -= dE;
            if (fabs(dE) < 1e-15) break;
        }
        if (iters >= 100) converged = 0;

        /* True anomaly */
        double nu = 2.0 * atan2(sqrt(1.0+e)*sin(E/2.0), sqrt(1.0-e)*cos(E/2.0));

        /* Self-consistency residual */
        double residual = fabs(E - e*sin(E) - M_rad);

        if (i > 0) printf(",\n");
        printf("{\"M_rad\":%.17e,\"e\":%.17e,", M_rad, e);
        printf("\"E_rad\":%.17e,\"nu_rad\":%.17e,", E, nu);
        printf("\"residual_rad\":%.17e,\"iters\":%d,\"converged\":%s}",
               residual, iters, converged ? "true" : "false");
    }
    printf("\n]}\n");
}

/* ------------------------------------------------------------------ */
/* Performance timing helper                                           */
/* ------------------------------------------------------------------ */

static void run_frame_rotation_bpn_perf(void) {
    int n;
    if (scanf("%d", &n) != 1) { fprintf(stderr, "bad N\n"); exit(1); }

    /* Read all inputs */
    double *jds = malloc(n * sizeof(double));
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
        double rnpb[3][3];
        eraPnm06a(2451545.0, jds[i] - 2451545.0, rnpb);
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double vout[3];
    for (int i = 0; i < n; i++) {
        double rnpb[3][3];
        eraPnm06a(2451545.0, jds[i] - 2451545.0, rnpb);
        double vin[3] = {vecs[3*i], vecs[3*i+1], vecs[3*i+2]};
        normalize3(vin);
        mv3(rnpb, vin, vout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"frame_rotation_bpn_perf\",\"library\":\"erfa\",");
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
        double jd_ut1, jd_tt;
        if (scanf("%lf %lf", &jd_ut1, &jd_tt) != 2) {
            fprintf(stderr, "bad input line %d\n", i);
            exit(1);
        }
        jd_ut1_arr[i] = jd_ut1;
        jd_tt_arr[i] = jd_tt;
    }

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double gmst = eraGmst06(2451545.0, jd_ut1_arr[i] - 2451545.0,
                                2451545.0, jd_tt_arr[i] - 2451545.0);
        (void)gmst;
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double gmst = eraGmst06(2451545.0, jd_ut1_arr[i] - 2451545.0,
                                2451545.0, jd_tt_arr[i] - 2451545.0);
        sink += gmst;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"gmst_era_perf\",\"library\":\"erfa\",");
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

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double rm[3][3];
        eraEcm06(2451545.0, jds[i] - 2451545.0, rm);
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double rm[3][3];
        eraEcm06(2451545.0, jds[i] - 2451545.0, rm);
        sink += rm[0][0];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"equ_ecl_perf\",\"library\":\"erfa\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(jds);
    free(ras);
    free(decs);
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

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double jd_ut1 = params[6*i];
        double jd_tt = params[6*i+1];
        double ra = params[6*i+2];
        double dec = params[6*i+3];
        double lon = params[6*i+4];
        double lat = params[6*i+5];

        double gmst = eraGmst06(2451545.0, jd_ut1 - 2451545.0,
                                2451545.0, jd_tt - 2451545.0);
        double ha = normalize_angle(gmst + lon - ra);
        double az, alt;
        eraHd2ae(ha, dec, lat, &az, &alt);
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double jd_ut1 = params[6*i];
        double jd_tt = params[6*i+1];
        double ra = params[6*i+2];
        double dec = params[6*i+3];
        double lon = params[6*i+4];
        double lat = params[6*i+5];

        double gmst = eraGmst06(2451545.0, jd_ut1 - 2451545.0,
                                2451545.0, jd_tt - 2451545.0);
        double ha = normalize_angle(gmst + lon - ra);
        double az, alt;
        eraHd2ae(ha, dec, lat, &az, &alt);
        sink += az;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"equ_horizontal_perf\",\"library\":\"erfa\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(params);
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

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double pvh[2][3], pvb[2][3];
        eraEpv00(2451545.0, jds[i] - 2451545.0, pvh, pvb);
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double pvh[2][3], pvb[2][3];
        eraEpv00(2451545.0, jds[i] - 2451545.0, pvh, pvb);
        sink += pvh[0][0];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"solar_position_perf\",\"library\":\"erfa\",");
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

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double rm[3][3], rp[3];
        eraMoon98(2451545.0, jds[i] - 2451545.0, rp);
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double rp[3];
        eraMoon98(2451545.0, jds[i] - 2451545.0, rp);
        sink += rp[0];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"lunar_position_perf\",\"library\":\"erfa\",");
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

    /* Warm-up */
    for (int i = 0; i < n && i < 100; i++) {
        double E = m_arr[i];
        for (int iter = 0; iter < 20; iter++) {
            double f = E - e_arr[i] * sin(E) - m_arr[i];
            double fp = 1.0 - e_arr[i] * cos(E);
            E -= f / fp;
            if (fabs(f) < 1e-12) break;
        }
    }

    /* Timed run */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double sink = 0.0;
    for (int i = 0; i < n; i++) {
        double E = m_arr[i];
        for (int iter = 0; iter < 20; iter++) {
            double f = E - e_arr[i] * sin(E) - m_arr[i];
            double fp = 1.0 - e_arr[i] * cos(E);
            E -= f / fp;
            if (fabs(f) < 1e-12) break;
        }
        sink += E;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double per_op_ns = elapsed_ns / n;

    printf("{\"experiment\":\"kepler_solver_perf\",\"library\":\"erfa\",");
    printf("\"count\":%d,\"total_ns\":%.0f,\"per_op_ns\":%.1f,", n, elapsed_ns, per_op_ns);
    printf("\"throughput_ops_s\":%.0f,\"_sink\":%.17e}\n",
           (double)n / (elapsed_ns * 1e-9), sink);

    free(m_arr);
    free(e_arr);
}
/* ------------------------------------------------------------------ */
/* Main dispatcher                                                     */
/* ------------------------------------------------------------------ */

int main(void) {
    char experiment[256];
    if (scanf("%255s", experiment) != 1) {
        fprintf(stderr, "Usage: echo 'experiment_name\\nN\\n...' | ./erfa_adapter\n");
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
    } else {
        fprintf(stderr, "Unknown experiment: %s\n", experiment);
        return 1;
    }

    return 0;
}
