#define _POSIX_C_SOURCE 199309L

/*
 * ERFA Adapter for the Siderust Lab
 *
 * Reads a JSON-ish line-protocol from stdin, runs ERFA transformations,
 * writes JSON results to stdout.
 *
 * Protocol (one experiment per invocation):
 *   Line 1: experiment name (e.g. "frame_rotation_bpn")
 *   Line 2: N (number of test cases)
 *   Lines 3..N+2: space-separated values depending on experiment
 *
 * Output: JSON to stdout.
 *
 * Supported experiments:
 *   frame_rotation_bpn  — Bias-Precession-Nutation matrix (GCRS→CIRS)
 *     Input per line: jd_tt  vx vy vz
 *     Output: transformed direction + matrix elements
 *
 *   gmst_era — Greenwich Mean Sidereal Time and Earth Rotation Angle
 *     Input per line: jd_ut1  jd_tt
 *     Output: GMST (rad), ERA (rad)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "erfa.h"

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
    } else if (strcmp(experiment, "frame_rotation_bpn_perf") == 0) {
        run_frame_rotation_bpn_perf();
    } else {
        fprintf(stderr, "Unknown experiment: %s\n", experiment);
        return 1;
    }

    return 0;
}
