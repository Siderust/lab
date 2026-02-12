//! Siderust Adapter for the Lab Pipeline
//!
//! Reads experiment + inputs from stdin (simple text protocol),
//! runs Siderust transformations, writes JSON results to stdout.
//!
//! Supported experiments:
//!   frame_rotation_bpn — Bias + Precession + Nutation rotation (ICRS → TrueOfDate)
//!     Input per line: jd_tt  vx vy vz
//!     Output: transformed direction + matrix elements
//!
//!   gmst_era — Greenwich Mean Sidereal Time
//!     Input per line: jd_ut1  jd_tt (jd_tt ignored; siderust GST uses JD directly)
//!     Output: GMST (rad)

use siderust::coordinates::frames::{EquatorialTrueOfDate, ICRS};
use siderust::coordinates::transform::providers::frame_rotation;
use siderust::coordinates::transform::AstroContext;
use siderust::time::JulianDate;

use std::io::{self, BufRead, Write};
use std::time::Instant;

fn normalize3(v: &mut [f64; 3]) {
    let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if r > 0.0 {
        v[0] /= r;
        v[1] /= r;
        v[2] /= r;
    }
}

fn ang_sep(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    dot.clamp(-1.0, 1.0).acos()
}

/// Compute the full BPN matrix: ICRS → EquatorialTrueOfDate
/// This composes: frame_bias * precession * nutation
fn bpn_matrix(jd: JulianDate) -> [[f64; 3]; 3] {
    let ctx = AstroContext::default();
    let rot = frame_rotation::<ICRS, EquatorialTrueOfDate>(jd, &ctx);
    *rot.as_matrix()
}

fn mat_mul(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn run_frame_rotation_bpn(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"frame_rotation_bpn\",\"library\":\"siderust\",\
         \"model\":\"Meeus_precession+IAU1980_nutation+IERS2003_bias\",\
         \"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line.trim().split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let jd_tt = parts[0];
        let mut vin = [parts[1], parts[2], parts[3]];
        normalize3(&mut vin);

        let jd = JulianDate::new(jd_tt);
        let mat = bpn_matrix(jd);
        let mut vout = mat_mul(&mat, &vin);
        normalize3(&mut vout);

        // Closure test: inverse (transpose for rotation) then apply
        let mat_t = mat_transpose(&mat);
        let mut vinv = mat_mul(&mat_t, &vout);
        normalize3(&mut vinv);
        let closure_rad = ang_sep(&vin, &vinv);

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\"input\":[{:.17e},{:.17e},{:.17e}],\
             \"output\":[{:.17e},{:.17e},{:.17e}],\
             \"closure_rad\":{:.17e},\
             \"matrix\":[[{:.17e},{:.17e},{:.17e}],[{:.17e},{:.17e},{:.17e}],[{:.17e},{:.17e},{:.17e}]]}}",
            jd_tt, vin[0], vin[1], vin[2],
            vout[0], vout[1], vout[2],
            closure_rad,
            mat[0][0], mat[0][1], mat[0][2],
            mat[1][0], mat[1][1], mat[1][2],
            mat[2][0], mat[2][1], mat[2][2],
        )
        .unwrap();
    }

    writeln!(out, "\n]}}").unwrap();
}

fn run_gmst_era(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"gmst_era\",\"library\":\"siderust\",\
         \"model\":\"Siderust_GST_IAU2006poly\",\
         \"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line.trim().split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let jd_ut1 = parts[0];
        let jd_tt = parts[1];

        let jd = JulianDate::new(jd_ut1);

        // Siderust computes GST in degrees; convert to radians
        let gst_deg = siderust::astro::sidereal::calculate_gst(jd);
        let gst_rad = gst_deg.value().to_radians();

        // Siderust doesn't have a separate ERA function; ERA ≈ GST for comparison
        // ERA is defined as: θ = 2π(0.7790572732640 + 1.00273781191135448 × Du)
        // where Du = JD(UT1) − 2451545.0
        let du = jd_ut1 - 2451545.0;
        let era_rad = 2.0 * std::f64::consts::PI
            * (0.7790572732640 + 1.00273781191135448 * du);
        // Normalize to [0, 2π)
        let era_rad = era_rad.rem_euclid(2.0 * std::f64::consts::PI);

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_ut1\":{:.15},\"jd_tt\":{:.15},\
             \"gmst_rad\":{:.17e},\"era_rad\":{:.17e}}}",
            jd_ut1, jd_tt, gst_rad, era_rad,
        )
        .unwrap();
    }

    writeln!(out, "\n]}}").unwrap();
}

fn run_frame_rotation_bpn_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut jds: Vec<f64> = Vec::with_capacity(n);
    let mut vecs: Vec<[f64; 3]> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line.trim().split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        jds.push(parts[0]);
        let mut v = [parts[1], parts[2], parts[3]];
        normalize3(&mut v);
        vecs.push(v);
    }

    // Warm-up
    for i in 0..n.min(100) {
        let jd = JulianDate::new(jds[i]);
        let mat = bpn_matrix(jd);
        std::hint::black_box(&mat);
    }

    // Timed run
    let start = Instant::now();
    let mut sink = [0.0f64; 3];
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let mat = bpn_matrix(jd);
        let vout = mat_mul(&mat, &vecs[i]);
        sink = vout;
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"frame_rotation_bpn_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        per_op_ns,
        n as f64 / (total_ns * 1e-9),
        sink[0],
    );
}

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines().map(|l| l.unwrap());

    let experiment = lines.next().expect("expected experiment name");
    let experiment = experiment.trim();

    match experiment {
        "frame_rotation_bpn" => run_frame_rotation_bpn(&mut lines),
        "gmst_era" => run_gmst_era(&mut lines),
        "frame_rotation_bpn_perf" => run_frame_rotation_bpn_perf(&mut lines),
        _ => {
            eprintln!("Unknown experiment: {}", experiment);
            std::process::exit(1);
        }
    }
}
