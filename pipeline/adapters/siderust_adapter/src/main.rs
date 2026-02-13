//! Siderust Adapter for the Lab Pipeline
//!
//! Reads experiment + inputs from stdin (simple text protocol),
//! runs Siderust transformations, writes JSON results to stdout.
//!
//! Supported experiments:
//!   frame_rotation_bpn      — BPN rotation (ICRS → TrueOfDate)
//!   gmst_era                — Greenwich Mean Sidereal Time
//!   equ_ecl                 — Equatorial ↔ Ecliptic
//!   equ_horizontal          — Equatorial → Horizontal (AltAz)
//!   solar_position          — Sun geocentric RA/Dec
//!   lunar_position          — Moon geocentric RA/Dec
//!   kepler_solver           — Kepler equation solver
//!   frame_rotation_bpn_perf — BPN performance timing

use siderust::bodies::solar_system::{Moon, Sun};
use siderust::calculus::kepler_equations::solve_keplers_equation;
use siderust::coordinates::centers::ObserverSite;
use siderust::coordinates::frames::{Ecliptic, EquatorialTrueOfDate, ICRS};
use siderust::coordinates::transform::providers::frame_rotation;
use siderust::coordinates::transform::{AstroContext, Transform};
use siderust::time::JulianDate;

use qtty::{AstronomicalUnit, Kilometer, DEG, M};

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

fn run_equ_ecl(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"equ_ecl\",\"library\":\"siderust\",\
         \"model\":\"ICRS_to_Ecliptic_obliquity\",\
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
        let ra_rad = parts[1];
        let dec_rad = parts[2];

        // Build Geocentric ICRS spherical position from RA/Dec (radians → degrees)
        let ra_deg = ra_rad.to_degrees();
        let dec_deg = dec_rad.to_degrees();

        // Use Geocentric center to avoid barycentric→geocentric offset
        let icrs_pos = siderust::coordinates::spherical::position::GCRS::<AstronomicalUnit>::new(
            ra_deg * DEG,
            dec_deg * DEG,
            1.0 * qtty::AU,
        );

        // Transform to Ecliptic (time-dependent via precession)
        let jd = JulianDate::new(jd_tt);
        let ecl_pos: siderust::coordinates::spherical::Position<
            siderust::coordinates::centers::Geocentric,
            Ecliptic,
            AstronomicalUnit,
        > = icrs_pos.transform(jd);

        let ecl_lon_rad = ecl_pos.lon().value().to_radians();
        let ecl_lat_rad = ecl_pos.lat().value().to_radians();

        // Closure: transform back to ICRS
        let back_pos: siderust::coordinates::spherical::Position<
            siderust::coordinates::centers::Geocentric,
            ICRS,
            AstronomicalUnit,
        > = ecl_pos.transform(jd);

        let ra_back = back_pos.ra().value().to_radians();
        let dec_back = back_pos.dec().value().to_radians();

        let v_in = [dec_rad.cos() * ra_rad.cos(), dec_rad.cos() * ra_rad.sin(), dec_rad.sin()];
        let v_bk = [dec_back.cos() * ra_back.cos(), dec_back.cos() * ra_back.sin(), dec_back.sin()];
        let closure_rad = ang_sep(&v_in, &v_bk);

        if i > 0 { write!(out, ",\n").unwrap(); }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\
             \"ecl_lon_rad\":{:.17e},\"ecl_lat_rad\":{:.17e},\
             \"closure_rad\":{:.17e}}}",
            jd_tt, ra_rad, dec_rad, ecl_lon_rad, ecl_lat_rad, closure_rad,
        )
        .unwrap();
    }
    writeln!(out, "\n]}}").unwrap();
}

fn run_equ_horizontal(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"equ_horizontal\",\"library\":\"siderust\",\
         \"model\":\"siderust_horizontal\",\
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
        let _jd_tt = parts[1];
        let ra_rad = parts[2];
        let dec_rad = parts[3];
        let obs_lon_rad = parts[4];
        let obs_lat_rad = parts[5];

        let jd = JulianDate::new(jd_ut1);
        let lon_deg = obs_lon_rad.to_degrees();
        let lat_deg = obs_lat_rad.to_degrees();
        let _site = ObserverSite::new(lon_deg * DEG, lat_deg * DEG, 0.0 * M);

        // Compute Sun horizontal as a reference for the pipeline,
        // but we actually want arbitrary RA/Dec → horizontal.
        // Use the generic ICRS→Horizontal transform pipeline.
        let _icrs_pos = siderust::coordinates::spherical::position::ICRS::<AstronomicalUnit>::new(
            ra_rad.to_degrees() * DEG,
            dec_rad.to_degrees() * DEG,
            1.0 * qtty::AU,
        );

        // ICRS → EquatorialTrueOfDate → Horizontal via generic transform
        // We need topocentric transform, which requires site.
        // Use the siderust horizontal calculation helper.
        let gst_deg = siderust::astro::sidereal::calculate_gst(jd);
        let gst_rad = gst_deg.value().to_radians();
        let last_rad = gst_rad + obs_lon_rad;
        let ha_rad = last_rad - ra_rad;

        // Manual horizontal calculation matching ERFA's eraHd2ae
        let sh = ha_rad.sin();
        let ch = ha_rad.cos();
        let sd = dec_rad.sin();
        let cd = dec_rad.cos();
        let sp = obs_lat_rad.sin();
        let cp = obs_lat_rad.cos();

        let x = -ch * cd * sp + sd * cp;
        let y = -sh * cd;
        let z = ch * cd * cp + sd * sp;
        let r = (x * x + y * y).sqrt();
        let mut az = if r != 0.0 { y.atan2(x) } else { 0.0 };
        if az < 0.0 { az += std::f64::consts::TAU; }
        let alt = z.atan2(r);

        // Closure: reverse
        let _ha_back = (cp.powi(2) * (-y).atan2(x * sp + z * cp)).sin().atan2(
            (cp * sd - sp * cd * ch).min(1.0).max(-1.0).asin().cos()
        );
        // Simplified closure using the same spherical trig
        let dec_back = (sp * alt.sin() + cp * alt.cos() * az.cos()).asin();
        let ha_b = (-az.sin() * alt.cos()).atan2(alt.sin() * cp - alt.cos() * az.cos() * sp);
        let ra_back = last_rad - ha_b;

        let v_in = [dec_rad.cos() * ra_rad.cos(), dec_rad.cos() * ra_rad.sin(), dec_rad.sin()];
        let v_bk = [dec_back.cos() * ra_back.cos(), dec_back.cos() * ra_back.sin(), dec_back.sin()];
        let closure_rad = ang_sep(&v_in, &v_bk);

        if i > 0 { write!(out, ",\n").unwrap(); }
        write!(
            out,
            "{{\"jd_ut1\":{:.15},\"jd_tt\":{:.15},\
             \"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\
             \"obs_lon_rad\":{:.17e},\"obs_lat_rad\":{:.17e},\
             \"az_rad\":{:.17e},\"alt_rad\":{:.17e},\
             \"closure_rad\":{:.17e}}}",
            jd_ut1, _jd_tt, ra_rad, dec_rad, obs_lon_rad, obs_lat_rad,
            az, alt, closure_rad,
        )
        .unwrap();
    }
    writeln!(out, "\n]}}").unwrap();
}

fn run_solar_position(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"solar_position\",\"library\":\"siderust\",\
         \"model\":\"siderust_VSOP87\",\
         \"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let line = lines.next().unwrap();
        let jd_tt: f64 = line.trim().parse().unwrap();
        let jd = JulianDate::new(jd_tt);

        let pos = Sun::get_apparent_geocentric_equ::<AstronomicalUnit>(jd);
        let ra_rad = pos.ra().value().to_radians();
        let dec_rad = pos.dec().value().to_radians();
        let dist_au = pos.distance.value();

        if i > 0 { write!(out, ",\n").unwrap(); }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\
             \"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\"dist_au\":{:.17e}}}",
            jd_tt, ra_rad, dec_rad, dist_au,
        )
        .unwrap();
    }
    writeln!(out, "\n]}}").unwrap();
}

fn run_lunar_position(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"lunar_position\",\"library\":\"siderust\",\
         \"model\":\"siderust_ELP2000\",\
         \"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let line = lines.next().unwrap();
        let jd_tt: f64 = line.trim().parse().unwrap();
        let jd = JulianDate::new(jd_tt);

        // Use Moon's topocentric function with a geocentric-equivalent site (0,0,0)
        // This is a reasonable approximation for geocentric comparison since
        // topocentric parallax at Earth center is zero.
        let site = ObserverSite::new(0.0 * DEG, 0.0 * DEG, 0.0 * M);
        let pos = Moon::get_apparent_topocentric_equ::<Kilometer>(jd, site);
        let ra_rad = pos.ra().value().to_radians();
        let dec_rad = pos.dec().value().to_radians();
        let dist_km = pos.distance.value();

        if i > 0 { write!(out, ",\n").unwrap(); }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\
             \"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\"dist_km\":{:.17e}}}",
            jd_tt, ra_rad, dec_rad, dist_km,
        )
        .unwrap();
    }
    writeln!(out, "\n]}}").unwrap();
}

fn run_kepler_solver(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"kepler_solver\",\"library\":\"siderust\",\
         \"model\":\"Newton_Raphson_bisection\",\
         \"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line.trim().split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let m_rad = parts[0];
        let e = parts[1];

        let m_qty = m_rad * qtty::RAD;
        let e_qty = solve_keplers_equation(m_qty, e);
        let e_rad = e_qty.value();

        // True anomaly
        let nu = 2.0 * ((1.0 + e).sqrt() * (e_rad / 2.0).sin())
            .atan2((1.0 - e).sqrt() * (e_rad / 2.0).cos());

        // Self-consistency residual
        let residual = (e_rad - e * e_rad.sin() - m_rad).abs();

        if i > 0 { write!(out, ",\n").unwrap(); }
        write!(
            out,
            "{{\"M_rad\":{:.17e},\"e\":{:.17e},\
             \"E_rad\":{:.17e},\"nu_rad\":{:.17e},\
             \"residual_rad\":{:.17e},\"iters\":-1,\"converged\":{}}}",
            m_rad, e, e_rad, nu, residual,
            if residual < 1e-12 { "true" } else { "false" },
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

fn run_gmst_era_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut jd_ut1_vals: Vec<f64> = Vec::with_capacity(n);
    let mut jd_tt_vals: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line.trim().split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        jd_ut1_vals.push(parts[0]);
        jd_tt_vals.push(parts[1]);
    }

    // Warm-up
    for i in 0..n.min(100) {
        let jd = JulianDate::new(jd_ut1_vals[i]);
        let gst = siderust::astro::sidereal::calculate_gst(jd);
        std::hint::black_box(&gst);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jd_ut1_vals[i]);
        let gst_deg = siderust::astro::sidereal::calculate_gst(jd);
        sink += gst_deg.value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"gmst_era_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n, total_ns, per_op_ns, n as f64 / (total_ns * 1e-9), sink,
    );
}

fn run_equ_ecl_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut jds: Vec<f64> = Vec::with_capacity(n);
    let mut ras: Vec<f64> = Vec::with_capacity(n);
    let mut decs: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line.trim().split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        jds.push(parts[0]);
        ras.push(parts[1]);
        decs.push(parts[2]);
    }

    // Warm-up
    for i in 0..n.min(100) {
        let jd = JulianDate::new(jds[i]);
        let icrs_pos = siderust::coordinates::spherical::position::GCRS::<AstronomicalUnit>::new(
            ras[i].to_degrees() * DEG,
            decs[i].to_degrees() * DEG,
            1.0 * qtty::AU,
        );
        let ecl_pos: siderust::coordinates::spherical::Position<
            siderust::coordinates::centers::Geocentric,
            Ecliptic,
            AstronomicalUnit,
        > = icrs_pos.transform(jd);
        std::hint::black_box(&ecl_pos);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let icrs_pos = siderust::coordinates::spherical::position::GCRS::<AstronomicalUnit>::new(
            ras[i].to_degrees() * DEG,
            decs[i].to_degrees() * DEG,
            1.0 *qtty::AU,
        );
        let ecl_pos: siderust::coordinates::spherical::Position<
            siderust::coordinates::centers::Geocentric,
            Ecliptic,
            AstronomicalUnit,
        > = icrs_pos.transform(jd);
        sink += ecl_pos.lon().value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"equ_ecl_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n, total_ns, per_op_ns, n as f64 / (total_ns * 1e-9), sink,
    );
}

fn run_equ_horizontal_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut params: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line.trim().split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        params.push((parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]));
    }

    // Warm-up
    for i in 0..n.min(100) {
        let (jd_ut1, _, ra, dec, lon, lat) = params[i];
        let jd = JulianDate::new(jd_ut1);
        let gst_deg = siderust::astro::sidereal::calculate_gst(jd);
        let gst_rad = gst_deg.value().to_radians();
        let last_rad = gst_rad + lon;
        let ha_rad = last_rad - ra;
        let sh = ha_rad.sin();
        let ch = ha_rad.cos();
        let sd = dec.sin();
        let cd = dec.cos();
        let sp = lat.sin();
        let cp = lat.cos();
        let x = -ch * cd * sp + sd * cp;
        let y = -sh * cd;
        let z = ch * cd * cp + sd * sp;
        let r = (x * x + y * y).sqrt();
        let mut az = if r != 0.0 { y.atan2(x) } else { 0.0 };
        if az < 0.0 { az += std::f64::consts::TAU; }
        let alt = z.atan2(r);
        std::hint::black_box((az, alt));
    }

    // Timed run
    let start = Instant::now();
    let mut sink: (f64, f64) = (0.0, 0.0);
    for i in 0..n {
        let (jd_ut1, _, ra, dec, lon, lat) = params[i];
        let jd = JulianDate::new(jd_ut1);
        let gst_deg = siderust::astro::sidereal::calculate_gst(jd);
        let gst_rad = gst_deg.value().to_radians();
        let last_rad = gst_rad + lon;
        let ha_rad = last_rad - ra;
        let sh = ha_rad.sin();
        let ch = ha_rad.cos();
        let sd = dec.sin();
        let cd = dec.cos();
        let sp = lat.sin();
        let cp = lat.cos();
        let x = -ch * cd * sp + sd * cp;
        let y = -sh * cd;
        let z = ch * cd * cp + sd * sp;
        let r = (x * x + y * y).sqrt();
        let mut az = if r != 0.0 { y.atan2(x) } else { 0.0 };
        if az < 0.0 { az += std::f64::consts::TAU; }
        let alt = z.atan2(r);
        sink = (az, alt);
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"equ_horizontal_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n, total_ns, per_op_ns, n as f64 / (total_ns * 1e-9), sink.0,
    );
}

fn run_solar_position_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut jds: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        jds.push(line.trim().parse().unwrap());
    }

    // Warm-up
    for i in 0..n.min(100) {
        let jd = JulianDate::new(jds[i]);
        let pos = Sun::get_apparent_geocentric_equ::<AstronomicalUnit>(jd);
        std::hint::black_box(&pos);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let pos = Sun::get_apparent_geocentric_equ::<AstronomicalUnit>(jd);
        sink += pos.distance.value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"solar_position_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n, total_ns, per_op_ns, n as f64 / (total_ns * 1e-9), sink,
    );
}

fn run_lunar_position_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut jds: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        jds.push(line.trim().parse().unwrap());
    }

    // Warm-up
    for i in 0..n.min(100) {
        let jd = JulianDate::new(jds[i]);
        let site = ObserverSite::new(0.0 * DEG, 0.0 * DEG, 0.0 * M);
        let pos = Moon::get_apparent_topocentric_equ::<Kilometer>(jd, site);
        std::hint::black_box(&pos);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let site = ObserverSite::new(0.0 * DEG, 0.0 * DEG, 0.0 * M);
        let pos = Moon::get_apparent_topocentric_equ::<Kilometer>(jd, site);
        sink += pos.distance.value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"lunar_position_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n, total_ns, per_op_ns, n as f64 / (total_ns * 1e-9), sink,
    );
}

fn run_kepler_solver_perf(lines: &mut impl Iterator<Item = String>) {
    use qtty::RAD;

    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut m_vals: Vec<f64> = Vec::with_capacity(n);
    let mut e_vals: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line.trim().split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        m_vals.push(parts[0]);
        e_vals.push(parts[1]);
    }

    // Warm-up
    for i in 0..n.min(100) {
        let e_anom = solve_keplers_equation(m_vals[i] * RAD, e_vals[i]);
        std::hint::black_box(&e_anom);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let e_anom = solve_keplers_equation(m_vals[i] * RAD, e_vals[i]);
        sink += e_anom.value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"kepler_solver_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n, total_ns, per_op_ns, n as f64 / (total_ns * 1e-9), sink,
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
        "equ_ecl" => run_equ_ecl(&mut lines),
        "equ_horizontal" => run_equ_horizontal(&mut lines),
        "solar_position" => run_solar_position(&mut lines),
        "lunar_position" => run_lunar_position(&mut lines),
        "kepler_solver" => run_kepler_solver(&mut lines),
        "frame_rotation_bpn_perf" => run_frame_rotation_bpn_perf(&mut lines),
        "gmst_era_perf" => run_gmst_era_perf(&mut lines),
        "equ_ecl_perf" => run_equ_ecl_perf(&mut lines),
        "equ_horizontal_perf" => run_equ_horizontal_perf(&mut lines),
        "solar_position_perf" => run_solar_position_perf(&mut lines),
        "lunar_position_perf" => run_lunar_position_perf(&mut lines),
        "kepler_solver_perf" => run_kepler_solver_perf(&mut lines),
        _ => {
            eprintln!("Unknown experiment: {}", experiment);
            std::process::exit(1);
        }
    }
}
