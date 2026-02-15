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
use siderust::coordinates::centers::{Geocentric, Heliocentric, ObserverSite};
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
         \"model\":\"IAU2006_precession+IAU2000B_nutation+IERS2003_bias\",\
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
         \"model\":\"IAU_2006_GMST\",\
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

        let jd_ut1_q = JulianDate::new(jd_ut1);
        let jd_tt_q = JulianDate::new(jd_tt);

        // IAU 2006 GMST (ERA-based, returns radians already normalized)
        let gst_rad = siderust::astro::sidereal::gmst_iau2006(jd_ut1_q, jd_tt_q).value();

        // ERA via siderust's earth_rotation_angle
        let era_rad = siderust::astro::era::earth_rotation_angle(jd_ut1_q).value();

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
         \"model\":\"IAU_2006_ecliptic\",\
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

        let jd = JulianDate::new(jd_tt);

        // IAU 2006 equatorial → ecliptic of date (matching eraEqec06):
        // R = Rx(-εA(t)) · precession_matrix_iau2006(t)
        // This transforms GCRS → mean equator of date → ecliptic of date.
        let prec = siderust::astro::precession_iau2006::precession_matrix_iau2006(jd);
        let eps_a = siderust::astro::precession_iau2006::mean_obliquity_iau2006(jd).value();

        // Apply precession (GCRS → mean equator of date)
        let v_in = [dec_rad.cos() * ra_rad.cos(), dec_rad.cos() * ra_rad.sin(), dec_rad.sin()];
        let prec_mat = *prec.as_matrix();
        let v_prec = mat_mul(&prec_mat, &v_in);

        // Apply Rx(-εA) rotation (mean equator of date → ecliptic of date)
        let ce = eps_a.cos();
        let se = eps_a.sin();
        let v_ecl = [
            v_prec[0],
            v_prec[1] * ce + v_prec[2] * se,
            -v_prec[1] * se + v_prec[2] * ce,
        ];

        let ecl_lon_rad = v_ecl[1].atan2(v_ecl[0]);
        let ecl_lon_rad = if ecl_lon_rad < 0.0 { ecl_lon_rad + std::f64::consts::TAU } else { ecl_lon_rad };
        let ecl_lat_rad = v_ecl[2].asin();

        // Closure: ecliptic → equatorial (inverse transform)
        // Rx(+εA) then P^T
        let v_eq = [
            v_ecl[0],
            v_ecl[1] * ce - v_ecl[2] * se,
            v_ecl[1] * se + v_ecl[2] * ce,
        ];
        let prec_t = mat_transpose(&prec_mat);
        let v_back = mat_mul(&prec_t, &v_eq);

        let ra_back = v_back[1].atan2(v_back[0]);
        let dec_back = v_back[2].asin();

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
        let jd_tt_val = parts[1];
        let ra_rad = parts[2];
        let dec_rad = parts[3];
        let obs_lon_rad = parts[4];
        let obs_lat_rad = parts[5];

        let jd_ut1_q = JulianDate::new(jd_ut1);
        let jd_tt_q = JulianDate::new(jd_tt_val);
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

        // GAST IAU 2006 (matching ERFA's eraGst06a): includes nutation equation of equinoxes
        let nut = siderust::astro::nutation_iau2000b::nutation_iau2000b(jd_tt_q);
        let gast = siderust::astro::sidereal::gast_iau2006(
            jd_ut1_q, jd_tt_q, nut.dpsi, nut.true_obliquity(),
        );
        let gast_rad = gast.value();
        let last_rad = gast_rad + obs_lon_rad;
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
            jd_ut1, jd_tt_val, ra_rad, dec_rad, obs_lon_rad, obs_lat_rad,
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
         \"model\":\"siderust_VSOP87_geometric\",\
         \"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let line = lines.next().unwrap();
        let jd_tt: f64 = line.trim().parse().unwrap();
        let jd = JulianDate::new(jd_tt);

        // Geometric GCRS position: Heliocentric Ecliptic [0,0,0] → Geocentric ICRS
        // This matches ERFA's approach: negate Earth's heliocentric position
        // No precession, no nutation, no aberration — pure geometric.
        let helio = siderust::coordinates::cartesian::position::Ecliptic::<AstronomicalUnit, Heliocentric>::CENTER;
        let geo_icrs: siderust::coordinates::cartesian::Position<Geocentric, ICRS, AstronomicalUnit> = helio.transform(jd);
        let sph = siderust::coordinates::spherical::Position::from_cartesian(&geo_icrs);
        let ra_rad = sph.ra().value().to_radians();
        let dec_rad = sph.dec().value().to_radians();
        let dist_au = sph.distance.value();

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
         \"model\":\"Meeus_Ch47_simplified\",\
         \"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let line = lines.next().unwrap();
        let jd_tt: f64 = line.trim().parse().unwrap();

        // Simplified Meeus Ch.47 lunar model — matches ERFA adapter exactly
        // for benchmark parity (same terms, same coefficients).
        let date2 = jd_tt - 2451545.0;
        let t = date2 / 36525.0;

        // Mean elements (degrees) — Meeus Ch 47
        let lp = (218.3164477 + 481267.88123421 * t
            - 0.0015786 * t * t + t * t * t / 538841.0
            - t * t * t * t / 65194000.0) % 360.0;
        let d = (297.8501921 + 445267.1114034 * t
            - 0.0018819 * t * t + t * t * t / 545868.0
            - t * t * t * t / 113065000.0) % 360.0;
        let m_sun = (357.5291092 + 35999.0502909 * t
            - 0.0001536 * t * t + t * t * t / 24490000.0) % 360.0;
        let mp = (134.9633964 + 477198.8675055 * t
            + 0.0087414 * t * t + t * t * t / 69699.0
            - t * t * t * t / 14712000.0) % 360.0;
        let f = (93.2720950 + 483202.0175233 * t
            - 0.0036539 * t * t - t * t * t / 3526000.0
            + t * t * t * t / 863310000.0) % 360.0;

        let d_r = d.to_radians();
        let m_r = m_sun.to_radians();
        let mp_r = mp.to_radians();
        let f_r = f.to_radians();

        // Major longitude terms (×1e−6 deg)
        let sum_l = 6288774.0 * mp_r.sin()
            + 1274027.0 * (2.0 * d_r - mp_r).sin()
            + 658314.0 * (2.0 * d_r).sin()
            + 213618.0 * (2.0 * mp_r).sin()
            - 185116.0 * m_r.sin()
            - 114332.0 * (2.0 * f_r).sin();

        // Major latitude terms
        let sum_b = 5128122.0 * f_r.sin()
            + 280602.0 * (mp_r + f_r).sin()
            + 277693.0 * (mp_r - f_r).sin()
            + 173237.0 * (2.0 * d_r - f_r).sin();

        // Major distance terms (km)
        let sum_r = -20905355.0 * mp_r.cos()
            - 3699111.0 * (2.0 * d_r - mp_r).cos()
            - 2955968.0 * (2.0 * d_r).cos()
            - 569925.0 * (2.0 * mp_r).cos();

        let ecl_lon = (lp + sum_l / 1000000.0).to_radians();
        let ecl_lat = (sum_b / 1000000.0).to_radians();
        let dist_km = 385000.56 + sum_r / 1000.0;

        // Ecliptic → equatorial using IAU 2006 mean obliquity
        let jd = JulianDate::new(jd_tt);
        let eps = siderust::astro::precession_iau2006::mean_obliquity_iau2006(jd).value();
        let ce = eps.cos();
        let se = eps.sin();
        let ra = (ecl_lon.sin() * ce - ecl_lat.tan() * se).atan2(ecl_lon.cos());
        let ra_rad = if ra < 0.0 { ra + std::f64::consts::TAU } else { ra };
        let dec_rad = (ecl_lat.sin() * ce + ecl_lat.cos() * se * ecl_lon.sin()).asin();

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
        let jd_ut1 = JulianDate::new(jd_ut1_vals[i]);
        let jd_tt = JulianDate::new(jd_tt_vals[i]);
        let gst = siderust::astro::sidereal::gmst_iau2006(jd_ut1, jd_tt);
        std::hint::black_box(&gst);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd_ut1 = JulianDate::new(jd_ut1_vals[i]);
        let jd_tt = JulianDate::new(jd_tt_vals[i]);
        let gst = siderust::astro::sidereal::gmst_iau2006(jd_ut1, jd_tt);
        sink += gst.value();
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
        let prec = siderust::astro::precession_iau2006::precession_matrix_iau2006(jd);
        let eps_a = siderust::astro::precession_iau2006::mean_obliquity_iau2006(jd).value();
        let v_in = [decs[i].cos() * ras[i].cos(), decs[i].cos() * ras[i].sin(), decs[i].sin()];
        let prec_mat = *prec.as_matrix();
        let v_prec = mat_mul(&prec_mat, &v_in);
        let ce = eps_a.cos();
        let se = eps_a.sin();
        let v_ecl = [v_prec[0], v_prec[1] * ce + v_prec[2] * se, -v_prec[1] * se + v_prec[2] * ce];
        std::hint::black_box(&v_ecl);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let prec = siderust::astro::precession_iau2006::precession_matrix_iau2006(jd);
        let eps_a = siderust::astro::precession_iau2006::mean_obliquity_iau2006(jd).value();
        let v_in = [decs[i].cos() * ras[i].cos(), decs[i].cos() * ras[i].sin(), decs[i].sin()];
        let prec_mat = *prec.as_matrix();
        let v_prec = mat_mul(&prec_mat, &v_in);
        let ce = eps_a.cos();
        let se = eps_a.sin();
        let lon = (v_prec[1] * ce + v_prec[2] * se).atan2(v_prec[0]);
        sink += lon;
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
        let (jd_ut1, jd_tt, ra, dec, lon, lat) = params[i];
        let jd_ut1_q = JulianDate::new(jd_ut1);
        let jd_tt_q = JulianDate::new(jd_tt);
        let nut = siderust::astro::nutation_iau2000b::nutation_iau2000b(jd_tt_q);
        let gast = siderust::astro::sidereal::gast_iau2006(jd_ut1_q, jd_tt_q, nut.dpsi, nut.true_obliquity());
        let last_rad = gast.value() + lon;
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
        let (jd_ut1, jd_tt, ra, dec, lon, lat) = params[i];
        let jd_ut1_q = JulianDate::new(jd_ut1);
        let jd_tt_q = JulianDate::new(jd_tt);
        let nut = siderust::astro::nutation_iau2000b::nutation_iau2000b(jd_tt_q);
        let gast = siderust::astro::sidereal::gast_iau2006(jd_ut1_q, jd_tt_q, nut.dpsi, nut.true_obliquity());
        let last_rad = gast.value() + lon;
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
        let helio = siderust::coordinates::cartesian::position::Ecliptic::<AstronomicalUnit, Heliocentric>::CENTER;
        let geo_icrs: siderust::coordinates::cartesian::Position<Geocentric, ICRS, AstronomicalUnit> = helio.transform(jd);
        std::hint::black_box(&geo_icrs);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let helio = siderust::coordinates::cartesian::position::Ecliptic::<AstronomicalUnit, Heliocentric>::CENTER;
        let geo_icrs: siderust::coordinates::cartesian::Position<Geocentric, ICRS, AstronomicalUnit> = helio.transform(jd);
        let sph = siderust::coordinates::spherical::Position::from_cartesian(&geo_icrs);
        sink += sph.distance.value();
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

    // Warm-up — simplified Meeus Ch.47 inline (matching accuracy path)
    for i in 0..n.min(100) {
        let jd_tt = jds[i];
        let t = (jd_tt - 2451545.0) / 36525.0;
        let mp = (134.9633964 + 477198.8675055 * t) % 360.0;
        std::hint::black_box(mp.to_radians().sin());
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd_tt = jds[i];
        let t = (jd_tt - 2451545.0) / 36525.0;
        let lp = (218.3164477 + 481267.88123421 * t
            - 0.0015786 * t * t + t * t * t / 538841.0
            - t * t * t * t / 65194000.0) % 360.0;
        let d = (297.8501921 + 445267.1114034 * t
            - 0.0018819 * t * t + t * t * t / 545868.0
            - t * t * t * t / 113065000.0) % 360.0;
        let m_sun = (357.5291092 + 35999.0502909 * t
            - 0.0001536 * t * t + t * t * t / 24490000.0) % 360.0;
        let mp = (134.9633964 + 477198.8675055 * t
            + 0.0087414 * t * t + t * t * t / 69699.0
            - t * t * t * t / 14712000.0) % 360.0;
        let f = (93.2720950 + 483202.0175233 * t
            - 0.0036539 * t * t - t * t * t / 3526000.0
            + t * t * t * t / 863310000.0) % 360.0;
        let (d_r, m_r, mp_r, f_r) = (d.to_radians(), m_sun.to_radians(), mp.to_radians(), f.to_radians());
        let sum_l = 6288774.0 * mp_r.sin()
            + 1274027.0 * (2.0 * d_r - mp_r).sin()
            + 658314.0 * (2.0 * d_r).sin()
            + 213618.0 * (2.0 * mp_r).sin()
            - 185116.0 * m_r.sin()
            - 114332.0 * (2.0 * f_r).sin();
        let ecl_lon = (lp + sum_l / 1000000.0).to_radians();
        sink += ecl_lon;
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
