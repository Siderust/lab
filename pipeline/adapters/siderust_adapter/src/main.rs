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

use siderust::calculus::kepler_equations::solve_keplers_equation;
use siderust::calculus::lunar::meeus_ch47;
use siderust::coordinates::cartesian;
use siderust::coordinates::centers::{Geocentric, Heliocentric, ObserverSite, Topocentric};
use siderust::coordinates::frames::{Ecliptic, EquatorialMeanJ2000, EquatorialMeanOfDate, EquatorialTrueOfDate, Horizontal, ICRS};
use siderust::coordinates::spherical;
use siderust::coordinates::transform::providers::frame_rotation;
use siderust::coordinates::transform::{AstroContext, Transform, TransformFrame};
use siderust::time::JulianDate;

use qtty::{AstronomicalUnit, Meter, Radians, M, RAD};
use nalgebra::Vector3;

use std::io::{self, BufRead, Write};
use std::time::Instant;

fn normalize_vec3(v: Vector3<f64>) -> Vector3<f64> {
    let norm = v.norm();
    if norm > 0.0 {
        v / norm
    } else {
        v
    }
}

fn ang_sep(a: &Vector3<f64>, b: &Vector3<f64>) -> Radians {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos() * RAD
}

/// Compute the full BPN rotation: ICRS → EquatorialTrueOfDate
/// This composes: frame_bias * precession * nutation
fn bpn_direction(jd: JulianDate, dir_icrs: &cartesian::Direction<ICRS>) -> cartesian::Direction<EquatorialTrueOfDate> {
    // Transform ICRS -> EquatorialMeanJ2000 -> EquatorialMeanOfDate -> EquatorialTrueOfDate
    // Currently we only have simple frame transformations, so we need to use the rotation matrix
    let ctx = AstroContext::default();
    let rot = frame_rotation::<ICRS, EquatorialTrueOfDate>(jd, &ctx);
    let v_in = dir_icrs.as_vec3();
    let mat = rot.as_matrix();
    let v_out = Vector3::new(
        mat[0][0] * v_in[0] + mat[0][1] * v_in[1] + mat[0][2] * v_in[2],
        mat[1][0] * v_in[0] + mat[1][1] * v_in[1] + mat[1][2] * v_in[2],
        mat[2][0] * v_in[0] + mat[2][1] * v_in[1] + mat[2][2] * v_in[2],
    );
    cartesian::Direction::<EquatorialTrueOfDate>::from_vec3(v_out)
}

fn bpn_matrix(jd: JulianDate) -> [[f64; 3]; 3] {
    let ctx = AstroContext::default();
    let rot = frame_rotation::<ICRS, EquatorialTrueOfDate>(jd, &ctx);
    *rot.as_matrix()
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
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let jd_tt = parts[0];
        let vin = normalize_vec3(Vector3::new(parts[1], parts[2], parts[3]));

        let jd = JulianDate::new(jd_tt);
        let dir_icrs = cartesian::Direction::<ICRS>::from_vec3(vin);
        let dir_tod = bpn_direction(jd, &dir_icrs);
        let vout = dir_tod.as_vec3();

        // Closure test: inverse transformation
        let ctx = AstroContext::default();
        let rot = frame_rotation::<EquatorialTrueOfDate, ICRS>(jd, &ctx);
        let mat_inv = rot.as_matrix();
        let vinv = Vector3::new(
            mat_inv[0][0] * vout[0] + mat_inv[0][1] * vout[1] + mat_inv[0][2] * vout[2],
            mat_inv[1][0] * vout[0] + mat_inv[1][1] * vout[1] + mat_inv[1][2] * vout[2],
            mat_inv[2][0] * vout[0] + mat_inv[2][1] * vout[1] + mat_inv[2][2] * vout[2],
        );
        let closure_rad = ang_sep(&vin, &vinv).value();

        // Get matrix for output compatibility
        let mat = bpn_matrix(jd);

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
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
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

fn equatorial_to_ecliptic_of_date(dir_eq: spherical::Direction<EquatorialMeanJ2000>) -> spherical::Direction<Ecliptic> {
    // Use siderust's proper cartesian Direction and TransformFrame
    let ra_rad = dir_eq.azimuth.value().to_radians();
    let dec_rad = dir_eq.polar.value().to_radians();
    let (sd, cd) = dec_rad.sin_cos();
    let (sa, ca) = ra_rad.sin_cos();
    let cart_eq = cartesian::Direction::<EquatorialMeanJ2000>::from_vec3(
        Vector3::new(cd * ca, cd * sa, sd)
    );
    let cart_ecl: cartesian::Direction<Ecliptic> = cart_eq.to_frame();
    spherical::Direction::from_cartesian(&cart_ecl)
}

fn ecliptic_of_date_to_equatorial(dir_ecl: spherical::Direction<Ecliptic>) -> spherical::Direction<EquatorialMeanJ2000> {
    // Use siderust's proper cartesian Direction and TransformFrame
    let lon_rad = dir_ecl.azimuth.value().to_radians();
    let lat_rad = dir_ecl.polar.value().to_radians();
    let (slat, clat) = lat_rad.sin_cos();
    let (slon, clon) = lon_rad.sin_cos();
    let cart_ecl = cartesian::Direction::<Ecliptic>::from_vec3(
        Vector3::new(clat * clon, clat * slon, slat)
    );
    let cart_eq: cartesian::Direction<EquatorialMeanJ2000> = cart_ecl.to_frame();
    spherical::Direction::from_cartesian(&cart_eq)
}

fn equatorial_to_horizontal(
    jd_tt: JulianDate,
    dir_eq: spherical::Direction<EquatorialMeanOfDate>,
    obs_site: ObserverSite,
) -> spherical::Direction<Horizontal> {
    // Use siderust's proper Transform trait with Topocentric center and Horizontal frame
    let pos_eq = dir_eq.position_with_params::<Topocentric, Meter>(obs_site, 1.0 * M);
    let pos_hor: spherical::Position<Topocentric, Horizontal, Meter> = pos_eq.transform(jd_tt);
    spherical::Direction::new_raw(pos_hor.polar, pos_hor.azimuth)
}

fn horizontal_to_equatorial(
    jd_tt: JulianDate,
    dir_hor: spherical::Direction<Horizontal>,
    obs_site: ObserverSite,
) -> spherical::Direction<EquatorialMeanOfDate> {
    // Use siderust's proper Transform trait with Topocentric center and Horizontal frame
    let pos_hor = dir_hor.position_with_params::<Topocentric, Meter>(obs_site, 1.0 * M);
    let pos_eq: spherical::Position<Topocentric, EquatorialMeanOfDate, Meter> = pos_hor.transform(jd_tt);
    spherical::Direction::new_raw(pos_eq.polar, pos_eq.azimuth)
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
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let jd_tt = parts[0];
        let ra_deg = parts[1].to_degrees();
        let dec_deg = parts[2].to_degrees();

        let dir_eq = spherical::Direction::<EquatorialMeanJ2000>::new_raw(
            dec_deg * qtty::DEG,
            ra_deg * qtty::DEG,
        );

        let ra_rad = parts[1];
        let dec_rad = parts[2];
        let v_in = Vector3::new(
            dec_rad.cos() * ra_rad.cos(),
            dec_rad.cos() * ra_rad.sin(),
            dec_rad.sin(),
        );
        
        let dir_ecl = equatorial_to_ecliptic_of_date(dir_eq);
        let dir_eq_back = ecliptic_of_date_to_equatorial(dir_ecl);

        let ra_back_rad = dir_eq_back.azimuth.value().to_radians();
        let dec_back_rad = dir_eq_back.polar.value().to_radians();
        let v_bk = Vector3::new(
            dec_back_rad.cos() * ra_back_rad.cos(),
            dec_back_rad.cos() * ra_back_rad.sin(),
            dec_back_rad.sin(),
        );
        let closure = ang_sep(&v_in, &v_bk);

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\
             \"ecl_lon_rad\":{:.17e},\"ecl_lat_rad\":{:.17e},\
             \"closure_rad\":{:.17e}}}",
            jd_tt, ra_rad, dec_rad, 
            dir_ecl.azimuth.value().to_radians(), dir_ecl.polar.value().to_radians(), 
            closure.value(),
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
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let jd_ut1 = parts[0];
        let jd_tt_val = parts[1];
        let ra_deg = parts[2].to_degrees();
        let dec_deg = parts[3].to_degrees();
        let obs_lon_deg = parts[4].to_degrees();
        let obs_lat_deg = parts[5].to_degrees();

        let dir_eq = spherical::Direction::<EquatorialMeanOfDate>::new_raw(
            dec_deg * qtty::DEG,
            ra_deg * qtty::DEG,
        );
        let obs_site = ObserverSite::new(
            obs_lon_deg * qtty::DEG,
            obs_lat_deg * qtty::DEG,
            0.0 * M
        );

        let jd_tt_q = JulianDate::new(jd_tt_val);
        let dir_hor = equatorial_to_horizontal(jd_tt_q, dir_eq, obs_site.clone());
        let dir_eq_back = horizontal_to_equatorial(jd_tt_q, dir_hor, obs_site);

        let ra_rad = parts[2];
        let dec_rad = parts[3];
        let v_in = Vector3::new(
            dec_rad.cos() * ra_rad.cos(),
            dec_rad.cos() * ra_rad.sin(),
            dec_rad.sin(),
        );
        let ra_back_rad = dir_eq_back.azimuth.value().to_radians();
        let dec_back_rad = dir_eq_back.polar.value().to_radians();
        let v_bk = Vector3::new(
            dec_back_rad.cos() * ra_back_rad.cos(),
            dec_back_rad.cos() * ra_back_rad.sin(),
            dec_back_rad.sin(),
        );
        let closure = ang_sep(&v_in, &v_bk);

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_ut1\":{:.15},\"jd_tt\":{:.15},\
             \"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\
             \"obs_lon_rad\":{:.17e},\"obs_lat_rad\":{:.17e},\
             \"az_rad\":{:.17e},\"alt_rad\":{:.17e},\
             \"closure_rad\":{:.17e}}}",
            jd_ut1, jd_tt_val, ra_rad, dec_rad, parts[4], parts[5],
            dir_hor.azimuth.value().to_radians(), dir_hor.polar.value().to_radians(), 
            closure.value(),
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
        let helio = siderust::coordinates::cartesian::position::Ecliptic::<
            AstronomicalUnit,
            Heliocentric,
        >::CENTER;
        let geo_icrs: siderust::coordinates::cartesian::Position<
            Geocentric,
            ICRS,
            AstronomicalUnit,
        > = helio.transform(jd);
        let sph = siderust::coordinates::spherical::Position::from_cartesian(&geo_icrs);
        let ra_rad = sph.azimuth.value().to_radians();
        let dec_rad = sph.polar.value().to_radians();
        let dist_au = sph.distance.value();

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
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
        let moon = meeus_ch47::moon_position_meeus_ch47(JulianDate::new(jd_tt));
        let ra_rad = moon.ra.to::<qtty::Radian>().value();
        let dec_rad = moon.dec.to::<qtty::Radian>().value();
        let dist_km = moon.dist.value();

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
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
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let m_rad = parts[0];
        let e = parts[1];

        let m_qty = m_rad * qtty::RAD;
        let e_qty = solve_keplers_equation(m_qty, e);
        let e_rad = e_qty.value();

        // True anomaly
        let nu = 2.0
            * ((1.0 + e).sqrt() * (e_rad / 2.0).sin())
                .atan2((1.0 - e).sqrt() * (e_rad / 2.0).cos());

        // Self-consistency residual
        let residual = (e_rad - e * e_rad.sin() - m_rad).abs();

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"M_rad\":{:.17e},\"e\":{:.17e},\
             \"E_rad\":{:.17e},\"nu_rad\":{:.17e},\
             \"residual_rad\":{:.17e},\"iters\":-1,\"converged\":{}}}",
            m_rad,
            e,
            e_rad,
            nu,
            residual,
            if residual < 1e-12 { "true" } else { "false" },
        )
        .unwrap();
    }
    writeln!(out, "\n]}}").unwrap();
}

fn run_frame_rotation_bpn_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut jds: Vec<f64> = Vec::with_capacity(n);
    let mut dirs: Vec<cartesian::Direction<ICRS>> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        jds.push(parts[0]);
        let v = normalize_vec3(Vector3::new(parts[1], parts[2], parts[3]));
        dirs.push(cartesian::Direction::<ICRS>::from_vec3(v));
    }

    // Warm-up
    for i in 0..n.min(100) {
        let jd = JulianDate::new(jds[i]);
        let dir_tod = bpn_direction(jd, &dirs[i]);
        std::hint::black_box(&dir_tod);
    }

    // Timed run
    let start = Instant::now();
    let mut sink = Vector3::zeros();
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let dir_tod = bpn_direction(jd, &dirs[i]);
        sink = dir_tod.as_vec3();
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
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
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
        n,
        total_ns,
        per_op_ns,
        n as f64 / (total_ns * 1e-9),
        sink,
    );
}

fn run_equ_ecl_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut jds: Vec<f64> = Vec::with_capacity(n);
    let mut ras: Vec<f64> = Vec::with_capacity(n);
    let mut decs: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        jds.push(parts[0]);
        ras.push(parts[1]);
        decs.push(parts[2]);
    }

    // Warm-up
    for i in 0..n.min(100) {
        let dir_eq = spherical::Direction::<EquatorialMeanJ2000>::new_raw(
            decs[i].to_degrees() * qtty::DEG,
            ras[i].to_degrees() * qtty::DEG,
        );
        let res = equatorial_to_ecliptic_of_date(dir_eq);
        std::hint::black_box(&res);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let dir_eq = spherical::Direction::<EquatorialMeanJ2000>::new_raw(
            decs[i].to_degrees() * qtty::DEG,
            ras[i].to_degrees() * qtty::DEG,
        );
        let dir_ecl = equatorial_to_ecliptic_of_date(dir_eq);
        sink += dir_ecl.azimuth.value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"equ_ecl_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        per_op_ns,
        n as f64 / (total_ns * 1e-9),
        sink,
    );
}

fn run_equ_horizontal_perf(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut params: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        params.push((parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]));
    }

    // Warm-up
    for i in 0..n.min(100) {
        let (_jd_ut1, jd_tt, ra, dec, lon, lat) = params[i];
        let jd_tt_q = JulianDate::new(jd_tt);
        let dir_eq = spherical::Direction::<EquatorialMeanOfDate>::new_raw(
            dec.to_degrees() * qtty::DEG,
            ra.to_degrees() * qtty::DEG,
        );
        let obs_site = ObserverSite::new(
            lon.to_degrees() * qtty::DEG,
            lat.to_degrees() * qtty::DEG,
            0.0 * M
        );
        let dir_hor = equatorial_to_horizontal(jd_tt_q, dir_eq, obs_site);
        std::hint::black_box(&dir_hor);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: (f64, f64) = (0.0, 0.0);
    for i in 0..n {
        let (_jd_ut1, jd_tt, ra, dec, lon, lat) = params[i];
        let jd_tt_q = JulianDate::new(jd_tt);
        let dir_eq = spherical::Direction::<EquatorialMeanOfDate>::new_raw(
            dec.to_degrees() * qtty::DEG,
            ra.to_degrees() * qtty::DEG,
        );
        let obs_site = ObserverSite::new(
            lon.to_degrees() * qtty::DEG,
            lat.to_degrees() * qtty::DEG,
            0.0 * M
        );
        let dir_hor = equatorial_to_horizontal(jd_tt_q, dir_eq, obs_site);
        sink = (dir_hor.azimuth.value(), dir_hor.polar.value());
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"equ_horizontal_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        per_op_ns,
        n as f64 / (total_ns * 1e-9),
        sink.0,
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
        let helio = siderust::coordinates::cartesian::position::Ecliptic::<
            AstronomicalUnit,
            Heliocentric,
        >::CENTER;
        let geo_icrs: siderust::coordinates::cartesian::Position<
            Geocentric,
            ICRS,
            AstronomicalUnit,
        > = helio.transform(jd);
        std::hint::black_box(&geo_icrs);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let helio = siderust::coordinates::cartesian::position::Ecliptic::<
            AstronomicalUnit,
            Heliocentric,
        >::CENTER;
        let geo_icrs: siderust::coordinates::cartesian::Position<
            Geocentric,
            ICRS,
            AstronomicalUnit,
        > = helio.transform(jd);
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
        n,
        total_ns,
        per_op_ns,
        n as f64 / (total_ns * 1e-9),
        sink,
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
        let moon = meeus_ch47::moon_position_meeus_ch47(JulianDate::new(jds[i]));
        std::hint::black_box(moon.ecl_lon.value());
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let moon = meeus_ch47::moon_position_meeus_ch47(JulianDate::new(jds[i]));
        sink += moon.ecl_lon.value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;
    let per_op_ns = total_ns / n as f64;

    println!(
        "{{\"experiment\":\"lunar_position_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        per_op_ns,
        n as f64 / (total_ns * 1e-9),
        sink,
    );
}

fn run_kepler_solver_perf(lines: &mut impl Iterator<Item = String>) {
    use qtty::RAD;

    let n: usize = lines.next().unwrap().trim().parse().unwrap();

    let mut m_vals: Vec<f64> = Vec::with_capacity(n);
    let mut e_vals: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        let line = lines.next().unwrap();
        let parts: Vec<f64> = line
            .trim()
            .split_whitespace()
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
        n,
        total_ns,
        per_op_ns,
        n as f64 / (total_ns * 1e-9),
        sink,
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
