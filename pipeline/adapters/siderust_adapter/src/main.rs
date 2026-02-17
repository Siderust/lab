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
use siderust::coordinates::centers::{Geocentric, Heliocentric, ObserverSite};
use siderust::coordinates::frames::{EquatorialTrueOfDate, ICRS};
use siderust::coordinates::transform::ecliptic_of_date::FromEclipticTrueOfDate;
use siderust::coordinates::transform::horizontal::FromHorizontal;
use siderust::coordinates::transform::providers::frame_rotation;
use siderust::coordinates::transform::{AstroContext, DirectionAstroExt, Transform};
use siderust::time::JulianDate;

use qtty::{AstronomicalUnit, Degrees, Meters, Radians};

use std::io::{self, BufRead, Write};
use std::time::Instant;

mod direction_experiments;
use direction_experiments::*;

fn ang_sep(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    dot.clamp(-1.0, 1.0).acos()
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
        let jd = JulianDate::new(jd_tt);

        let dir_icrs = cartesian::Direction::<ICRS>::new(parts[1], parts[2], parts[3]);
        let vin = dir_icrs.as_vec3();

        // Forward: ICRS → TrueOfDate via siderust DirectionAstroExt (IAU default)
        let dir_tod: cartesian::Direction<EquatorialTrueOfDate> = dir_icrs.to_frame(&jd);
        let vout = dir_tod.as_vec3();

        // Closure test: inverse transformation via siderust
        let dir_back: cartesian::Direction<ICRS> = dir_tod.to_frame(&jd);
        let closure_rad = dir_icrs.angle_to(&dir_back);

        // Get matrix for output compatibility
        let mat =
            *frame_rotation::<ICRS, EquatorialTrueOfDate>(jd, &AstroContext::default()).as_matrix();

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
        let ra_rad = parts[1];
        let dec_rad = parts[2];

        let jd = JulianDate::new(jd_tt);

        // Create ICRS cartesian direction from spherical coordinates
        let spherical_icrs = affn::spherical::Direction::<ICRS>::new_raw(
            Degrees::new(dec_rad.to_degrees()),
            Degrees::new(ra_rad.to_degrees()),
        );
        let icrs_direction = spherical_icrs.to_cartesian();

        // Transform to ecliptic of date (via DirectionAstroExt convenience)
        let ecliptic_direction = DirectionAstroExt::to_ecliptic_of_date(&icrs_direction, &jd);
        let ecliptic_spherical = ecliptic_direction.to_spherical();
        let ecl_lon = Radians::from(ecliptic_spherical.azimuth);
        let ecl_lat = Radians::from(ecliptic_spherical.polar);

        // Transform back to ICRS
        let icrs_back = ecliptic_direction.to_icrs(&jd);
        let icrs_back_spherical = icrs_back.to_spherical();
        let ra_back = Radians::from(icrs_back_spherical.azimuth);
        let dec_back = Radians::from(icrs_back_spherical.polar);

        let v_in = [
            dec_rad.cos() * ra_rad.cos(),
            dec_rad.cos() * ra_rad.sin(),
            dec_rad.sin(),
        ];
        let v_back = [
            dec_back.value().cos() * ra_back.value().cos(),
            dec_back.value().cos() * ra_back.value().sin(),
            dec_back.value().sin(),
        ];
        let closure_rad = ang_sep(&v_in, &v_back);

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\
             \"ecl_lon_rad\":{:.17e},\"ecl_lat_rad\":{:.17e},\
             \"closure_rad\":{:.17e}}}",
            jd_tt,
            ra_rad,
            dec_rad,
            ecl_lon.value(),
            ecl_lat.value(),
            closure_rad,
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
        let ra_rad = parts[2];
        let dec_rad = parts[3];
        let obs_lon_rad = parts[4];
        let obs_lat_rad = parts[5];

        let jd_ut1_q = JulianDate::new(jd_ut1);
        let jd_tt_q = JulianDate::new(jd_tt_val);

        // Create equatorial true-of-date cartesian direction
        let spherical_equ = affn::spherical::Direction::<EquatorialTrueOfDate>::new_raw(
            Degrees::new(dec_rad.to_degrees()),
            Degrees::new(ra_rad.to_degrees()),
        );
        let equatorial_direction = spherical_equ.to_cartesian();

        // Create observer site
        let site = ObserverSite::new(
            Degrees::new(obs_lon_rad.to_degrees()),
            Degrees::new(obs_lat_rad.to_degrees()),
            Meters::new(0.0),
        );

        // Transform to horizontal (precise: explicit UT1+TT for benchmark fairness)
        let horizontal_direction = DirectionAstroExt::to_horizontal_precise(
            &equatorial_direction,
            &jd_tt_q,
            &jd_ut1_q,
            &site,
        );
        let horizontal_spherical = horizontal_direction.to_spherical();
        let az = Radians::from(horizontal_spherical.azimuth);
        let alt = Radians::from(horizontal_spherical.polar);

        // Transform back to equatorial
        let equatorial_back: affn::cartesian::Direction<EquatorialTrueOfDate> =
            horizontal_direction.to_equatorial(&jd_ut1_q, &jd_tt_q, &site);
        let equatorial_back_spherical = equatorial_back.to_spherical();
        let ra_back = Radians::from(equatorial_back_spherical.azimuth);
        let dec_back = Radians::from(equatorial_back_spherical.polar);

        let v_in = [
            dec_rad.cos() * ra_rad.cos(),
            dec_rad.cos() * ra_rad.sin(),
            dec_rad.sin(),
        ];
        let v_back = [
            dec_back.value().cos() * ra_back.value().cos(),
            dec_back.value().cos() * ra_back.value().sin(),
            dec_back.value().sin(),
        ];
        let closure_rad = ang_sep(&v_in, &v_back);

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
            jd_ut1,
            jd_tt_val,
            ra_rad,
            dec_rad,
            obs_lon_rad,
            obs_lat_rad,
            az.value(),
            alt.value(),
            closure_rad,
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
        let helio = siderust::coordinates::cartesian::position::EclipticMeanJ2000::<
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
        dirs.push(cartesian::Direction::<ICRS>::new(
            parts[1], parts[2], parts[3],
        ));
    }

    // Warm-up
    for i in 0..n.min(100) {
        let jd = JulianDate::new(jds[i]);
        let dir_tod: cartesian::Direction<EquatorialTrueOfDate> = dirs[i].to_frame(&jd);
        std::hint::black_box(&dir_tod);
    }

    // Timed run
    let start = Instant::now();
    let mut sink = 0.0_f64;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let dir_tod: cartesian::Direction<EquatorialTrueOfDate> = dirs[i].to_frame(&jd);
        sink = dir_tod.x();
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
        sink,
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
        let jd = JulianDate::new(jds[i]);
        let spherical_icrs = affn::spherical::Direction::<ICRS>::new_raw(
            Degrees::new(decs[i].to_degrees()),
            Degrees::new(ras[i].to_degrees()),
        );
        let icrs_direction = spherical_icrs.to_cartesian();
        let res = icrs_direction.to_ecliptic_of_date(&jd);
        std::hint::black_box(&res);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let spherical_icrs = affn::spherical::Direction::<ICRS>::new_raw(
            Degrees::new(decs[i].to_degrees()),
            Degrees::new(ras[i].to_degrees()),
        );
        let icrs_direction = spherical_icrs.to_cartesian();
        let ecliptic_direction = icrs_direction.to_ecliptic_of_date(&jd);
        let ecliptic_spherical = ecliptic_direction.to_spherical();
        let lon = Radians::from(ecliptic_spherical.azimuth);
        sink += lon.value();
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
        let (jd_ut1, jd_tt, ra, dec, lon, lat) = params[i];
        let jd_ut1_q = JulianDate::new(jd_ut1);
        let jd_tt_q = JulianDate::new(jd_tt);

        let spherical_equ = affn::spherical::Direction::<EquatorialTrueOfDate>::new_raw(
            Degrees::new(dec.to_degrees()),
            Degrees::new(ra.to_degrees()),
        );
        let equatorial_direction = spherical_equ.to_cartesian();
        let site = ObserverSite::new(
            Degrees::new(lon.to_degrees()),
            Degrees::new(lat.to_degrees()),
            Meters::new(0.0),
        );
        let dir_hor = equatorial_direction.to_horizontal_precise(&jd_tt_q, &jd_ut1_q, &site);
        std::hint::black_box(&dir_hor);
    }

    // Timed run
    let start = Instant::now();
    let mut sink: (f64, f64) = (0.0, 0.0);
    for i in 0..n {
        let (jd_ut1, jd_tt, ra, dec, lon, lat) = params[i];
        let jd_ut1_q = JulianDate::new(jd_ut1);
        let jd_tt_q = JulianDate::new(jd_tt);

        let spherical_equ = affn::spherical::Direction::<EquatorialTrueOfDate>::new_raw(
            Degrees::new(dec.to_degrees()),
            Degrees::new(ra.to_degrees()),
        );
        let equatorial_direction = spherical_equ.to_cartesian();
        let site = ObserverSite::new(
            Degrees::new(lon.to_degrees()),
            Degrees::new(lat.to_degrees()),
            Meters::new(0.0),
        );
        let horizontal_direction =
            equatorial_direction.to_horizontal_precise(&jd_tt_q, &jd_ut1_q, &site);
        let horizontal_spherical = horizontal_direction.to_spherical();
        let az = Radians::from(horizontal_spherical.azimuth);
        let alt = Radians::from(horizontal_spherical.polar);
        sink = (az.value(), alt.value());
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
        let helio = siderust::coordinates::cartesian::position::EclipticMeanJ2000::<
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

    // Timed run — perf contract: compute RA + Dec + distance
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let helio = siderust::coordinates::cartesian::position::EclipticMeanJ2000::<
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
        sink += ra_rad + dec_rad + dist_au;
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
        let ra = moon.ra.to::<qtty::Radian>().value();
        let dec = moon.dec.to::<qtty::Radian>().value();
        let dist = moon.dist.value();
        std::hint::black_box(ra + dec + dist);
    }

    // Timed run — perf contract: compute RA + Dec + distance
    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let moon = meeus_ch47::moon_position_meeus_ch47(JulianDate::new(jds[i]));
        let ra = moon.ra.to::<qtty::Radian>().value();
        let dec = moon.dec.to::<qtty::Radian>().value();
        let dist = moon.dist.value();
        sink += ra + dec + dist;
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
        "frame_bias" => run_frame_bias(&mut lines),
        "frame_bias_perf" => run_frame_bias_perf(&mut lines),
        "precession" => run_precession(&mut lines),
        "precession_perf" => run_precession_perf(&mut lines),
        "nutation" => run_nutation(&mut lines),
        "nutation_perf" => run_nutation_perf(&mut lines),
        "icrs_ecl_j2000" => run_icrs_ecl_j2000(&mut lines),
        "icrs_ecl_j2000_perf" => run_icrs_ecl_j2000_perf(&mut lines),
        "icrs_ecl_tod" => run_icrs_ecl_tod(&mut lines),
        "icrs_ecl_tod_perf" => run_icrs_ecl_tod_perf(&mut lines),
        "horiz_to_equ" => run_horiz_to_equ(&mut lines),
        "horiz_to_equ_perf" => run_horiz_to_equ_perf(&mut lines),
        // 13 new matrix experiments
        "inv_frame_bias" => run_inv_frame_bias(&mut lines),
        "inv_frame_bias_perf" => run_inv_frame_bias_perf(&mut lines),
        "inv_precession" => run_inv_precession(&mut lines),
        "inv_precession_perf" => run_inv_precession_perf(&mut lines),
        "inv_nutation" => run_inv_nutation(&mut lines),
        "inv_nutation_perf" => run_inv_nutation_perf(&mut lines),
        "inv_bpn" => run_inv_bpn(&mut lines),
        "inv_bpn_perf" => run_inv_bpn_perf(&mut lines),
        "inv_icrs_ecl_j2000" => run_inv_icrs_ecl_j2000(&mut lines),
        "inv_icrs_ecl_j2000_perf" => run_inv_icrs_ecl_j2000_perf(&mut lines),
        "obliquity" => run_obliquity(&mut lines),
        "obliquity_perf" => run_obliquity_perf(&mut lines),
        "inv_obliquity" => run_inv_obliquity(&mut lines),
        "inv_obliquity_perf" => run_inv_obliquity_perf(&mut lines),
        "bias_precession" => run_bias_precession(&mut lines),
        "bias_precession_perf" => run_bias_precession_perf(&mut lines),
        "inv_bias_precession" => run_inv_bias_precession(&mut lines),
        "inv_bias_precession_perf" => run_inv_bias_precession_perf(&mut lines),
        "precession_nutation" => run_precession_nutation(&mut lines),
        "precession_nutation_perf" => run_precession_nutation_perf(&mut lines),
        "inv_precession_nutation" => run_inv_precession_nutation(&mut lines),
        "inv_precession_nutation_perf" => run_inv_precession_nutation_perf(&mut lines),
        "inv_icrs_ecl_tod" => run_inv_icrs_ecl_tod(&mut lines),
        "inv_icrs_ecl_tod_perf" => run_inv_icrs_ecl_tod_perf(&mut lines),
        "inv_equ_ecl" => run_inv_equ_ecl(&mut lines),
        "inv_equ_ecl_perf" => run_inv_equ_ecl_perf(&mut lines),
        _ => {
            eprintln!("Unknown experiment: {}", experiment);
            std::process::exit(1);
        }
    }
}
