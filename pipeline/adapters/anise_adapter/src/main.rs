//! ANISE adapter for the benchmark lab.
//!
//! Reads experiment input from stdin and writes one JSON object to stdout.
//! Unsupported experiments return `{ "skipped": true }` responses.

use anise::constants::frames::{EARTH_J2000, MOON_J2000, SUN_J2000};
use anise::constants::orientations::{ECLIPJ2000, J2000, J2000_TO_ECLIPJ2000_ANGLE_RAD};
use anise::math::Vector3;
use anise::math::rotation::DCM;
use anise::prelude::{Aberration, Almanac, Epoch, TimeScale};

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

const KM_PER_AU: f64 = 149_597_870.7;

fn parse_numbers(line: &str) -> Vec<f64> {
    line.trim()
        .split_whitespace()
        .map(|s| s.parse::<f64>().unwrap())
        .collect()
}

fn normalize(v: Vector3) -> Vector3 {
    let n = v.norm();
    if n > 0.0 {
        v / n
    } else {
        v
    }
}

fn ang_sep(a: Vector3, b: Vector3) -> f64 {
    let dot = a.dot(&b).clamp(-1.0, 1.0);
    dot.acos()
}

fn epoch_from_jd_tt(jd_tt: f64) -> Epoch {
    Epoch::from_jde_in_time_scale(jd_tt, TimeScale::TT)
}

fn read_n(lines: &mut impl Iterator<Item = String>) -> usize {
    lines.next().unwrap().trim().parse::<usize>().unwrap()
}

fn skip_experiment(lines: &mut impl Iterator<Item = String>, exp: &str, reason: &str) {
    let n = read_n(lines);
    for _ in 0..n {
        let _ = lines.next();
    }
    println!(
        "{{\"experiment\":\"{}\",\"library\":\"anise\",\"skipped\":true,\"reason\":\"{}\"}}",
        exp, reason
    );
}

fn load_ephemeris_almanac() -> Result<Almanac, String> {
    let candidates = [
        "siderust/scripts/jpl/de440/dataset/de440.bsp",
        "anise/data/de440s.bsp",
        "anise/data/de440.bsp",
    ];

    for path in candidates {
        if Path::new(path).is_file() {
            if let Ok(almanac) = Almanac::new(path) {
                return Ok(almanac);
            }
        }
    }

    Err("ANISE SPK not found. Expected one of: siderust/scripts/jpl/de440/dataset/de440.bsp, anise/data/de440s.bsp, anise/data/de440.bsp".to_string())
}

fn run_solar_position(lines: &mut impl Iterator<Item = String>) {
    let n = read_n(lines);
    let almanac = match load_ephemeris_almanac() {
        Ok(a) => a,
        Err(reason) => {
            for _ in 0..n {
                let _ = lines.next();
            }
            println!(
                "{{\"experiment\":\"solar_position\",\"library\":\"anise\",\"skipped\":true,\"reason\":\"{}\"}}",
                reason
            );
            return;
        }
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"solar_position\",\"library\":\"anise\",\"model\":\"SPK_geometric_translation(SUN_J2000->EARTH_J2000)\",\"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let jd_tt: f64 = lines.next().unwrap().trim().parse().unwrap();
        let epoch = epoch_from_jd_tt(jd_tt);

        let state = match almanac.translate(SUN_J2000, EARTH_J2000, epoch, Aberration::NONE) {
            Ok(s) => s,
            Err(_) => {
                if i > 0 {
                    write!(out, ",\n").unwrap();
                }
                write!(
                    out,
                    "{{\"jd_tt\":{:.15},\"ra_rad\":null,\"dec_rad\":null,\"dist_au\":null}}",
                    jd_tt
                )
                .unwrap();
                continue;
            }
        };

        let x = state.radius_km[0];
        let y = state.radius_km[1];
        let z = state.radius_km[2];
        let dist_km = state.radius_km.norm();
        let ra = y.atan2(x).rem_euclid(2.0 * std::f64::consts::PI);
        let dec = (z / dist_km).asin();
        let dist_au = dist_km / KM_PER_AU;

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\"dist_au\":{:.17e}}}",
            jd_tt, ra, dec, dist_au
        )
        .unwrap();
    }

    writeln!(out, "\n]}}")
        .unwrap();
}

fn run_lunar_position(lines: &mut impl Iterator<Item = String>) {
    let n = read_n(lines);
    let almanac = match load_ephemeris_almanac() {
        Ok(a) => a,
        Err(reason) => {
            for _ in 0..n {
                let _ = lines.next();
            }
            println!(
                "{{\"experiment\":\"lunar_position\",\"library\":\"anise\",\"skipped\":true,\"reason\":\"{}\"}}",
                reason
            );
            return;
        }
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"lunar_position\",\"library\":\"anise\",\"model\":\"SPK_geometric_translation(MOON_J2000->EARTH_J2000)\",\"count\":{},\"cases\":[\n",
        n
    )
    .unwrap();

    for i in 0..n {
        let jd_tt: f64 = lines.next().unwrap().trim().parse().unwrap();
        let epoch = epoch_from_jd_tt(jd_tt);

        let state = match almanac.translate(MOON_J2000, EARTH_J2000, epoch, Aberration::NONE) {
            Ok(s) => s,
            Err(_) => {
                if i > 0 {
                    write!(out, ",\n").unwrap();
                }
                write!(
                    out,
                    "{{\"jd_tt\":{:.15},\"ra_rad\":null,\"dec_rad\":null,\"dist_km\":null}}",
                    jd_tt
                )
                .unwrap();
                continue;
            }
        };

        let x = state.radius_km[0];
        let y = state.radius_km[1];
        let z = state.radius_km[2];
        let dist_km = state.radius_km.norm();
        let ra = y.atan2(x).rem_euclid(2.0 * std::f64::consts::PI);
        let dec = (z / dist_km).asin();

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\"dist_km\":{:.17e}}}",
            jd_tt, ra, dec, dist_km
        )
        .unwrap();
    }

    writeln!(out, "\n]}}")
        .unwrap();
}

fn run_direction_rotation(
    lines: &mut impl Iterator<Item = String>,
    exp_name: &str,
    model: &str,
    j2000_to_ecliptic: bool,
) {
    let n = read_n(lines);
    let base = DCM::r1(J2000_TO_ECLIPJ2000_ANGLE_RAD, J2000, ECLIPJ2000);

    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"{}\",\"library\":\"anise\",\"model\":\"{}\",\"count\":{},\"cases\":[\n",
        exp_name, model, n
    )
    .unwrap();

    for i in 0..n {
        let p = parse_numbers(&lines.next().unwrap());
        let jd_tt = p[0];

        let dcm = if j2000_to_ecliptic {
            base
        } else {
            base.transpose()
        };

        let vin = normalize(Vector3::new(p[1], p[2], p[3]));
        let vout = dcm * vin;
        let vback = dcm.transpose() * vout;
        let closure = ang_sep(vin, vback);

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\"input\":[{:.17e},{:.17e},{:.17e}],\"output\":[{:.17e},{:.17e},{:.17e}],\"closure_rad\":{:.17e},\"matrix\":[[{:.17e},{:.17e},{:.17e}],[{:.17e},{:.17e},{:.17e}],[{:.17e},{:.17e},{:.17e}]]}}",
            jd_tt,
            vin[0],
            vin[1],
            vin[2],
            vout[0],
            vout[1],
            vout[2],
            closure,
            dcm.rot_mat[(0, 0)],
            dcm.rot_mat[(0, 1)],
            dcm.rot_mat[(0, 2)],
            dcm.rot_mat[(1, 0)],
            dcm.rot_mat[(1, 1)],
            dcm.rot_mat[(1, 2)],
            dcm.rot_mat[(2, 0)],
            dcm.rot_mat[(2, 1)],
            dcm.rot_mat[(2, 2)],
        )
        .unwrap();
    }

    writeln!(out, "\n]}}")
        .unwrap();
}

fn run_direction_rotation_perf(
    lines: &mut impl Iterator<Item = String>,
    exp_name: &str,
    j2000_to_ecliptic: bool,
) {
    let n = read_n(lines);
    let base = DCM::r1(J2000_TO_ECLIPJ2000_ANGLE_RAD, J2000, ECLIPJ2000);

    let mut jds = Vec::with_capacity(n);
    let mut vecs = Vec::with_capacity(n);
    for _ in 0..n {
        let p = parse_numbers(&lines.next().unwrap());
        jds.push(p[0]);
        vecs.push(normalize(Vector3::new(p[1], p[2], p[3])));
    }

    for i in 0..n.min(100) {
        let _ = jds[i];
        let dcm = if j2000_to_ecliptic {
            base
        } else {
            base.transpose()
        };
        let out = dcm * vecs[i];
        std::hint::black_box(out);
    }

    let start = Instant::now();
    let mut sink = 0.0_f64;
    for i in 0..n {
        let _ = jds[i];
        let dcm = if j2000_to_ecliptic {
            base
        } else {
            base.transpose()
        };
        let out = dcm * vecs[i];
        sink = out[0];
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;

    println!(
        "{{\"experiment\":\"{}\",\"library\":\"anise\",\"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        exp_name,
        n,
        total_ns,
        total_ns / n as f64,
        n as f64 / (total_ns * 1e-9),
        sink
    );
}

fn run_solar_position_perf(lines: &mut impl Iterator<Item = String>) {
    let n = read_n(lines);
    let almanac = match load_ephemeris_almanac() {
        Ok(a) => a,
        Err(reason) => {
            for _ in 0..n {
                let _ = lines.next();
            }
            println!(
                "{{\"experiment\":\"solar_position_perf\",\"library\":\"anise\",\"skipped\":true,\"reason\":\"{}\"}}",
                reason
            );
            return;
        }
    };

    let mut jds = Vec::with_capacity(n);
    for _ in 0..n {
        jds.push(lines.next().unwrap().trim().parse::<f64>().unwrap());
    }

    for jd in jds.iter().take(n.min(100)) {
        let epoch = epoch_from_jd_tt(*jd);
        if let Ok(state) = almanac.translate(SUN_J2000, EARTH_J2000, epoch, Aberration::NONE) {
            let x = state.radius_km[0];
            let y = state.radius_km[1];
            let z = state.radius_km[2];
            let dist_km = state.radius_km.norm();
            let ra = y.atan2(x).rem_euclid(2.0 * std::f64::consts::PI);
            let dec = (z / dist_km).asin();
            std::hint::black_box((ra, dec, dist_km));
        }
    }

    let start = Instant::now();
    let mut sink = 0.0_f64;
    for jd in &jds {
        let epoch = epoch_from_jd_tt(*jd);
        if let Ok(state) = almanac.translate(SUN_J2000, EARTH_J2000, epoch, Aberration::NONE) {
            let x = state.radius_km[0];
            let y = state.radius_km[1];
            let z = state.radius_km[2];
            let dist_km = state.radius_km.norm();
            let ra = y.atan2(x).rem_euclid(2.0 * std::f64::consts::PI);
            let dec = (z / dist_km).asin();
            sink += ra + dec + dist_km / KM_PER_AU;
        }
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;

    println!(
        "{{\"experiment\":\"solar_position_perf\",\"library\":\"anise\",\"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        total_ns / n as f64,
        n as f64 / (total_ns * 1e-9),
        sink
    );
}

fn run_lunar_position_perf(lines: &mut impl Iterator<Item = String>) {
    let n = read_n(lines);
    let almanac = match load_ephemeris_almanac() {
        Ok(a) => a,
        Err(reason) => {
            for _ in 0..n {
                let _ = lines.next();
            }
            println!(
                "{{\"experiment\":\"lunar_position_perf\",\"library\":\"anise\",\"skipped\":true,\"reason\":\"{}\"}}",
                reason
            );
            return;
        }
    };

    let mut jds = Vec::with_capacity(n);
    for _ in 0..n {
        jds.push(lines.next().unwrap().trim().parse::<f64>().unwrap());
    }

    for jd in jds.iter().take(n.min(100)) {
        let epoch = epoch_from_jd_tt(*jd);
        if let Ok(state) = almanac.translate(MOON_J2000, EARTH_J2000, epoch, Aberration::NONE) {
            let x = state.radius_km[0];
            let y = state.radius_km[1];
            let z = state.radius_km[2];
            let dist_km = state.radius_km.norm();
            let ra = y.atan2(x).rem_euclid(2.0 * std::f64::consts::PI);
            let dec = (z / dist_km).asin();
            std::hint::black_box((ra, dec, dist_km));
        }
    }

    let start = Instant::now();
    let mut sink = 0.0_f64;
    for jd in &jds {
        let epoch = epoch_from_jd_tt(*jd);
        if let Ok(state) = almanac.translate(MOON_J2000, EARTH_J2000, epoch, Aberration::NONE) {
            let x = state.radius_km[0];
            let y = state.radius_km[1];
            let z = state.radius_km[2];
            let dist_km = state.radius_km.norm();
            let ra = y.atan2(x).rem_euclid(2.0 * std::f64::consts::PI);
            let dec = (z / dist_km).asin();
            sink += ra + dec + dist_km;
        }
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;

    println!(
        "{{\"experiment\":\"lunar_position_perf\",\"library\":\"anise\",\"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        total_ns / n as f64,
        n as f64 / (total_ns * 1e-9),
        sink
    );
}

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines().map(|l| l.unwrap());

    let experiment = lines.next().expect("expected experiment name");
    let experiment = experiment.trim();

    match experiment {
        // Supported ephemerides
        "solar_position" => run_solar_position(&mut lines),
        "solar_position_perf" => run_solar_position_perf(&mut lines),
        "lunar_position" => run_lunar_position(&mut lines),
        "lunar_position_perf" => run_lunar_position_perf(&mut lines),

        // Supported J2000 ecliptic rotations
        "icrs_ecl_j2000" => run_direction_rotation(
            &mut lines,
            "icrs_ecl_j2000",
            "ANISE_J2000_to_ECLIPJ2000_rotation",
            true,
        ),
        "icrs_ecl_j2000_perf" => {
            run_direction_rotation_perf(&mut lines, "icrs_ecl_j2000_perf", true)
        }
        "inv_icrs_ecl_j2000" => run_direction_rotation(
            &mut lines,
            "inv_icrs_ecl_j2000",
            "ANISE_ECLIPJ2000_to_J2000_rotation",
            false,
        ),
        "inv_icrs_ecl_j2000_perf" => {
            run_direction_rotation_perf(&mut lines, "inv_icrs_ecl_j2000_perf", false)
        }
        "obliquity" => run_direction_rotation(
            &mut lines,
            "obliquity",
            "ANISE_ECLIPJ2000_to_J2000_obliquity_rotation",
            false,
        ),
        "obliquity_perf" => run_direction_rotation_perf(&mut lines, "obliquity_perf", false),
        "inv_obliquity" => run_direction_rotation(
            &mut lines,
            "inv_obliquity",
            "ANISE_J2000_to_ECLIPJ2000_obliquity_rotation",
            true,
        ),
        "inv_obliquity_perf" => {
            run_direction_rotation_perf(&mut lines, "inv_obliquity_perf", true)
        }

        // Unsupported (explicit skip)
        "frame_rotation_bpn" => skip_experiment(
            &mut lines,
            "frame_rotation_bpn",
            "ANISE adapter does not expose IAU2006/2000A BPN rotation in this lab integration",
        ),
        "frame_rotation_bpn_perf" => skip_experiment(
            &mut lines,
            "frame_rotation_bpn_perf",
            "ANISE adapter does not expose IAU2006/2000A BPN rotation in this lab integration",
        ),
        "gmst_era" => skip_experiment(
            &mut lines,
            "gmst_era",
            "ANISE adapter does not expose direct GMST/ERA benchmark API in this lab integration",
        ),
        "gmst_era_perf" => skip_experiment(
            &mut lines,
            "gmst_era_perf",
            "ANISE adapter does not expose direct GMST/ERA benchmark API in this lab integration",
        ),
        "equ_ecl" => skip_experiment(
            &mut lines,
            "equ_ecl",
            "ANISE adapter currently supports only J2000 ecliptic rotation, not ecliptic-of-date transform",
        ),
        "equ_ecl_perf" => skip_experiment(
            &mut lines,
            "equ_ecl_perf",
            "ANISE adapter currently supports only J2000 ecliptic rotation, not ecliptic-of-date transform",
        ),
        "equ_horizontal" => skip_experiment(
            &mut lines,
            "equ_horizontal",
            "ANISE adapter does not include equatorial-horizontal benchmark path in this lab integration",
        ),
        "equ_horizontal_perf" => skip_experiment(
            &mut lines,
            "equ_horizontal_perf",
            "ANISE adapter does not include equatorial-horizontal benchmark path in this lab integration",
        ),
        "kepler_solver" => skip_experiment(
            &mut lines,
            "kepler_solver",
            "ANISE adapter does not include standalone Kepler solver benchmark path in this lab integration",
        ),
        "kepler_solver_perf" => skip_experiment(
            &mut lines,
            "kepler_solver_perf",
            "ANISE adapter does not include standalone Kepler solver benchmark path in this lab integration",
        ),
        "frame_bias" => skip_experiment(
            &mut lines,
            "frame_bias",
            "ANISE adapter does not provide isolated frame-bias benchmark output in this lab integration",
        ),
        "frame_bias_perf" => skip_experiment(
            &mut lines,
            "frame_bias_perf",
            "ANISE adapter does not provide isolated frame-bias benchmark output in this lab integration",
        ),
        "precession" => skip_experiment(
            &mut lines,
            "precession",
            "ANISE adapter does not provide isolated precession benchmark output in this lab integration",
        ),
        "precession_perf" => skip_experiment(
            &mut lines,
            "precession_perf",
            "ANISE adapter does not provide isolated precession benchmark output in this lab integration",
        ),
        "nutation" => skip_experiment(
            &mut lines,
            "nutation",
            "ANISE adapter does not provide isolated nutation benchmark output in this lab integration",
        ),
        "nutation_perf" => skip_experiment(
            &mut lines,
            "nutation_perf",
            "ANISE adapter does not provide isolated nutation benchmark output in this lab integration",
        ),
        "icrs_ecl_tod" => skip_experiment(
            &mut lines,
            "icrs_ecl_tod",
            "ANISE adapter currently supports J2000 ecliptic rotation only (no ecliptic-of-date path)",
        ),
        "icrs_ecl_tod_perf" => skip_experiment(
            &mut lines,
            "icrs_ecl_tod_perf",
            "ANISE adapter currently supports J2000 ecliptic rotation only (no ecliptic-of-date path)",
        ),
        "horiz_to_equ" => skip_experiment(
            &mut lines,
            "horiz_to_equ",
            "ANISE adapter does not include horizontal-equatorial benchmark path in this lab integration",
        ),
        "horiz_to_equ_perf" => skip_experiment(
            &mut lines,
            "horiz_to_equ_perf",
            "ANISE adapter does not include horizontal-equatorial benchmark path in this lab integration",
        ),
        "inv_frame_bias" => skip_experiment(
            &mut lines,
            "inv_frame_bias",
            "ANISE adapter does not provide isolated inverse frame-bias benchmark output in this lab integration",
        ),
        "inv_frame_bias_perf" => skip_experiment(
            &mut lines,
            "inv_frame_bias_perf",
            "ANISE adapter does not provide isolated inverse frame-bias benchmark output in this lab integration",
        ),
        "inv_precession" => skip_experiment(
            &mut lines,
            "inv_precession",
            "ANISE adapter does not provide isolated inverse precession benchmark output in this lab integration",
        ),
        "inv_precession_perf" => skip_experiment(
            &mut lines,
            "inv_precession_perf",
            "ANISE adapter does not provide isolated inverse precession benchmark output in this lab integration",
        ),
        "inv_nutation" => skip_experiment(
            &mut lines,
            "inv_nutation",
            "ANISE adapter does not provide isolated inverse nutation benchmark output in this lab integration",
        ),
        "inv_nutation_perf" => skip_experiment(
            &mut lines,
            "inv_nutation_perf",
            "ANISE adapter does not provide isolated inverse nutation benchmark output in this lab integration",
        ),
        "inv_bpn" => skip_experiment(
            &mut lines,
            "inv_bpn",
            "ANISE adapter does not provide isolated inverse BPN benchmark output in this lab integration",
        ),
        "inv_bpn_perf" => skip_experiment(
            &mut lines,
            "inv_bpn_perf",
            "ANISE adapter does not provide isolated inverse BPN benchmark output in this lab integration",
        ),
        "bias_precession" => skip_experiment(
            &mut lines,
            "bias_precession",
            "ANISE adapter does not provide isolated bias+precession benchmark output in this lab integration",
        ),
        "bias_precession_perf" => skip_experiment(
            &mut lines,
            "bias_precession_perf",
            "ANISE adapter does not provide isolated bias+precession benchmark output in this lab integration",
        ),
        "inv_bias_precession" => skip_experiment(
            &mut lines,
            "inv_bias_precession",
            "ANISE adapter does not provide isolated inverse bias+precession benchmark output in this lab integration",
        ),
        "inv_bias_precession_perf" => skip_experiment(
            &mut lines,
            "inv_bias_precession_perf",
            "ANISE adapter does not provide isolated inverse bias+precession benchmark output in this lab integration",
        ),
        "precession_nutation" => skip_experiment(
            &mut lines,
            "precession_nutation",
            "ANISE adapter does not provide isolated precession+nutation benchmark output in this lab integration",
        ),
        "precession_nutation_perf" => skip_experiment(
            &mut lines,
            "precession_nutation_perf",
            "ANISE adapter does not provide isolated precession+nutation benchmark output in this lab integration",
        ),
        "inv_precession_nutation" => skip_experiment(
            &mut lines,
            "inv_precession_nutation",
            "ANISE adapter does not provide isolated inverse precession+nutation benchmark output in this lab integration",
        ),
        "inv_precession_nutation_perf" => skip_experiment(
            &mut lines,
            "inv_precession_nutation_perf",
            "ANISE adapter does not provide isolated inverse precession+nutation benchmark output in this lab integration",
        ),
        "inv_icrs_ecl_tod" => skip_experiment(
            &mut lines,
            "inv_icrs_ecl_tod",
            "ANISE adapter currently supports J2000 ecliptic rotation only (no ecliptic-of-date path)",
        ),
        "inv_icrs_ecl_tod_perf" => skip_experiment(
            &mut lines,
            "inv_icrs_ecl_tod_perf",
            "ANISE adapter currently supports J2000 ecliptic rotation only (no ecliptic-of-date path)",
        ),
        "inv_equ_ecl" => skip_experiment(
            &mut lines,
            "inv_equ_ecl",
            "ANISE adapter currently supports J2000 ecliptic rotation only (no ecliptic-of-date path)",
        ),
        "inv_equ_ecl_perf" => skip_experiment(
            &mut lines,
            "inv_equ_ecl_perf",
            "ANISE adapter currently supports J2000 ecliptic rotation only (no ecliptic-of-date path)",
        ),
        _ => {
            eprintln!("Unknown experiment: {}", experiment);
            std::process::exit(1);
        }
    }
}
