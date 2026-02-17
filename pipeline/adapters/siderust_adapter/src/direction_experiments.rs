use std::io::{self, Write};
use std::time::Instant;

use qtty::{Degrees, Meters, Radians};

use siderust::coordinates::cartesian;
use siderust::coordinates::centers::ObserverSite;
use siderust::coordinates::frames::{
    EclipticMeanJ2000, EclipticTrueOfDate, EquatorialMeanJ2000, EquatorialMeanOfDate,
    EquatorialTrueOfDate, Horizontal, ICRS,
};
use siderust::coordinates::transform::ecliptic_of_date::FromEclipticTrueOfDate;
use siderust::coordinates::transform::horizontal::FromHorizontal;
use siderust::coordinates::transform::DirectionAstroExt;
use siderust::time::JulianDate;

use crate::ang_sep;

macro_rules! dir_experiment {
    ($fn_acc:ident, $fn_perf:ident, $exp:expr, $model:expr, $Src:ty, $Dst:ty) => {
        pub fn $fn_acc(lines: &mut impl Iterator<Item = String>) {
            let n: usize = lines.next().unwrap().trim().parse().unwrap();
            let stdout = io::stdout();
            let mut out = stdout.lock();
            write!(
                out,
                concat!(
                    "{{\"experiment\":\"",
                    $exp,
                    "\",\"library\":\"siderust\",",
                    "\"model\":\"",
                    $model,
                    "\",\"count\":{},\"cases\":[\n"
                ),
                n
            )
            .unwrap();
            for i in 0..n {
                let line = lines.next().unwrap();
                let p: Vec<f64> = line
                    .trim()
                    .split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect();
                let jd = JulianDate::new(p[0]);
                let src = cartesian::Direction::<$Src>::new(p[1], p[2], p[3]);
                let vin = src.as_vec3();
                let dst: cartesian::Direction<$Dst> = src.to_frame(&jd);
                let vout = dst.as_vec3();
                let bk: cartesian::Direction<$Src> = dst.to_frame(&jd);
                let cl = src.angle_to(&bk);
                if i > 0 {
                    write!(out, ",\n").unwrap();
                }
                write!(
                    out,
                    "{{\"jd_tt\":{:.15},\"input\":[{:.17e},{:.17e},{:.17e}],\
                     \"output\":[{:.17e},{:.17e},{:.17e}],\"closure_rad\":{:.17e}}}",
                    p[0], vin[0], vin[1], vin[2], vout[0], vout[1], vout[2], cl,
                )
                .unwrap();
            }
            writeln!(out, "\n]}}").unwrap();
        }

        pub fn $fn_perf(lines: &mut impl Iterator<Item = String>) {
            let n: usize = lines.next().unwrap().trim().parse().unwrap();
            let mut jds = Vec::with_capacity(n);
            let mut dirs: Vec<cartesian::Direction<$Src>> = Vec::with_capacity(n);
            for _ in 0..n {
                let line = lines.next().unwrap();
                let p: Vec<f64> = line
                    .trim()
                    .split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect();
                jds.push(p[0]);
                dirs.push(cartesian::Direction::<$Src>::new(p[1], p[2], p[3]));
            }
            for i in 0..n.min(100) {
                let jd = JulianDate::new(jds[i]);
                let d: cartesian::Direction<$Dst> = dirs[i].to_frame(&jd);
                std::hint::black_box(&d);
            }
            let start = Instant::now();
            let mut sink = 0.0_f64;
            for i in 0..n {
                let jd = JulianDate::new(jds[i]);
                let d: cartesian::Direction<$Dst> = dirs[i].to_frame(&jd);
                sink = d.x();
            }
            let elapsed = start.elapsed();
            let total_ns = elapsed.as_nanos() as f64;
            println!(
                concat!(
                    "{{\"experiment\":\"",
                    $exp,
                    "_perf\",\"library\":\"siderust\",",
                    "\"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},",
                    "\"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}"
                ),
                n,
                total_ns,
                total_ns / n as f64,
                n as f64 / (total_ns * 1e-9),
                sink,
            );
        }
    };
}

macro_rules! inv_ecl_tod_experiment {
    ($fn_acc:ident, $fn_perf:ident, $exp:expr, $model:expr, $Dst:ty, $to_dst:ident) => {
        pub fn $fn_acc(lines: &mut impl Iterator<Item = String>) {
            let n: usize = lines.next().unwrap().trim().parse().unwrap();
            let stdout = io::stdout();
            let mut out = stdout.lock();
            write!(
                out,
                "{{\"experiment\":\"{}\",\"library\":\"siderust\",\
                 \"model\":\"{}\",\"count\":{},\"cases\":[\n",
                $exp, $model, n
            )
            .unwrap();
            for i in 0..n {
                let line = lines.next().unwrap();
                let p: Vec<f64> = line
                    .trim()
                    .split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect();
                let jd = JulianDate::new(p[0]);
                let src = cartesian::Direction::<EclipticTrueOfDate>::new(p[1], p[2], p[3]);
                let vin = src.as_vec3();
                let dst: cartesian::Direction<$Dst> = src.$to_dst(&jd);
                let vout = dst.as_vec3();
                let bk = DirectionAstroExt::to_ecliptic_of_date(&dst, &jd);
                let cl = src.angle_to(&bk);
                if i > 0 {
                    write!(out, ",\n").unwrap();
                }
                write!(
                    out,
                    "{{\"jd_tt\":{:.15},\"input\":[{:.17e},{:.17e},{:.17e}],\
                     \"output\":[{:.17e},{:.17e},{:.17e}],\"closure_rad\":{:.17e}}}",
                    p[0], vin[0], vin[1], vin[2], vout[0], vout[1], vout[2], cl,
                )
                .unwrap();
            }
            writeln!(out, "\n]}}").unwrap();
        }

        pub fn $fn_perf(lines: &mut impl Iterator<Item = String>) {
            let n: usize = lines.next().unwrap().trim().parse().unwrap();
            let mut jds = Vec::with_capacity(n);
            let mut dirs: Vec<cartesian::Direction<EclipticTrueOfDate>> = Vec::with_capacity(n);
            for _ in 0..n {
                let line = lines.next().unwrap();
                let p: Vec<f64> = line
                    .trim()
                    .split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect();
                jds.push(p[0]);
                dirs.push(cartesian::Direction::<EclipticTrueOfDate>::new(
                    p[1], p[2], p[3],
                ));
            }
            for i in 0..n.min(100) {
                let jd = JulianDate::new(jds[i]);
                let d: cartesian::Direction<$Dst> = dirs[i].$to_dst(&jd);
                std::hint::black_box(&d);
            }
            let start = Instant::now();
            let mut sink = 0.0_f64;
            for i in 0..n {
                let jd = JulianDate::new(jds[i]);
                let d: cartesian::Direction<$Dst> = dirs[i].$to_dst(&jd);
                sink = d.x();
            }
            let elapsed = start.elapsed();
            let total_ns = elapsed.as_nanos() as f64;
            println!(
                "{{\"experiment\":\"{}_perf\",\"library\":\"siderust\",\
                 \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
                 \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
                $exp,
                n,
                total_ns,
                total_ns / n as f64,
                n as f64 / (total_ns * 1e-9),
                sink,
            );
        }
    };
}

dir_experiment!(
    run_frame_bias,
    run_frame_bias_perf,
    "frame_bias",
    "IERS2003_frame_bias",
    ICRS,
    EquatorialMeanJ2000
);
dir_experiment!(
    run_precession,
    run_precession_perf,
    "precession",
    "IAU_2006_precession",
    EquatorialMeanJ2000,
    EquatorialMeanOfDate
);
dir_experiment!(
    run_nutation,
    run_nutation_perf,
    "nutation",
    "IAU_2000B_nutation",
    EquatorialMeanOfDate,
    EquatorialTrueOfDate
);

pub fn run_icrs_ecl_j2000(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"icrs_ecl_j2000\",\"library\":\"siderust\",\
         \"model\":\"ICRS_to_EclipticMeanJ2000\",\
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

        let dir_ecl: cartesian::Direction<EclipticMeanJ2000> = dir_icrs.to_frame(&jd);
        let vout = dir_ecl.as_vec3();

        let ecl_lon = vout[1].atan2(vout[0]).rem_euclid(std::f64::consts::TAU);
        let ecl_lat = vout[2].clamp(-1.0, 1.0).asin();

        let dir_back: cartesian::Direction<ICRS> = dir_ecl.to_frame(&jd);
        let closure_rad = dir_icrs.angle_to(&dir_back);

        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        write!(
            out,
            "{{\"jd_tt\":{:.15},\"input\":[{:.17e},{:.17e},{:.17e}],\
             \"output\":[{:.17e},{:.17e},{:.17e}],\
             \"ecl_lon_rad\":{:.17e},\"ecl_lat_rad\":{:.17e},\
             \"closure_rad\":{:.17e}}}",
            jd_tt, vin[0], vin[1], vin[2], vout[0], vout[1], vout[2], ecl_lon, ecl_lat, closure_rad,
        )
        .unwrap();
    }
    writeln!(out, "\n]}}").unwrap();
}

pub fn run_icrs_ecl_j2000_perf(lines: &mut impl Iterator<Item = String>) {
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

    for i in 0..n.min(100) {
        let jd = JulianDate::new(jds[i]);
        let d: cartesian::Direction<EclipticMeanJ2000> = dirs[i].to_frame(&jd);
        std::hint::black_box(&d);
    }

    let start = Instant::now();
    let mut sink = 0.0_f64;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let d: cartesian::Direction<EclipticMeanJ2000> = dirs[i].to_frame(&jd);
        sink = d.x();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;

    println!(
        "{{\"experiment\":\"icrs_ecl_j2000_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        total_ns / n as f64,
        n as f64 / (total_ns * 1e-9),
        sink,
    );
}

pub fn run_icrs_ecl_tod(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"icrs_ecl_tod\",\"library\":\"siderust\",\
         \"model\":\"IAU_2006_ecliptic_of_date\",\
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

        let spherical_icrs = affn::spherical::Direction::<ICRS>::new_raw(
            Degrees::new(dec_rad.to_degrees()),
            Degrees::new(ra_rad.to_degrees()),
        );
        let icrs_direction = spherical_icrs.to_cartesian();

        let ecliptic_direction = DirectionAstroExt::to_ecliptic_of_date(&icrs_direction, &jd);
        let ecliptic_spherical = ecliptic_direction.to_spherical();
        let ecl_lon = Radians::from(ecliptic_spherical.azimuth);
        let ecl_lat = Radians::from(ecliptic_spherical.polar);

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

pub fn run_icrs_ecl_tod_perf(lines: &mut impl Iterator<Item = String>) {
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

    for i in 0..n.min(100) {
        let jd = JulianDate::new(jds[i]);
        let sph = affn::spherical::Direction::<ICRS>::new_raw(
            Degrees::new(decs[i].to_degrees()),
            Degrees::new(ras[i].to_degrees()),
        );
        let d = sph.to_cartesian();
        let res = d.to_ecliptic_of_date(&jd);
        std::hint::black_box(&res);
    }

    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let jd = JulianDate::new(jds[i]);
        let sph = affn::spherical::Direction::<ICRS>::new_raw(
            Degrees::new(decs[i].to_degrees()),
            Degrees::new(ras[i].to_degrees()),
        );
        let d = sph.to_cartesian();
        let ecl = d.to_ecliptic_of_date(&jd);
        let ecl_sph = ecl.to_spherical();
        let lon = Radians::from(ecl_sph.azimuth);
        sink += lon.value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;

    println!(
        "{{\"experiment\":\"icrs_ecl_tod_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        total_ns / n as f64,
        n as f64 / (total_ns * 1e-9),
        sink,
    );
}

pub fn run_horiz_to_equ(lines: &mut impl Iterator<Item = String>) {
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "{{\"experiment\":\"horiz_to_equ\",\"library\":\"siderust\",\
         \"model\":\"siderust_horizontal_inverse\",\
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
        let jd_ut1_val = parts[0];
        let jd_tt_val = parts[1];
        let az_rad = parts[2];
        let alt_rad = parts[3];
        let obs_lon_rad = parts[4];
        let obs_lat_rad = parts[5];

        let jd_ut1 = JulianDate::new(jd_ut1_val);
        let jd_tt = JulianDate::new(jd_tt_val);
        let site = ObserverSite::new(
            Degrees::new(obs_lon_rad.to_degrees()),
            Degrees::new(obs_lat_rad.to_degrees()),
            Meters::new(0.0),
        );

        let sph_hor = affn::spherical::Direction::<Horizontal>::new_raw(
            Degrees::new(alt_rad.to_degrees()),
            Degrees::new(az_rad.to_degrees()),
        );
        let hor_dir = sph_hor.to_cartesian();

        let equ_dir: cartesian::Direction<EquatorialTrueOfDate> =
            hor_dir.to_equatorial(&jd_ut1, &jd_tt, &site);
        let equ_sph = equ_dir.to_spherical();
        let ra = Radians::from(equ_sph.azimuth);
        let dec = Radians::from(equ_sph.polar);

        let hor_back = DirectionAstroExt::to_horizontal_precise(&equ_dir, &jd_tt, &jd_ut1, &site);
        let equ_back: cartesian::Direction<EquatorialTrueOfDate> =
            hor_back.to_equatorial(&jd_ut1, &jd_tt, &site);
        let equ_back_sph = equ_back.to_spherical();
        let ra_back = Radians::from(equ_back_sph.azimuth);
        let dec_back = Radians::from(equ_back_sph.polar);

        let v_in = [
            dec.value().cos() * ra.value().cos(),
            dec.value().cos() * ra.value().sin(),
            dec.value().sin(),
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
             \"az_rad\":{:.17e},\"alt_rad\":{:.17e},\
             \"obs_lon_rad\":{:.17e},\"obs_lat_rad\":{:.17e},\
             \"ra_rad\":{:.17e},\"dec_rad\":{:.17e},\
             \"closure_rad\":{:.17e}}}",
            jd_ut1_val,
            jd_tt_val,
            az_rad,
            alt_rad,
            obs_lon_rad,
            obs_lat_rad,
            ra.value(),
            dec.value(),
            closure_rad,
        )
        .unwrap();
    }
    writeln!(out, "\n]}}").unwrap();
}

pub fn run_horiz_to_equ_perf(lines: &mut impl Iterator<Item = String>) {
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

    for i in 0..n.min(100) {
        let (jd_ut1_v, jd_tt_v, az, alt, lon, lat) = params[i];
        let jd_ut1 = JulianDate::new(jd_ut1_v);
        let jd_tt = JulianDate::new(jd_tt_v);
        let site = ObserverSite::new(
            Degrees::new(lon.to_degrees()),
            Degrees::new(lat.to_degrees()),
            Meters::new(0.0),
        );
        let sph = affn::spherical::Direction::<Horizontal>::new_raw(
            Degrees::new(alt.to_degrees()),
            Degrees::new(az.to_degrees()),
        );
        let hor = sph.to_cartesian();
        let d: cartesian::Direction<EquatorialTrueOfDate> =
            hor.to_equatorial(&jd_ut1, &jd_tt, &site);
        std::hint::black_box(&d);
    }

    let start = Instant::now();
    let mut sink: f64 = 0.0;
    for i in 0..n {
        let (jd_ut1_v, jd_tt_v, az, alt, lon, lat) = params[i];
        let jd_ut1 = JulianDate::new(jd_ut1_v);
        let jd_tt = JulianDate::new(jd_tt_v);
        let site = ObserverSite::new(
            Degrees::new(lon.to_degrees()),
            Degrees::new(lat.to_degrees()),
            Meters::new(0.0),
        );
        let sph = affn::spherical::Direction::<Horizontal>::new_raw(
            Degrees::new(alt.to_degrees()),
            Degrees::new(az.to_degrees()),
        );
        let hor = sph.to_cartesian();
        let equ: cartesian::Direction<EquatorialTrueOfDate> =
            hor.to_equatorial(&jd_ut1, &jd_tt, &site);
        let equ_sph = equ.to_spherical();
        let ra = Radians::from(equ_sph.azimuth);
        let dec = Radians::from(equ_sph.polar);
        sink += ra.value() + dec.value();
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as f64;

    println!(
        "{{\"experiment\":\"horiz_to_equ_perf\",\"library\":\"siderust\",\
         \"count\":{},\"total_ns\":{:.0},\"per_op_ns\":{:.1},\
         \"throughput_ops_s\":{:.0},\"_sink\":{:.17e}}}",
        n,
        total_ns,
        total_ns / n as f64,
        n as f64 / (total_ns * 1e-9),
        sink,
    );
}

dir_experiment!(
    run_inv_frame_bias,
    run_inv_frame_bias_perf,
    "inv_frame_bias",
    "IERS2003_inv_bias",
    EquatorialMeanJ2000,
    ICRS
);
dir_experiment!(
    run_inv_precession,
    run_inv_precession_perf,
    "inv_precession",
    "IAU2006_inv_prec",
    EquatorialMeanOfDate,
    EquatorialMeanJ2000
);
dir_experiment!(
    run_inv_nutation,
    run_inv_nutation_perf,
    "inv_nutation",
    "IAU2000A_inv_nut",
    EquatorialTrueOfDate,
    EquatorialMeanOfDate
);
dir_experiment!(
    run_inv_bpn,
    run_inv_bpn_perf,
    "inv_bpn",
    "IAU2006_inv_bpn",
    EquatorialTrueOfDate,
    ICRS
);
dir_experiment!(
    run_inv_icrs_ecl_j2000,
    run_inv_icrs_ecl_j2000_perf,
    "inv_icrs_ecl_j2000",
    "inv_ICRS_EclMeanJ2000",
    EclipticMeanJ2000,
    ICRS
);
dir_experiment!(
    run_obliquity,
    run_obliquity_perf,
    "obliquity",
    "obliquity_EclMeanJ2000_EqMeanJ2000",
    EclipticMeanJ2000,
    EquatorialMeanJ2000
);
dir_experiment!(
    run_inv_obliquity,
    run_inv_obliquity_perf,
    "inv_obliquity",
    "inv_obliquity_EqMeanJ2000_EclMeanJ2000",
    EquatorialMeanJ2000,
    EclipticMeanJ2000
);
dir_experiment!(
    run_bias_precession,
    run_bias_precession_perf,
    "bias_precession",
    "IAU2006_bias_prec",
    ICRS,
    EquatorialMeanOfDate
);
dir_experiment!(
    run_inv_bias_precession,
    run_inv_bias_precession_perf,
    "inv_bias_precession",
    "IAU2006_inv_bias_prec",
    EquatorialMeanOfDate,
    ICRS
);
dir_experiment!(
    run_precession_nutation,
    run_precession_nutation_perf,
    "precession_nutation",
    "IAU2006_prec_nut",
    EquatorialMeanJ2000,
    EquatorialTrueOfDate
);
dir_experiment!(
    run_inv_precession_nutation,
    run_inv_precession_nutation_perf,
    "inv_precession_nutation",
    "IAU2006_inv_prec_nut",
    EquatorialTrueOfDate,
    EquatorialMeanJ2000
);

inv_ecl_tod_experiment!(
    run_inv_icrs_ecl_tod,
    run_inv_icrs_ecl_tod_perf,
    "inv_icrs_ecl_tod",
    "IAU2006_inv_ecl_tod",
    ICRS,
    to_icrs
);
inv_ecl_tod_experiment!(
    run_inv_equ_ecl,
    run_inv_equ_ecl_perf,
    "inv_equ_ecl",
    "IAU2006_inv_equ_ecl",
    EquatorialMeanOfDate,
    to_equatorial_mean_of_date
);
