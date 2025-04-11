use siderust::bodies::Moon;
use siderust::bodies::stars;
use siderust::bodies::planets;
use siderust::calculus::vsop87::VSOP87;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::{io, fs, env};
use std::path::{PathBuf};
use siderust::coordinates::{CartesianCoord, centers::*, frames::*};

fn export_barycentric(
    body: &dyn VSOP87,
    start: f64,
    end: f64,
    step: f64,
    filename: &PathBuf,
) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "JD,X_AU,Y_AU,Z_AU")?;

    let mut jd = start;
    while jd <= end {
        let coords: CartesianCoord<Barycentric, ICRS> = (&body.vsop87e(jd)).into();
        writeln!(writer, "{:.2},{:.17},{:.17},{:.17}", jd, coords.x(), coords.y(), coords.z())?;
        jd += step;
    }

    Ok(())
}

fn export_heliocentric(
    body: &dyn VSOP87,
    start: f64,
    end: f64,
    step: f64,
    filename: &PathBuf,
) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "JD,X_AU,Y_AU,Z_AU")?;

    let mut jd = start;
    while jd <= end {
        let coords: CartesianCoord<Heliocentric, Ecliptic> = body.vsop87a(jd);
        writeln!(writer, "{:.2},{:.17},{:.17},{:.17}", jd, coords.x(), coords.y(), coords.z())?;
        jd += step;
    }

    Ok(())
}


fn export_geocentric(
    body: &dyn VSOP87,
    start: f64,
    end: f64,
    step: f64,
    filename: &PathBuf,
) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "JD,X_AU,Y_AU,Z_AU")?;

    let mut jd = start;
    while jd <= end {
        let coords: CartesianCoord<Geocentric, Equatorial> = (&body.vsop87e(jd)).into();
        writeln!(writer, "{:.2},{:.17},{:.17},{:.17}", jd, coords.x(), coords.y(), coords.z())?;
        jd += step;
    }

    Ok(())
}


fn create_directory(path: &PathBuf) -> io::Result<()> {
    fs::create_dir_all(path)?;
    println!("Directory created (or already exists): {}", path.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root_str = env::var("SIDERUST_LAB_ROOT")
        .expect("SIDERUST_LAB_ROOT is not set");

    let base_path = PathBuf::from(&root_str).join("siderust/dataset");
    let helio_path = base_path.join("helio");
    let icrs_path = base_path.join("icrs");
    let geo_path = base_path.join("geo");

    create_directory(&helio_path)?;
    create_directory(&icrs_path)?;

    let start_jd = 2460676.5;
    let end_jd   = 2461041.5;
    let step: f64 = 0.25;

    let all_bodies: Vec<(&str, Box<dyn VSOP87>)> = vec![
        ("sun",     Box::new(stars::Sun)),
        ("moon",    Box::new(Moon)),
        ("mercury", Box::new(planets::Mercury)),
        ("venus",   Box::new(planets::Venus)),
        ("earth",   Box::new(planets::Earth)),
        ("mars",    Box::new(planets::Mars)),
        ("jupiter", Box::new(planets::Jupiter)),
        ("saturn",  Box::new(planets::Saturn)),
        ("uranus",  Box::new(planets::Uranus)),
        ("neptune", Box::new(planets::Neptune)),
    ];

    for (body_name, planet_obj) in &all_bodies {
        let helio_file = helio_path.join(format!("{}_helio.csv", body_name));
        let icrs_file  = icrs_path.join(format!("{}_icrs.csv", body_name));
        let geo_file  = geo_path.join(format!("{}_geo.csv", body_name));

        export_heliocentric(
            planet_obj.as_ref(),
            start_jd,
            end_jd,
            step,
            &helio_file
        ).expect(&format!("Error exporting vsop87a for {}", body_name));

        export_barycentric(
            planet_obj.as_ref(),
            start_jd,
            end_jd,
            step,
            &icrs_file
        ).expect(&format!("Error exporting vsop87e for {}", body_name));

        export_geocentric(
            planet_obj.as_ref(),
            start_jd,
            end_jd,
            step,
            &geo_file
        ).expect(&format!("Error exporting vsop87e for {}", body_name));
    }

    println!("CSV files successfully created and organized by coordinate system!");
    Ok(())
}
