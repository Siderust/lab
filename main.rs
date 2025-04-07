use siderust::bodies::planets;
use siderust::calculus::vsop87::VSOP87;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::{io, fs};

fn export_coords_to_csv_vsop87a(
    planet: &dyn VSOP87,
    start: f64,
    end: f64,
    step: f64,
    filename: &str,
) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "jd,x,y,z")?;

    let mut jd = start;
    while jd <= end {
        let coords = planet.vsop87a(jd);
        writeln!(writer, "{},{},{},{}", jd, coords.x(), coords.y(), coords.z())?;
        jd += step;
    }

    Ok(())
}

fn export_coords_to_csv_vsop87e(
    planet: &dyn VSOP87,
    start: f64,
    end: f64,
    step: f64,
    filename: &str,
) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "jd,x,y,z")?;

    let mut jd = start;
    while jd <= end {
        let coords = planet.vsop87e(jd);
        writeln!(writer, "{},{},{},{}", jd, coords.x(), coords.y(), coords.z())?;
        jd += step;
    }

    Ok(())
}

fn create_directory(path: &str) -> io::Result<()> {
    fs::create_dir_all(path)?;
    println!("Directory created (or already exists): {}", path);
    Ok(())
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    create_directory("/home/user/src/siderust/dataset/")?;

    let start_jd = 2460676.5; // Start of 2025 (approx.)
    let end_jd = 2461041.5;   // Start of 2026 (approx.)
    let step: f64 = 0.25; // 6-hour increments

    // Gather all major planets in a vector along with a friendly name.
    // If you also want Pluto or other objects, you can add them here.
    let all_planets: Vec<(&str, Box<dyn VSOP87>)> = vec![
        ("mercury", Box::new(planets::Mercury::new())),
        ("venus",   Box::new(planets::Venus::new())),
        ("earth",   Box::new(planets::Earth::new())),
        ("mars",    Box::new(planets::Mars::new())),
        ("jupiter", Box::new(planets::Jupiter::new())),
        ("saturn",  Box::new(planets::Saturn::new())),
        ("uranus",  Box::new(planets::Uranus::new())),
        ("neptune", Box::new(planets::Neptune::new())),
    ];

    for (planet_name, planet_obj) in &all_planets {
        let filename_a = format!("/home/user/src/siderust/dataset/{}_heliocentric_ecliptic.csv", planet_name);
        let filename_e = format!("/home/user/src/siderust/dataset/{}_barycentric_equatorial.csv", planet_name);

        export_coords_to_csv_vsop87a(
            planet_obj.as_ref(),
            start_jd,
            end_jd,
            step,
            &filename_a
        ).expect("Error exporting vsop87a");

        export_coords_to_csv_vsop87e(
            planet_obj.as_ref(),
            start_jd,
            end_jd,
            step,
            &filename_e
        ).expect("Error exporting vsop87e");
    }

    println!("CSV files successfully created for all planets!");
    Ok(())
}
