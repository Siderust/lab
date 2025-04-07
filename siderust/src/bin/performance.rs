use siderust::bodies::planets;
use siderust::calculus::vsop87::VSOP87;
use std::time::Instant;

fn compute_coords_using_vsop87e(
    planet: &dyn VSOP87,
    start: f64,
    end: f64,
    step: f64,
) {
    let mut jd = start;
    while jd <= end {
        let _coords = planet.vsop87a(jd);
        jd += step;
    }
}


fn main() -> Result<(), Box<dyn std::error::Error>> {

    let start_jd = 2460676.5; // Start of 2025 (approx.)
    let end_jd = 2461041.5;   // Start of 2026 (approx.)
    let step: f64 = 0.25; // 6-hour increments
    let n_steps: u32 = ((end_jd - start_jd) / step) as u32;

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
        let start_time = Instant::now();
    
        compute_coords_using_vsop87e(
            planet_obj.as_ref(),
            start_jd,
            end_jd,
            step,
        );
    
        let elapsed = start_time.elapsed();
        println!("{} took {:.2?}", planet_name, elapsed / n_steps);
    }

    Ok(())
}
