use siderust::time::julian_date::J2000;
use siderust::bodies::planets;
use siderust::calculus::vsop87::VSOP87;
use std::time::Instant;

fn performance_test_vsop87a(planet: &dyn VSOP87, start: f64, end: f64, step: f64) -> f64 {
    let mut jd = start;
    let mut count = 0;
    let start_time = Instant::now();

    while jd <= end {
        let _coords = planet.vsop87a(jd);
        jd += step;
        count += 1;
    }

    let duration = start_time.elapsed();
    duration.as_secs_f64() / count as f64
}

fn performance_test_vsop87e(planet: &dyn VSOP87, start: f64, end: f64, step: f64) -> f64 {
    let mut jd = start;
    let mut count = 0;
    let start_time = Instant::now();

    while jd <= end {
        let _coords = planet.vsop87e(jd);
        jd += step;
        count += 1;
    }

    let duration = start_time.elapsed();
    duration.as_secs_f64() / count as f64
}

fn main() {
    let a_year: f64 = 365.0;
    let _2025 = J2000 + 25.0 * a_year;
    let _2026 = J2000 + 26.0 * a_year;
    let step: f64 = 0.25; // 6 hours

    let planets: Vec<(&str, Box<dyn VSOP87>)> = vec![
        ("Mercury", Box::new(planets::Mercury::new())),
        ("Venus", Box::new(planets::Venus::new())),
        ("Earth", Box::new(planets::Earth::new())),
        ("Mars", Box::new(planets::Mars::new())),
        ("Jupiter", Box::new(planets::Jupiter::new())),
        ("Saturn", Box::new(planets::Saturn::new())),
        ("Uranus", Box::new(planets::Uranus::new())),
        ("Neptune", Box::new(planets::Neptune::new())),
    ];

    println!("{:<8} | {:>15} | {:>15}", "Planet", "vsop87a (s)", "vsop87e (s)");
    println!("{:-<43}", "");

    for (name, planet) in planets {
        let avg_a = performance_test_vsop87a(&*planet, _2025, _2026, step);
        let avg_e = performance_test_vsop87e(&*planet, _2025, _2026, step);
        println!("{:<8} | {:>15.9} | {:>15.9}", name, avg_a, avg_e);
    }
}
