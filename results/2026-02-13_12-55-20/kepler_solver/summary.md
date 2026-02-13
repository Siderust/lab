# kepler_solver — Summary

Timestamp: 2026-02-13_12-55-20

### Kepler Solver (M→E→ν)

| Library | E p50 (rad) | E max (rad) | ν p50 (rad) | ν max (rad) | Consistency max (rad) |
|---------|-------------|-------------|-------------|-------------|-----------------------|
| siderust | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 8.88e-16 |
| astropy | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 8.88e-16 |
| libnova | 0.00e+00 | 8.88e-16 | 0.00e+00 | 1.78e-15 | 1.33e-15 |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| kepler_solver | Newton-Raphson iteration (100 iters, tol 1e-15) | Newton-Raphson iteration in Python (100 iters, ... | Sinnott bisection via ln_solve_kepler (internal... | solve_keplers_equation (internal algorithm) |


## Alignment Checklist

```json
{
  "units": {
    "angles": "radians (internal), mas for error reporting",
    "distances": "meters",
    "float_type": "f64"
  },
  "time_input": "JD (Julian Date), TT scale for precession/nutation, UT1 for sidereal time",
  "time_scales": "TT for BPN matrix; UT1\u2248TT-69.184s simplified",
  "leap_seconds": "not applicable (JD input, no UTC conversion in this experiment)",
  "earth_orientation": {
    "ut1_minus_utc": "not used (JD(TT) input)",
    "polar_motion_xp_yp": "zero (not applied)",
    "eop_mode": "disabled"
  },
  "geodesy": "not applicable (direction-only experiment)",
  "refraction": "disabled",
  "ephemeris_source": "not applicable (no aberration/parallax)",
  "models": {
    "erfa": "Newton-Raphson iteration (100 iters, tol 1e-15)",
    "siderust": "solve_keplers_equation (internal algorithm)",
    "astropy": "Newton-Raphson iteration in Python (100 iters, tol 1e-15)",
    "libnova": "Sinnott bisection via ln_solve_kepler (internal convergence ~1e-6 deg)"
  },
  "note": "Kepler's equation M = E - e*sin(E) is solved for E given (M, e). Self-consistency M_reconstructed = E - e*sin(E) is the primary metric. libnova uses a bisection method with lower convergence tolerance."
}
```
