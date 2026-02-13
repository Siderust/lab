# solar_position — Summary

Date: 2026-02-13

### Sun Geocentric Position

| Library | Sep p50 (arcsec) | Sep p99 (arcsec) | Sep max (arcsec) | RA bias (arcsec) | Dec bias (arcsec) |
|---------|------------------|------------------|------------------|------------------|-------------------|
| siderust | 2094.3027 | 4804.1631 | 4940.4208 | -74.0741 | 35.5080 |
| astropy | 0.0000 | 0.0031 | 0.0031 | 0.0000 | 0.0000 |
| libnova | 17.0015 | 45.7365 | 48.0742 | -0.8259 | -2.1562 |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| solar_position | VSOP87 via eraEpv00: heliocentric Earth → geoce... | VSOP87 via erfa.epv00 (same as ERFA) | VSOP87 via ln_get_solar_equ_coords (different t... | VSOP87 via Sun::get_apparent_geocentric_equ (in... |


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
  "ephemeris_source": "VSOP87 (analytic, all libraries)",
  "models": {
    "erfa": "VSOP87 via eraEpv00: heliocentric Earth \u2192 geocentric Sun (negate); BCRS equatorial output",
    "siderust": "VSOP87 via Sun::get_apparent_geocentric_equ (includes aberration + FK5)",
    "astropy": "VSOP87 via erfa.epv00 (same as ERFA)",
    "libnova": "VSOP87 via ln_get_solar_equ_coords (different truncation/corrections)"
  },
  "note": "ERFA epv00 returns BCRS-aligned equatorial (no obliquity rotation). Differences reflect VSOP87 truncation levels and aberration correction details."
}
```
