# lunar_position — Summary

Date: 2026-02-12

### Moon Geocentric Position

| Library | Sep p50 (arcsec) | Sep p99 (arcsec) | Sep max (arcsec) | RA bias (arcsec) | Dec bias (arcsec) |
|---------|------------------|------------------|------------------|------------------|-------------------|
| siderust | 2601.3155 | 4015.0246 | 4185.9215 | -186.6608 | 24.8349 |
| astropy | 0.0246 | 0.0579 | 0.0594 | -0.0002 | -0.0001 |
| libnova | 2398.8417 | 4973.9021 | 5226.2013 | 3.0394 | 7.1779 |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| lunar_position | Simplified Meeus Ch.47 (major terms only, ~10' ... | Simplified Meeus Ch.47 (same algorithm as ERFA ... | ELP 2000-82B via ln_get_lunar_equ_coords (full ... | ELP 2000 via Moon::get_apparent_topocentric_equ... |


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
  "ephemeris_source": "Meeus/ELP 2000 (varies by library)",
  "models": {
    "erfa": "Simplified Meeus Ch.47 (major terms only, ~10' accuracy)",
    "siderust": "ELP 2000 via Moon::get_apparent_topocentric_equ with site at (0,0,0)",
    "astropy": "Simplified Meeus Ch.47 (same algorithm as ERFA adapter)",
    "libnova": "ELP 2000-82B via ln_get_lunar_equ_coords (full model)"
  },
  "note": "ERFA and Astropy use simplified Meeus (~10' accuracy) \u2014 for benchmarking only. Siderust and libnova use full ELP 2000 \u2014 expect ~arcmin-level differences vs reference. No dedicated ERFA Moon ephemeris exists; cross-library comparison is the primary metric."
}
```
