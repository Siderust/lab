# equ_ecl — Summary

Date: 2026-02-12

### Equatorial ↔ Ecliptic Transform

| Library | Sep p50 (arcsec) | Sep p99 (arcsec) | Sep max (arcsec) | RA bias (arcsec) | Dec bias (arcsec) |
|---------|------------------|------------------|------------------|------------------|-------------------|
| siderust | 1815.1622 | 4731.6916 | 4995.4930 | -2486.9524 | -0.0418 |
| astropy | 0.0000 | 0.0031 | 0.0043 | 0.0000 | 0.0000 |
| libnova | 1815.1513 | 4731.6894 | 4995.5012 | -2486.9462 | -0.0422 |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| equ_ecl | IAU 2006 obliquity-based transform (eraEqec06 /... | IAU 2006 via bundled ERFA (erfa.eqec06 / erfa.e... | Meeus obliquity (Eq 22.2) via ln_get_ecl_from_e... | IAU 2006 obliquity, Ecliptic frame via Transfor... |


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
    "erfa": "IAU 2006 obliquity-based transform (eraEqec06 / eraEceq06)",
    "siderust": "IAU 2006 obliquity, Ecliptic frame via Transform trait",
    "astropy": "IAU 2006 via bundled ERFA (erfa.eqec06 / erfa.eceq06)",
    "libnova": "Meeus obliquity (Eq 22.2) via ln_get_ecl_from_equ / ln_get_equ_from_ecl"
  },
  "note": "ERFA and Astropy share the same IAU 2006 obliquity model (reference). Siderust uses IAU 2006 obliquity with its own Ecliptic frame implementation. libnova uses Meeus obliquity polynomial \u2014 expect ~arcsec-level differences."
}
```
