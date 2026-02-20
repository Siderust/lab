# equ_horizontal — Summary

Timestamp: 2026-02-20_00-47-25

### Equatorial → Horizontal (AltAz)

| Library | Sep p50 (arcsec) | Sep p99 (arcsec) | Sep max (arcsec) | RA bias (arcsec) | Dec bias (arcsec) |
|---------|------------------|------------------|------------------|------------------|-------------------|
| siderust | 0.0000 | 0.0043 | 0.0043 | -0.0001 | -0.0000 |
| astropy | 0.0000 | 0.0043 | 0.0043 | 0.0000 | 0.0000 |
| libnova | 8.0491 | 16.3621 | 16.7349 | 0.6853 | 0.1547 |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| equ_horizontal | Spherical trig via eraHd2ae / eraAe2hd; GAST vi... | eraHd2ae / eraAe2hd via bundled ERFA; GAST via ... | ln_get_hrz_from_equ / ln_get_equ_from_hrz; conv... | Spherical trig matching ERFA formulas; GAST IAU... |


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
  "library_notes": {
    "astropy": "The 'astropy' adapter calls ERFA/pyerfa C kernels directly from Python. Accuracy results are identical to ERFA. Performance results measure Python-loop + ERFA-kernel overhead, NOT the high-level astropy.coordinates stack."
  },
  "models": {
    "erfa": "Spherical trig via eraHd2ae / eraAe2hd; GAST via eraGst06a; no refraction",
    "siderust": "Spherical trig matching ERFA formulas; GAST IAU 2006 via siderust astro path",
    "astropy": "eraHd2ae / eraAe2hd via bundled ERFA; GAST via eraGst06a",
    "libnova": "ln_get_hrz_from_equ / ln_get_equ_from_hrz; convention fix: az_erfa = (360 - az_ln + 180) % 360",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-parity",
  "accuracy_interpretation": "accuracy vs ERFA reference (same spherical trig model)",
  "note": "Azimuth convention: ERFA 0\u00b0=North CW; libnova 0\u00b0=South. All adapters use the same spherical trig, differences arise from GAST model. No atmospheric refraction applied."
}
```
