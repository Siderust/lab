# gmst_era — Summary

Timestamp: 2026-02-13_12-53-28

### GMST / ERA (Time Scales)

| Library | GMST p50 (arcsec) | GMST p99 (arcsec) | GMST max (arcsec) | ERA p50 (rad) | ERA max (rad) |
|---------|-------------------|-------------------|-------------------|---------------|---------------|
| siderust | 0.037944 | 0.284093 | 0.285826 | 0.0000000000 | 0.0000000000 |
| astropy | 0.000000 | 0.000000 | 0.000000 | 0.0000000000 | 0.0000000000 |
| libnova | 0.037945 | 0.284093 | 0.285826 | — | — |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| gmst_era | GMST=IAU2006 (eraGmst06), ERA=IAU2000 (eraEra00) | GMST=IAU2006, ERA=IAU2000 via bundled ERFA | GMST=Meeus Formula 11.4, GAST=MST+nutation corr... | GST polynomial (IAU 2006 coefficients), ERA fro... |


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
    "erfa": "GMST=IAU2006 (eraGmst06), ERA=IAU2000 (eraEra00)",
    "siderust": "GST polynomial (IAU 2006 coefficients), ERA from IERS definition",
    "astropy": "GMST=IAU2006, ERA=IAU2000 via bundled ERFA",
    "libnova": "GMST=Meeus Formula 11.4, GAST=MST+nutation correction (no ERA)"
  },
  "mode": "common_denominator"
}
```
