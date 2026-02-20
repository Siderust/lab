# precession — Summary

Timestamp: 2026-02-20_00-47-25

### Precession (Mean J2000 → Mean of Date)

| Library | Ang p50 (mas) | Ang p99 (mas) | Ang max (mas) | Closure p99 (rad) | Perf (ns/op) | Speedup vs ref |
|---------|---------------|---------------|---------------|-------------------|--------------|----------------|
| siderust | 0.00 | 4.35 | 5.32 | 0.00 | 66.2 | 1.2× |
| astropy | 0.00 | 4.35 | 5.32 | 0.00 | 2359.2 | 0.0× |
| libnova | 206.27 | 1170.68 | 1291.00 | 0.00 | 520.0 | 0.1× |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| precession | IAU 2006 precession matrix from eraPmat06 | IAU 2006 precession via bundled ERFA (erfa.pmat06) | Meeus precession (ζ,z,θ Equ 20.3) via ln_get_eq... | IAU 2006 precession via frame rotation provider... |


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
    "erfa": "IAU 2006 precession matrix from eraPmat06",
    "siderust": "IAU 2006 precession via frame rotation provider (EquatorialMeanJ2000 \u2192 EquatorialMeanOfDate)",
    "astropy": "IAU 2006 precession via bundled ERFA (erfa.pmat06)",
    "libnova": "Meeus precession (\u03b6,z,\u03b8 Equ 20.3) via ln_get_equ_prec2",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-mismatch",
  "accuracy_interpretation": "agreement with ERFA baseline (libnova uses Meeus model)",
  "note": "ERFA, Astropy, and Siderust all use IAU 2006 precession. libnova uses Meeus precession formulae \u2014 expect arcsec-level differences."
}
```
