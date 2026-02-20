# nutation — Summary

Timestamp: 2026-02-20_00-47-25

### Nutation (Mean of Date → True of Date)

| Library | Ang p50 (mas) | Ang p99 (mas) | Ang max (mas) | Closure p99 (rad) | Perf (ns/op) | Speedup vs ref |
|---------|---------------|---------------|---------------|-------------------|--------------|----------------|
| siderust | 0.00 | 4.35 | 4.35 | 0.00 | 3164.8 | 24.9× |
| astropy | 0.00 | 4.35 | 5.32 | 0.00 | 78105.4 | 1.0× |
| libnova | 4.35 | 14.42 | 16.55 | 0.00 | 14297.0 | 5.5× |

### Feature / Model Parity Matrix

| Experiment | erfa | astropy | libnova | siderust |
|------------|---|---|---|---|
| nutation | IAU 2000A nutation (1365 terms) via eraNum06a | IAU 2000A nutation via bundled ERFA (erfa.num06a) | IAU 1980 nutation (69 terms) via ln_get_equ_nut... | IAU 2000B nutation (77 terms) via frame rotatio... |


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
    "erfa": "IAU 2000A nutation (1365 terms) via eraNum06a",
    "siderust": "IAU 2000B nutation (77 terms) via frame rotation provider",
    "astropy": "IAU 2000A nutation via bundled ERFA (erfa.num06a)",
    "libnova": "IAU 1980 nutation (69 terms) via ln_get_equ_nut / ln_nutation",
    "anise": "Not available in ANISE adapter for this experiment"
  },
  "model_parity_class": "model-mismatch",
  "accuracy_interpretation": "agreement with ERFA baseline (nutation models differ in term count)",
  "note": "IAU 2000A (ERFA/Astropy) has 1365 terms, 2000B (Siderust) has 77 terms (~1 mas difference), IAU 1980 (libnova) has 69 terms (~tens of mas difference)."
}
```
