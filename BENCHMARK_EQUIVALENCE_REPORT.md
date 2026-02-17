# Benchmark Equivalence Report

## Scope
- Reviewed benchmark adapters for `erfa`, `astropy`, `libnova`, and `siderust` in `pipeline/adapters/*`.
- Validated both functional paths (`<experiment>`) and performance paths (`<experiment>_perf`) used by `pipeline/orchestrator.py`.

## Comparison Matrix

| Tool | Operation | Inputs | Units | Setup Included? | Iterations | Notes |
| ---- | --------- | ------ | ----- | --------------- | ---------- | ----- |
| erfa | frame_rotation_bpn_perf | `jd_tt`, unit vector | JD, rad | No (parse excluded) | warmup=`min(N,100)`, timed=`N` | IAU 2006/2000A BPN |
| astropy | frame_rotation_bpn_perf | `jd_tt`, unit vector | JD, rad | No | warmup=`min(N,100)`, timed=`N` | via ERFA `pnm06a` |
| libnova | frame_rotation_bpn_perf | `jd_tt`, unit vector | JD, rad | No | warmup=`min(N,100)`, timed=`N` | Meeus precession + IAU1980 nutation |
| siderust | frame_rotation_bpn_perf | `jd_tt`, unit vector | JD, rad | No | warmup=`min(N,100)`, timed=`N` | siderust frame rotation pipeline |
| erfa | gmst_era_perf | `jd_ut1`, `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | Timed on GMST path |
| astropy | gmst_era_perf | `jd_ut1`, `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | Timed on GMST path |
| libnova | gmst_era_perf | `jd_ut1`, `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | Timed on libnova sidereal path |
| siderust | gmst_era_perf | `jd_ut1`, `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | Timed on siderust sidereal path |
| erfa | equ_ecl_perf | `jd_tt`, `ra`, `dec` | JD, rad | No | warmup=`min(N,100)`, timed=`N` | Timed on `Eqec06` transform |
| astropy | equ_ecl_perf | `jd_tt`, `ra`, `dec` | JD, rad | No | warmup=`min(N,100)`, timed=`N` | Timed on `erfa.eqec06` |
| libnova | equ_ecl_perf | `jd_tt`, `ra`, `dec` | JD, rad in input; deg API internally | No (deg preconvert excluded) | warmup=`min(N,100)`, timed=`N` | Timed on `ln_get_ecl_from_equ` |
| siderust | equ_ecl_perf | `jd_tt`, `ra`, `dec` | JD, rad in input; deg API internally | No (deg preconvert excluded) | warmup=`min(N,100)`, timed=`N` | Timed on `GCRS -> Ecliptic` transform |
| erfa | equ_horizontal_perf | `jd_ut1`,`jd_tt`,`ra`,`dec`,`lon`,`lat` | JD, rad | No | warmup=`min(N,100)`, timed=`N` | GAST (`gst06a`) + `hd2ae` |
| astropy | equ_horizontal_perf | `jd_ut1`,`jd_tt`,`ra`,`dec`,`lon`,`lat` | JD, rad | No | warmup=`min(N,100)`, timed=`N` | GAST (`gst06a`) + `hd2ae` |
| libnova | equ_horizontal_perf | `jd_ut1`,`jd_tt`,`ra`,`dec`,`lon`,`lat` | JD, rad input; deg API internally | No (deg preconvert excluded) | warmup=`min(N,100)`, timed=`N` | `ln_get_hrz_from_equ` |
| siderust | equ_horizontal_perf | `jd_ut1`,`jd_tt`,`ra`,`dec`,`lon`,`lat` | JD, rad | No | warmup=`min(N,100)`, timed=`N` | siderust sidereal + spherical trig |
| erfa | solar_position_perf | `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | Full epv00 -> RA/Dec/dist extraction |
| astropy | solar_position_perf | `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | Full epv00 -> RA/Dec/dist extraction |
| libnova | solar_position_perf | `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | RA/Dec + distance path |
| siderust | solar_position_perf | `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | `Sun::get_apparent_geocentric_equ` |
| erfa | lunar_position_perf | `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | Simplified Meeus Ch.47 path |
| astropy | lunar_position_perf | `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | Simplified Meeus Ch.47 path |
| libnova | lunar_position_perf | `jd_tt` | JD | No | warmup=`min(N,100)`, timed=`N` | ELP2000 + distance path |
| siderust | lunar_position_perf | `jd_tt` | JD | No (site fixed outside loop) | warmup=`min(N,100)`, timed=`N` | siderust lunar topocentric at geocenter site |
| erfa | kepler_solver_perf | `M`, `e` | rad, unitless | No | warmup=`min(N,100)`, timed=`N` | Newton settings match functional path |
| astropy | kepler_solver_perf | `M`, `e` | rad, unitless | No | warmup=`min(N,100)`, timed=`N` | Newton settings match functional path |
| libnova | kepler_solver_perf | `M`, `e` | rad input; deg API internally | No (rad->deg preconvert excluded) | warmup=`min(N,100)`, timed=`N` | `ln_solve_kepler` with correct units |
| siderust | kepler_solver_perf | `M`, `e` | rad, unitless | No | warmup=`min(N,100)`, timed=`N` | `solve_keplers_equation` |

## Differences Found
- `equ_ecl_perf` had non-equivalent work across tools (matrix build vs coordinate transform).
- `kepler_solver_perf` in libnova used wrong units (radians passed to degree API).
- Unit-conversion overhead was inconsistently included in timed loops for libnova/siderust in some perf paths.
- Solar/lunar perf scopes were inconsistent in whether RA/Dec/dist extraction was included.
- Remaining model-level differences exist across libraries (see limitations).

## Fixes Applied
- Aligned `equ_ecl_perf` to benchmark actual equatorial->ecliptic transforms across tools.
- Corrected libnova Kepler perf input units by pre-converting `M` radians -> degrees.
- Moved libnova/siderust pre-conversion setup outside timed sections where applicable.
- Aligned solar/lunar perf scope to include full per-case computation paths used by each adapter's functional experiment.
- Kepler Newton perf settings kept consistent with functional solver settings where Newton implementation is used.

## Remaining Limitations
- Library model parity is not exact for some experiments by design/capability:
  - `frame_rotation_bpn`: ERFA/Astropy IAU2006/2000A vs libnova/siderust Meeus/IAU1980-style paths.
  - `lunar_position`: ERFA/Astropy simplified Meeus vs libnova/siderust ELP-based implementations.
  - `gmst_era`: libnova lacks explicit ERA output path used in ERFA/Astropy.
- These are model/capability differences, not harness-timing mismatches.
