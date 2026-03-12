"""
JPL Horizons API client with persistent caching.
=================================================

Fetches astrometric geocentric RA/Dec from JPL Horizons via POST to
https://ssd.jpl.nasa.gov/api/horizons.api

Features:
  - Batches epochs in chunks of 200 via multiline TLIST
  - Parses RA, DEC, delta from $$SOE … $$EOE CSV blocks
  - Persistent cache under .cache/horizons/ keyed by experiment config + epoch hash
  - Records the Horizons source tag (e.g. DE441) from the response
"""

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# AU → km (IAU 2012 exact)
AU_KM = 149597870.700

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
BATCH_SIZE = 200

# Horizons body IDs
HORIZONS_BODIES = {
    "solar_position": "10",   # Sun
    "lunar_position": "301",  # Moon
    "mercury_position": "199",
    "venus_position": "299",
    "mars_position": "499",
    "jupiter_position": "599",
    "saturn_position": "699",
    "uranus_position": "799",
    "neptune_position": "899",
}

DIST_KM_EXPERIMENTS = {"lunar_position"}

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "horizons"


def _cache_key(experiment: str, body_id: str, epochs_sorted: list[float]) -> str:
    """Compute a deterministic cache key from experiment config and ordered epochs."""
    payload = json.dumps({
        "experiment": experiment,
        "body_id": body_id,
        "epochs": [f"{e:.15f}" for e in epochs_sorted],
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"


def _load_cache(cache_key: str) -> dict | None:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_cache(cache_key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_key)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _build_query_params(body_id: str, epoch_batch: list[float]) -> dict:
    """Build Horizons API query parameters for one batch of epochs."""
    tlist = "\n".join(f"{e:.15f}" for e in epoch_batch)
    return {
        "format": "text",
        "COMMAND": f"'{body_id}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "TIME_TYPE": "TT",
        "TLIST_TYPE": "JD",
        "TLIST": tlist,
        "REF_SYSTEM": "ICRF",
        "QUANTITIES": "'1,20'",
        "ANG_FORMAT": "DEG",
        "EXTRA_PREC": "YES",
        "CSV_FORMAT": "YES",
    }


def _parse_source_tag(response_text: str) -> str:
    """Extract the ephemeris source tag(s) from the Horizons response."""
    m = re.search(r"Ephemeris\s*/\s*(DE\d+)", response_text)
    if m:
        return m.group(1)
    m = re.search(r"Target body.*?source\s*:\s*(DE\d+)", response_text, re.IGNORECASE)
    if m:
        return m.group(1)
    tags = []
    for tag in re.findall(r"\{source\s*:\s*([A-Za-z0-9_+-]+)\}", response_text, re.IGNORECASE):
        if tag.lower() not in {seen.lower() for seen in tags}:
            tags.append(tag)
    if tags:
        return "+".join(tags)
    return "unknown"


def _parse_csv_block(response_text: str, expected_epochs: list[float] | None = None) -> list[dict]:
    """Parse $$SOE … $$EOE CSV block from Horizons observer-table output.

    Expected CSV columns (QUANTITIES='1,20'):
      either:
        JDTT, calendar_date, , RA, DEC, delta, deldot,
      or:
        calendar_date, , , RA, DEC, delta, deldot,

    With CSV_FORMAT='YES' and ANG_FORMAT='DEG', RA and DEC are in decimal degrees.
    delta is in AU.
    """
    soe_match = re.search(r"\$\$SOE\s*\n", response_text)
    eoe_match = re.search(r"\n\$\$EOE", response_text)
    if not soe_match or not eoe_match:
        raise ValueError("Could not find $$SOE/$$EOE markers in Horizons response")

    block = response_text[soe_match.end():eoe_match.start()]
    rows = []
    for row_idx, line in enumerate(block.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(",")]
        try:
            if re.match(r"^[+-]?\d+(\.\d+)?$", fields[0]):
                jd_tt = float(fields[0])
                ra_idx = 3
                dec_idx = 4
                delta_idx = 5
            else:
                if expected_epochs is None:
                    raise ValueError("Horizons CSV row omitted JD and no expected epochs were provided")
                jd_tt = float(expected_epochs[row_idx])
                ra_idx = 3
                dec_idx = 4
                delta_idx = 5

            ra_deg = float(fields[ra_idx])
            dec_deg = float(fields[dec_idx])
            delta_au = float(fields[delta_idx])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse Horizons CSV row: {line!r}"
            ) from exc

        rows.append({
            "jd_tt": jd_tt,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "delta_au": delta_au,
        })
    if expected_epochs is not None and len(rows) != len(expected_epochs):
        raise ValueError(
            f"Horizons response row count mismatch: expected {len(expected_epochs)}, got {len(rows)}"
        )
    return rows


def _fetch_batch(body_id: str, epoch_batch: list[float]) -> tuple[list[dict], str]:
    """Fetch one batch of epochs from Horizons. Returns (parsed_rows, source_tag)."""
    params = _build_query_params(body_id, epoch_batch)
    encoded = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        HORIZONS_URL,
        data=encoded.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise RuntimeError(f"Horizons API request failed: {exc}") from exc

    source_tag = _parse_source_tag(text)
    rows = _parse_csv_block(text, epoch_batch)
    return rows, source_tag


def fetch_horizons_reference(experiment: str, epochs, use_cache: bool = True) -> dict:
    """Fetch reference positions from JPL Horizons for the given experiment and epochs.

    Args:
        experiment: experiment ID, e.g. "solar_position", "lunar_position", or "mars_position"
        epochs: array-like of JD(TT) values
        use_cache: if True, use persistent cache

    Returns:
        dict with keys:
          - "cases": list of dicts with jd_tt, ra_rad, dec_rad, dist_au, dist_km
          - "source_tag": str like "DE441"
          - "cache_key": str
          - "from_cache": bool
          - "query_params": dict (representative)
          - "fetch_timestamp": str or None

    Raises:
        RuntimeError: on fetch failure (cache miss + network error)
        KeyError: on unknown experiment
    """
    body_id = HORIZONS_BODIES.get(experiment)
    if body_id is None:
        raise KeyError(f"No Horizons body mapping for experiment '{experiment}'")

    epoch_list = [float(e) for e in epochs]

    # Compute cache key from sorted unique epochs, but we need to preserve order + dupes
    sorted_for_key = sorted(set(epoch_list))
    key = _cache_key(experiment, body_id, sorted_for_key)

    # Try cache
    if use_cache:
        cached = _load_cache(key)
        if cached is not None:
            # Rebuild cases in original epoch order from cached rows
            cases = _reorder_cases(cached["rows"], epoch_list, experiment)
            return {
                "cases": cases,
                "source_tag": cached.get("source_tag", "unknown"),
                "cache_key": key,
                "from_cache": True,
                "query_params": cached.get("query_params", {}),
                "fetch_timestamp": cached.get("fetch_timestamp"),
            }

    # Fetch from Horizons in batches
    # We only need to fetch unique epochs, then map back
    unique_epochs = sorted(set(epoch_list))
    all_rows = []
    source_tag = "unknown"

    for i in range(0, len(unique_epochs), BATCH_SIZE):
        batch = unique_epochs[i:i + BATCH_SIZE]
        rows, tag = _fetch_batch(body_id, batch)
        all_rows.extend(rows)
        source_tag = tag  # Last batch's tag (should be consistent)

    # Save to cache
    representative_params = _build_query_params(body_id, unique_epochs[:5])
    cache_payload = {
        "experiment": experiment,
        "body_id": body_id,
        "source_tag": source_tag,
        "query_params": representative_params,
        "epochs_count": len(unique_epochs),
        "rows": all_rows,
        "fetch_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_cache(key, cache_payload)

    # Map to original order
    cases = _reorder_cases(all_rows, epoch_list, experiment)

    return {
        "cases": cases,
        "source_tag": source_tag,
        "cache_key": key,
        "from_cache": False,
        "query_params": representative_params,
        "fetch_timestamp": cache_payload["fetch_timestamp"],
    }


def _reorder_cases(rows: list[dict], epoch_list: list[float], experiment: str) -> list[dict]:
    """Reorder fetched rows to match the original epoch list, handling duplicates."""
    # Build lookup by JD (round to avoid float matching issues)
    lookup = {}
    for row in rows:
        jd_key = f"{row['jd_tt']:.10f}"
        lookup[jd_key] = row

    deg_to_rad = math.pi / 180.0
    cases = []
    for epoch in epoch_list:
        jd_key = f"{epoch:.10f}"
        row = lookup.get(jd_key)
        if row is None:
            # Try nearest match (within 1e-6 day = 0.086 seconds)
            best = None
            best_dist = 1e-6
            for r in rows:
                d = abs(r["jd_tt"] - epoch)
                if d < best_dist:
                    best_dist = d
                    best = r
            row = best

        if row is None:
            raise RuntimeError(
                f"Horizons response missing epoch JD {epoch:.15f} for {experiment}"
            )

        case = {
            "jd_tt": epoch,
            "ra_rad": row["ra_deg"] * deg_to_rad,
            "dec_rad": row["dec_deg"] * deg_to_rad,
            "dist_au": row["delta_au"],
        }
        if experiment in DIST_KM_EXPERIMENTS:
            case["dist_km"] = row["delta_au"] * AU_KM
        cases.append(case)
    return cases
