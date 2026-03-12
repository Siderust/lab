"""
Tests for JPL Horizons client and external-reference orchestrator path.
=====================================================================

Unit tests use canned text fixtures — no live network requests.
Orchestrator tests mock the Horizons fetcher to verify:
  - reference_library == "jpl_horizons"
  - erfa appears as a candidate for solar/lunar
  - reference_performance is empty
  - cache hit path skips network
  - cache miss + fetch failure aborts only affected experiment
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

TEST_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TEST_DIR.parent
LAB_ROOT = PIPELINE_DIR.parent

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(LAB_ROOT))

from horizons_client import (
    _parse_csv_block,
    _parse_source_tag,
    _reorder_cases,
    _cache_key,
    _cache_path,
    _load_cache,
    _save_cache,
    _build_query_params,
    fetch_horizons_reference,
    BATCH_SIZE,
    AU_KM,
    CACHE_DIR,
    HORIZONS_BODIES,
)


# ---------------------------------------------------------------------------
# Canned Horizons response fixture
# ---------------------------------------------------------------------------

SAMPLE_HORIZONS_RESPONSE = """\
*******************************************************************************
JPL/HORIZONS                      Sun (10)              2026-Mar-11 00:00:00
Rec #:10      Soln.date: 2025-Jan-01_00:00:00   # obs: 9999 (all types)

*******************************************************************************
Ephemeris / PORT_LOGIN Tue Mar 11 00:00:00 2026 Jpl/Horizons
Target body name: Sun (10)                    {source: DE441}
Center body name: Earth (399)                 {source: DE441}
*******************************************************************************
$$SOE
2451545.000000000, 2000-Jan-01 12:00:00.0000,  ,281.28783929, -23.01394503,  0.983320046, -0.0001234,
2460310.500000000, 2024-Jan-01 00:00:00.0000,  ,280.87654321, -22.98765432,  0.983412345,  0.0000567,
2451625.000000000, 2000-Mar-20 12:00:00.0000,  ,359.94567890,  -0.01234567,  0.995678901,  0.0002345,
$$EOE
*******************************************************************************
"""

SAMPLE_MOON_RESPONSE = """\
*******************************************************************************
Ephemeris / PORT_LOGIN Tue Mar 11 00:00:00 2026 Jpl/Horizons
Target body name: Moon (301)                  {source: DE441}
Center body name: Earth (399)                 {source: DE441}
*******************************************************************************
$$SOE
2451545.000000000, 2000-Jan-01 12:00:00.0000,  ,123.45678900, -5.67890123,  0.002710000,  0.0001234,
2460310.500000000, 2024-Jan-01 00:00:00.0000,  ,234.56789012,  12.34567890,  0.002650000, -0.0000567,
$$EOE
*******************************************************************************
"""

SAMPLE_MARS_RESPONSE = """\
*******************************************************************************
Ephemeris / API_USER Thu Mar 12 16:02:38 2026 Pasadena, USA      / Horizons
Target body name: Mars (499)                      {source: mar099}
Center body name: Earth (399)                     {source: DE441}
*******************************************************************************
$$SOE
 1987-Oct-12 08:24:50.515, , , 182.642373589,  -0.065640629,  2.59333643461701, -5.7254651,
 2000-Jan-01 12:00:00.000, , , 330.524049117, -13.180707612,  1.84968383439833,  9.3886287,
$$EOE
*******************************************************************************
"""


# ---------------------------------------------------------------------------
# Tests: Horizons response parsing
# ---------------------------------------------------------------------------

class TestParseSourceTag:
    def test_parses_de441(self):
        assert _parse_source_tag(SAMPLE_HORIZONS_RESPONSE) == "DE441"

    def test_fallback_unknown(self):
        assert _parse_source_tag("no ephemeris info here") == "unknown"

    def test_parses_de440(self):
        text = "Ephemeris / DE440\nsome other content"
        assert _parse_source_tag(text) == "DE440"

    def test_parses_combined_planetary_source_tags(self):
        assert _parse_source_tag(SAMPLE_MARS_RESPONSE) == "mar099+DE441"


class TestParseCsvBlock:
    def test_parses_sun_response(self):
        rows = _parse_csv_block(SAMPLE_HORIZONS_RESPONSE)
        assert len(rows) == 3

        assert rows[0]["jd_tt"] == 2451545.0
        assert abs(rows[0]["ra_deg"] - 281.28783929) < 1e-8
        assert abs(rows[0]["dec_deg"] - (-23.01394503)) < 1e-8
        assert abs(rows[0]["delta_au"] - 0.983320046) < 1e-9

    def test_parses_moon_response(self):
        rows = _parse_csv_block(SAMPLE_MOON_RESPONSE)
        assert len(rows) == 2

        assert rows[0]["jd_tt"] == 2451545.0
        assert abs(rows[0]["ra_deg"] - 123.456789) < 1e-6
        assert abs(rows[1]["dec_deg"] - 12.3456789) < 1e-6

    def test_missing_soe_raises(self):
        with pytest.raises(ValueError, match="SOE"):
            _parse_csv_block("no markers here")

    def test_parses_calendar_only_rows_when_epochs_are_known(self):
        epochs = [2447080.850584669, 2451545.0]
        rows = _parse_csv_block(SAMPLE_MARS_RESPONSE, epochs)
        assert len(rows) == 2
        assert rows[0]["jd_tt"] == epochs[0]
        assert abs(rows[0]["ra_deg"] - 182.642373589) < 1e-9
        assert abs(rows[1]["delta_au"] - 1.84968383439833) < 1e-12


class TestReorderCases:
    def test_preserves_original_order(self):
        rows = [
            {"jd_tt": 100.0, "ra_deg": 10.0, "dec_deg": 20.0, "delta_au": 1.0},
            {"jd_tt": 200.0, "ra_deg": 30.0, "dec_deg": 40.0, "delta_au": 2.0},
        ]
        # Request in reverse order
        cases = _reorder_cases(rows, [200.0, 100.0], "solar_position")
        assert len(cases) == 2
        assert cases[0]["jd_tt"] == 200.0
        assert cases[1]["jd_tt"] == 100.0

    def test_handles_duplicates(self):
        rows = [
            {"jd_tt": 100.0, "ra_deg": 10.0, "dec_deg": 20.0, "delta_au": 1.0},
        ]
        cases = _reorder_cases(rows, [100.0, 100.0], "solar_position")
        assert len(cases) == 2
        assert cases[0]["jd_tt"] == 100.0
        assert cases[1]["jd_tt"] == 100.0

    def test_converts_to_radians(self):
        rows = [
            {"jd_tt": 100.0, "ra_deg": 180.0, "dec_deg": 90.0, "delta_au": 1.0},
        ]
        cases = _reorder_cases(rows, [100.0], "solar_position")
        assert abs(cases[0]["ra_rad"] - math.pi) < 1e-10
        assert abs(cases[0]["dec_rad"] - math.pi / 2) < 1e-10

    def test_lunar_includes_dist_km(self):
        rows = [
            {"jd_tt": 100.0, "ra_deg": 10.0, "dec_deg": 20.0, "delta_au": 0.00257},
        ]
        cases = _reorder_cases(rows, [100.0], "lunar_position")
        assert "dist_km" in cases[0]
        assert abs(cases[0]["dist_km"] - 0.00257 * AU_KM) < 1.0

    def test_solar_no_dist_km(self):
        rows = [
            {"jd_tt": 100.0, "ra_deg": 10.0, "dec_deg": 20.0, "delta_au": 1.0},
        ]
        cases = _reorder_cases(rows, [100.0], "solar_position")
        assert "dist_km" not in cases[0]
        assert "dist_au" in cases[0]

    def test_planets_keep_dist_au(self):
        rows = [
            {"jd_tt": 100.0, "ra_deg": 10.0, "dec_deg": 20.0, "delta_au": 1.523},
        ]
        cases = _reorder_cases(rows, [100.0], "mars_position")
        assert "dist_km" not in cases[0]
        assert abs(cases[0]["dist_au"] - 1.523) < 1e-12


# ---------------------------------------------------------------------------
# Tests: Batching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_planet_body_ids_are_mapped(self):
        assert HORIZONS_BODIES["mercury_position"] == "199"
        assert HORIZONS_BODIES["venus_position"] == "299"
        assert HORIZONS_BODIES["mars_position"] == "499"
        assert HORIZONS_BODIES["jupiter_position"] == "599"
        assert HORIZONS_BODIES["saturn_position"] == "699"
        assert HORIZONS_BODIES["uranus_position"] == "799"
        assert HORIZONS_BODIES["neptune_position"] == "899"

    def test_build_query_params(self):
        params = _build_query_params("10", [2451545.0, 2460310.5])
        assert params["COMMAND"] == "'10'"
        assert params["CENTER"] == "'500@399'"
        assert params["TIME_TYPE"] == "TT"
        assert params["TLIST_TYPE"] == "JD"
        assert params["ANG_FORMAT"] == "DEG"
        assert params["CSV_FORMAT"] == "YES"
        assert "2451545" in params["TLIST"]
        assert "2460310" in params["TLIST"]

    def test_batch_size_constant(self):
        assert BATCH_SIZE == 200


# ---------------------------------------------------------------------------
# Tests: Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_key_deterministic(self):
        k1 = _cache_key("solar_position", "10", [2451545.0, 2460310.5])
        k2 = _cache_key("solar_position", "10", [2451545.0, 2460310.5])
        assert k1 == k2

    def test_cache_key_differs_for_different_epochs(self):
        k1 = _cache_key("solar_position", "10", [2451545.0])
        k2 = _cache_key("solar_position", "10", [2460310.5])
        assert k1 != k2

    def test_cache_key_differs_for_different_experiments(self):
        k1 = _cache_key("solar_position", "10", [2451545.0])
        k2 = _cache_key("lunar_position", "301", [2451545.0])
        assert k1 != k2

    def test_cache_round_trip(self, tmp_path):
        """Cache save + load returns identical data."""
        with patch("horizons_client.CACHE_DIR", tmp_path):
            data = {"test": "value", "rows": [{"jd_tt": 1.0}]}
            key = "test_key_abc"

            _save_cache(key, data)
            loaded = _load_cache(key)
            assert loaded == data

    def test_cache_miss_returns_none(self, tmp_path):
        with patch("horizons_client.CACHE_DIR", tmp_path):
            assert _load_cache("nonexistent_key") is None


# ---------------------------------------------------------------------------
# Tests: fetch_horizons_reference with mocked network
# ---------------------------------------------------------------------------

class TestFetchHorizonsReference:
    def test_cache_hit_skips_network(self, tmp_path):
        """When cache is warm, no network call is made."""
        cached_data = {
            "experiment": "solar_position",
            "body_id": "10",
            "source_tag": "DE441",
            "query_params": {},
            "epochs_count": 1,
            "rows": [
                {"jd_tt": 2451545.0, "ra_deg": 280.0, "dec_deg": -23.0, "delta_au": 0.98},
            ],
            "fetch_timestamp": "2026-01-01T00:00:00Z",
        }

        with patch("horizons_client.CACHE_DIR", tmp_path), \
             patch("horizons_client._fetch_batch") as mock_fetch:

            # Pre-populate cache
            key = _cache_key("solar_position", "10", [2451545.0])
            _save_cache(key, cached_data)

            result = fetch_horizons_reference("solar_position", [2451545.0], use_cache=True)

            mock_fetch.assert_not_called()
            assert result["from_cache"] is True
            assert result["source_tag"] == "DE441"
            assert len(result["cases"]) == 1
            assert abs(result["cases"][0]["ra_rad"] - 280.0 * math.pi / 180.0) < 1e-10

    def test_cache_miss_fetches_and_caches(self, tmp_path):
        """Cache miss triggers fetch & persists result."""
        mock_rows = [
            {"jd_tt": 2451545.0, "ra_deg": 280.0, "dec_deg": -23.0, "delta_au": 0.98},
        ]

        with patch("horizons_client.CACHE_DIR", tmp_path), \
             patch("horizons_client._fetch_batch", return_value=(mock_rows, "DE441")):

            result = fetch_horizons_reference("solar_position", [2451545.0], use_cache=True)

            assert result["from_cache"] is False
            assert result["source_tag"] == "DE441"
            assert len(result["cases"]) == 1

            # Verify cache was written
            key = result["cache_key"]
            cached = _load_cache(key)
            assert cached is not None
            assert cached["source_tag"] == "DE441"

    def test_fetch_failure_raises_runtime_error(self, tmp_path):
        """Network failure without cache raises RuntimeError."""
        with patch("horizons_client.CACHE_DIR", tmp_path), \
             patch("horizons_client._fetch_batch",
                   side_effect=RuntimeError("Connection refused")):

            with pytest.raises(RuntimeError, match="Connection refused"):
                fetch_horizons_reference("solar_position", [2451545.0], use_cache=True)

    def test_unknown_experiment_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown_experiment"):
            fetch_horizons_reference("unknown_experiment", [2451545.0])

    def test_batching_for_large_epoch_sets(self, tmp_path):
        """Epochs exceeding BATCH_SIZE are split across multiple calls."""
        epochs = [2451545.0 + i for i in range(450)]
        mock_rows = [
            {"jd_tt": e, "ra_deg": 280.0, "dec_deg": -23.0, "delta_au": 0.98}
            for e in epochs
        ]

        def batch_fetcher(body_id, batch):
            return (
                [r for r in mock_rows if r["jd_tt"] in batch],
                "DE441",
            )

        with patch("horizons_client.CACHE_DIR", tmp_path), \
             patch("horizons_client._fetch_batch", side_effect=batch_fetcher) as mock_fetch:

            result = fetch_horizons_reference("solar_position", epochs, use_cache=True)

            # 450 unique epochs / 200 batch size = 3 batches
            assert mock_fetch.call_count == 3
            assert len(result["cases"]) == 450


# ---------------------------------------------------------------------------
# Tests: Orchestrator external reference path (integration with mocks)
# ---------------------------------------------------------------------------

class TestExternalReferenceOrchestrator:
    """Test the orchestrator's _run_external_reference_experiment path."""

    @pytest.fixture
    def mock_horizons_solar(self):
        """Mock Horizons to return synthetic Sun positions."""
        import numpy as np

        def mock_fetch(experiment, epochs, use_cache=True):
            cases = []
            for e in epochs:
                # Synthetic reference: RA ~ 280 deg, Dec ~ -23 deg
                cases.append({
                    "jd_tt": float(e),
                    "ra_rad": 280.0 * math.pi / 180.0,
                    "dec_rad": -23.0 * math.pi / 180.0,
                    "dist_au": 0.98,
                })
            return {
                "cases": cases,
                "source_tag": "DE441",
                "cache_key": "test_cache_key",
                "from_cache": True,
                "query_params": {},
                "fetch_timestamp": "2026-01-01T00:00:00Z",
            }
        return mock_fetch

    @pytest.fixture
    def mock_adapter_result(self):
        """Return a minimal adapter result that won't cause errors."""
        def _make(n):
            return {
                "cases": [
                    {"jd_tt": 2451545.0 + i, "ra_rad": 4.89, "dec_rad": -0.40, "dist_au": 0.98}
                    for i in range(n)
                ],
                "per_op_ns": 1000.0,
                "total_ns": 1000000.0,
                "count": n,
            }
        return _make

    def test_solar_produces_jpl_horizons_reference(self, mock_horizons_solar, mock_adapter_result):
        """solar_position results should have reference_library == 'jpl_horizons'."""
        import orchestrator as orch
        with patch.object(orch, "fetch_horizons_reference", mock_horizons_solar), \
             patch.object(orch, "run_adapter", return_value=mock_adapter_result(10)), \
             patch.object(orch, "ensure_rust_adapters_built"):

            results = orch.run_experiment_solar_position(n=10, seed=42, run_perf=False)

            assert len(results) > 0
            for r in results:
                assert r["reference_library"] == "jpl_horizons"
                assert r["reference_performance"] == {}

    def test_erfa_appears_as_candidate(self, mock_horizons_solar, mock_adapter_result):
        """ERFA should appear as a candidate library, not just a reference."""
        import orchestrator as orch
        with patch.object(orch, "fetch_horizons_reference", mock_horizons_solar), \
             patch.object(orch, "run_adapter", return_value=mock_adapter_result(10)), \
             patch.object(orch, "ensure_rust_adapters_built"):

            results = orch.run_experiment_solar_position(n=10, seed=42, run_perf=False)

            candidate_libs = [r["candidate_library"] for r in results]
            assert "erfa" in candidate_libs

    def test_horizons_metadata_in_inputs(self, mock_horizons_solar, mock_adapter_result):
        """Results should contain Horizons metadata in inputs."""
        import orchestrator as orch
        with patch.object(orch, "fetch_horizons_reference", mock_horizons_solar), \
             patch.object(orch, "run_adapter", return_value=mock_adapter_result(10)), \
             patch.object(orch, "ensure_rust_adapters_built"):

            results = orch.run_experiment_solar_position(n=10, seed=42, run_perf=False)

            inputs = results[0]["inputs"]
            assert inputs["reference_source"] == "jpl_horizons"
            assert inputs["horizons_ephemeris"] == "DE441"
            assert inputs["horizons_mode"] == "astrometric geocentric RA/Dec"
            assert inputs["horizons_frame"] == "ICRF"
            assert inputs["horizons_center"] == "Earth geocenter (500@399)"
            assert inputs["horizons_time_scale"] == "TT"

    def test_horizons_failure_aborts_experiment_cleanly(self):
        """If Horizons fetch fails, the experiment returns empty results."""
        def failing_fetch(experiment, epochs, use_cache=True):
            raise RuntimeError("Network unreachable")

        import orchestrator as orch
        with patch.object(orch, "fetch_horizons_reference", failing_fetch), \
             patch.object(orch, "run_adapter"), \
             patch.object(orch, "ensure_rust_adapters_built"):

            results = orch.run_experiment_solar_position(n=10, seed=42, run_perf=False)

            assert results == []

    def test_alignment_shows_horizons_source(self, mock_horizons_solar, mock_adapter_result):
        """Alignment checklist should contain horizons_source metadata."""
        import orchestrator as orch
        with patch.object(orch, "fetch_horizons_reference", mock_horizons_solar), \
             patch.object(orch, "run_adapter", return_value=mock_adapter_result(10)), \
             patch.object(orch, "ensure_rust_adapters_built"):

            results = orch.run_experiment_solar_position(n=10, seed=42, run_perf=False)

            alignment = results[0]["alignment"]
            assert alignment["horizons_source"] == "DE441"
            assert "horizons_cache_key" in alignment

    def test_planet_produces_jpl_horizons_reference(self, mock_horizons_solar, mock_adapter_result):
        """Planet experiments should use Horizons as the external reference."""
        import orchestrator as orch
        with patch.object(orch, "fetch_horizons_reference", mock_horizons_solar), \
             patch.object(orch, "run_adapter", return_value=mock_adapter_result(10)), \
             patch.object(orch, "ensure_rust_adapters_built"):

            results = orch.run_experiment_planet_position(
                "mars_position", n=10, seed=42, run_perf=False
            )

            assert len(results) > 0
            for r in results:
                assert r["experiment"] == "mars_position"
                assert r["reference_library"] == "jpl_horizons"
                assert r["reference_performance"] == {}
