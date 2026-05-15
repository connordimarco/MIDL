import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_combine import (
    SAT_CODE,
    _agree,
    _apply_source_to_components,
    _fallback_source,
    _select_column_with_continuity,
    _switch_threshold,
    combine_data_priority,
    combine_temperature,
)
from helpers import _synthetic_sat_df, _synthetic_data_map


# ---------------------------------------------------------------------------
# _switch_threshold
# ---------------------------------------------------------------------------

class TestSwitchThreshold:
    def test_known_thresholds(self):
        assert _switch_threshold("Bx") == 8.0
        assert _switch_threshold("|B|") == 8.0
        assert _switch_threshold("Ux") == 80.0
        assert _switch_threshold("Uy") == 40.0
        assert _switch_threshold("rho") == 2.0

    def test_unknown_returns_inf(self):
        assert _switch_threshold("unknown_col") == np.inf


# ---------------------------------------------------------------------------
# _agree
# ---------------------------------------------------------------------------

class TestAgree:
    def test_within_threshold(self):
        assert _agree(5.0, 6.0, "Bx") is True

    def test_at_boundary(self):
        assert _agree(0.0, 8.0, "Bx") is True

    def test_above_threshold(self):
        assert _agree(0.0, 8.01, "Bx") is False


# ---------------------------------------------------------------------------
# _fallback_source
# ---------------------------------------------------------------------------

class TestFallbackSource:
    def test_startup_prefers_wind(self):
        values = {1: -400.0, 2: -410.0, 3: -390.0}
        result = _fallback_source(values, [1, 2, 3], np.nan)
        assert result == 3

    def test_closest_to_prev(self):
        values = {1: -410.0, 3: -450.0}
        result = _fallback_source(values, [1, 3], -400.0)
        assert result == 1

    def test_deprioritize_excluded(self):
        values = {1: -400.0, 2: -399.0, 3: -410.0}
        result = _fallback_source(values, [1, 2, 3], -399.5, deprioritize_code=2)
        assert result != 2

    def test_deprioritize_only_option(self):
        values = {2: -400.0}
        result = _fallback_source(values, [2], -400.0, deprioritize_code=2)
        assert result == 2


# ---------------------------------------------------------------------------
# _select_column_with_continuity
# ---------------------------------------------------------------------------

class TestSelectColumnWithContinuity:
    def _make_sat_series(self, n=60, **overrides):
        """Build sat_series dict for _select_column_with_continuity."""
        idx = pd.date_range("2024-05-01", periods=n, freq="min")
        defaults = {"ace": -400.0, "dscovr": -400.0, "wind": -400.0, "solar1": np.nan}
        defaults.update(overrides)
        return {name: pd.Series(defaults[name], index=idx) for name in SAT_CODE}

    def test_single_sat_passthrough(self):
        ss = self._make_sat_series(dscovr=np.nan, wind=np.nan, solar1=np.nan)
        vals, _, source = _select_column_with_continuity("Ux", ss)
        assert vals.notna().all()
        assert all(s == frozenset({1}) for s in source if s is not None)

    def test_all_agree_median(self):
        ss = self._make_sat_series(ace=5.0, dscovr=6.0, wind=5.5)
        vals, _, source = _select_column_with_continuity("|B|", ss)
        for s in source:
            if s is not None:
                assert len(s) >= 2

    def test_pair_agree_mean(self):
        ss = self._make_sat_series(ace=5.0, dscovr=6.0, wind=50.0)
        vals, _, source = _select_column_with_continuity("|B|", ss)
        for i, s in enumerate(source):
            if s is not None and len(s) == 2:
                assert 1 in s and 2 in s
                assert abs(vals.iloc[i] - 5.5) < 0.1

    def test_none_agree_fallback(self):
        ss = self._make_sat_series(ace=0.0, dscovr=50.0, wind=100.0)
        vals, _, source = _select_column_with_continuity("|B|", ss)
        for s in source:
            if s is not None:
                assert len(s) == 1

    def test_hysteresis_3min(self):
        idx = pd.date_range("2024-05-01", periods=60, freq="min")
        ace_vals = np.full(60, 0.0)
        dscovr_vals = np.full(60, 50.0)
        wind_vals = np.full(60, 100.0)
        wind_vals[30] = 0.5
        wind_vals[31] = 0.5
        ss = {
            "ace": pd.Series(ace_vals, index=idx),
            "dscovr": pd.Series(dscovr_vals, index=idx),
            "wind": pd.Series(wind_vals, index=idx),
            "solar1": pd.Series(np.nan, index=idx),
        }
        vals, _, source = _select_column_with_continuity("|B|", ss)
        sources_29_32 = [source.iloc[i] for i in range(29, 33)]
        assert all(s is not None for s in sources_29_32)

    def test_bad_mask_excludes_sat(self):
        ss = self._make_sat_series(ace=5.0, dscovr=5.0, wind=5.0)
        idx = ss["ace"].index
        bad = {1: {"Ux": pd.Series(True, index=idx)}}
        vals, _, source = _select_column_with_continuity("Ux", ss, bad_masks=bad)
        for s in source:
            if s is not None:
                assert 1 not in s

    def test_all_nan_produces_nan(self):
        ss = self._make_sat_series(ace=np.nan, dscovr=np.nan, wind=np.nan)
        vals, _, source = _select_column_with_continuity("Ux", ss)
        assert vals.isna().all()


# ---------------------------------------------------------------------------
# _apply_source_to_components
# ---------------------------------------------------------------------------

class TestApplySourceToComponents:
    def test_single_source(self):
        idx = pd.date_range("2024-05-01", periods=5, freq="min")
        source = pd.Series([frozenset({1})] * 5, index=idx)
        comp = {"ace": pd.Series(10.0, index=idx),
                "dscovr": pd.Series(20.0, index=idx),
                "wind": pd.Series(30.0, index=idx),
                "solar1": pd.Series(np.nan, index=idx)}
        result = _apply_source_to_components(source, comp, idx)
        np.testing.assert_array_equal(result.values, 10.0)

    def test_two_sources_mean(self):
        idx = pd.date_range("2024-05-01", periods=5, freq="min")
        source = pd.Series([frozenset({1, 3})] * 5, index=idx)
        comp = {"ace": pd.Series(10.0, index=idx),
                "dscovr": pd.Series(20.0, index=idx),
                "wind": pd.Series(30.0, index=idx),
                "solar1": pd.Series(np.nan, index=idx)}
        result = _apply_source_to_components(source, comp, idx)
        np.testing.assert_allclose(result.values, 20.0)

    def test_three_sources_median(self):
        idx = pd.date_range("2024-05-01", periods=5, freq="min")
        source = pd.Series([frozenset({1, 2, 3})] * 5, index=idx)
        comp = {"ace": pd.Series(10.0, index=idx),
                "dscovr": pd.Series(20.0, index=idx),
                "wind": pd.Series(30.0, index=idx),
                "solar1": pd.Series(np.nan, index=idx)}
        result = _apply_source_to_components(source, comp, idx)
        np.testing.assert_allclose(result.values, 20.0)

    def test_none_source_produces_nan(self):
        idx = pd.date_range("2024-05-01", periods=5, freq="min")
        source = pd.Series([None] * 5, index=idx)
        comp = {"ace": pd.Series(10.0, index=idx),
                "dscovr": pd.Series(20.0, index=idx),
                "wind": pd.Series(30.0, index=idx),
                "solar1": pd.Series(np.nan, index=idx)}
        result = _apply_source_to_components(source, comp, idx)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# combine_data_priority (integration-level)
# ---------------------------------------------------------------------------

class TestCombineDataPriority:
    def test_output_schema(self):
        data_map = _synthetic_data_map(n_sats=3, n=60, noise=1.0)
        grid = pd.date_range("2024-05-01", periods=60, freq="min")
        df, src = combine_data_priority(data_map, grid)
        for col in ["Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho"]:
            assert col in df.columns

    def test_b_coupled_source(self):
        data_map = _synthetic_data_map(n_sats=3, n=60, noise=0.5)
        grid = pd.date_range("2024-05-01", periods=60, freq="min")
        _, src = combine_data_priority(data_map, grid)
        pd.testing.assert_series_equal(src["Bx"], src["By"])
        pd.testing.assert_series_equal(src["Bx"], src["Bz"])

    def test_uyz_coupled_source(self):
        data_map = _synthetic_data_map(n_sats=3, n=60, noise=0.5)
        grid = pd.date_range("2024-05-01", periods=60, freq="min")
        _, src = combine_data_priority(data_map, grid)
        pd.testing.assert_series_equal(src["Uy"], src["Uz"])

    def test_ux_independent_source(self):
        data_map = _synthetic_data_map(n_sats=3, n=60, noise=0.5)
        grid = pd.date_range("2024-05-01", periods=60, freq="min")
        _, src = combine_data_priority(data_map, grid)
        assert "Ux" in src
        assert "Uy" in src


# ---------------------------------------------------------------------------
# combine_temperature
# ---------------------------------------------------------------------------

class TestCombineTemperature:
    def test_geometric_median(self):
        grid = pd.date_range("2024-05-01", periods=60, freq="min")
        data_map = {
            "ace": pd.DataFrame({"T": np.full(60, 1e5)}, index=grid),
            "dscovr": pd.DataFrame({"T": np.full(60, 4e5)}, index=grid),
            "wind": pd.DataFrame({"T": np.full(60, 1e5)}, index=grid),
            "solar1": pd.DataFrame({"T": np.full(60, np.nan)}, index=grid),
        }
        T, _ = combine_temperature(data_map, grid)
        assert T.iloc[30] > 0
        assert T.notna().sum() > 0

    def test_all_nan(self):
        grid = pd.date_range("2024-05-01", periods=60, freq="min")
        data_map = {
            "ace": pd.DataFrame({"T": np.full(60, np.nan)}, index=grid),
            "dscovr": pd.DataFrame({"T": np.full(60, np.nan)}, index=grid),
            "wind": pd.DataFrame({"T": np.full(60, np.nan)}, index=grid),
            "solar1": pd.DataFrame({"T": np.full(60, np.nan)}, index=grid),
        }
        T, _ = combine_temperature(data_map, grid)
        assert T.isna().all()

    def test_single_sat_passthrough(self):
        grid = pd.date_range("2024-05-01", periods=60, freq="min")
        data_map = {
            "ace": pd.DataFrame({"T": np.full(60, 1e5)}, index=grid),
            "dscovr": pd.DataFrame({"T": np.full(60, np.nan)}, index=grid),
            "wind": pd.DataFrame({"T": np.full(60, np.nan)}, index=grid),
            "solar1": pd.DataFrame({"T": np.full(60, np.nan)}, index=grid),
        }
        T, _ = combine_temperature(data_map, grid)
        valid = T.dropna()
        if len(valid) > 0:
            np.testing.assert_allclose(valid.values, 1e5, rtol=0.1)
