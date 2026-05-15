import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_filters import (
    INTERP_LIMITS,
    despike,
    interpolate_with_limits,
    median_filter_3,
    smooth_transitions,
    _jump_magnitude,
)
from helpers import _synthetic_sat_df, _synthetic_sat_df_with_gap, _synthetic_sat_df_with_spike


# ---------------------------------------------------------------------------
# median_filter_3
# ---------------------------------------------------------------------------

class TestMedianFilter3:
    def test_constant_unchanged(self):
        arr = np.full(20, 5.0)
        result = median_filter_3(arr)
        np.testing.assert_array_equal(result, arr)

    def test_removes_single_spike(self):
        arr = np.array([5.0, 5.0, 999.0, 5.0, 5.0])
        result = median_filter_3(arr)
        assert result[2] == 5.0

    def test_preserves_endpoints(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = median_filter_3(arr)
        assert np.isfinite(result[0])
        assert np.isfinite(result[-1])

    def test_short_input_passthrough(self):
        arr = np.array([1.0, 2.0])
        result = median_filter_3(arr)
        np.testing.assert_array_equal(result, arr)

    def test_length_1(self):
        arr = np.array([42.0])
        result = median_filter_3(arr)
        np.testing.assert_array_equal(result, arr)

    def test_nan_not_poisoning(self):
        arr = np.array([5.0, np.nan, 5.0, 5.0])
        result = median_filter_3(arr)
        assert result[2] == 5.0
        assert result[3] == 5.0

    def test_all_nan(self):
        arr = np.full(10, np.nan)
        result = median_filter_3(arr)
        assert np.all(np.isnan(result))

    def test_dtype_float64(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = median_filter_3(arr)
        assert result.dtype == np.float64

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            median_filter_3(np.ones((3, 3)))


# ---------------------------------------------------------------------------
# despike
# ---------------------------------------------------------------------------

class TestDespike:
    def test_removes_spike(self):
        df = _synthetic_sat_df_with_spike(n=20, spike_index=10, spike_col="Bx", spike_value=999.0)
        result = despike(df)
        assert abs(result["Bx"].iloc[10] - 5.0) < 1.0

    def test_preserves_clean(self):
        df = _synthetic_sat_df(n=20)
        result = despike(df)
        np.testing.assert_array_almost_equal(result["Bx"].values, df["Bx"].values)

    def test_all_seven_cols(self):
        df = _synthetic_sat_df(n=20)
        for col in ["Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho"]:
            df.iloc[10, df.columns.get_loc(col)] = 9999.0
        result = despike(df)
        for col in ["Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho"]:
            assert abs(result[col].iloc[10]) < 9000, f"{col} spike not removed"

    def test_T_not_filtered(self):
        df = _synthetic_sat_df(n=20, T=1e5)
        df.iloc[10, df.columns.get_loc("T")] = 1e9
        result = despike(df)
        assert result["T"].iloc[10] == 1e9

    def test_missing_col_ok(self):
        df = _synthetic_sat_df(n=20).drop(columns=["Ux"])
        result = despike(df)
        assert "Ux" not in result.columns

    def test_does_not_mutate_input(self):
        df = _synthetic_sat_df_with_spike(n=20, spike_index=10, spike_value=999.0)
        original_val = df["Bx"].iloc[10]
        despike(df)
        assert df["Bx"].iloc[10] == original_val


# ---------------------------------------------------------------------------
# interpolate_with_limits
# ---------------------------------------------------------------------------

class TestInterpolateWithLimits:
    def test_fills_small_bx_gap(self):
        df = _synthetic_sat_df_with_gap(n=60, gap_start=20, gap_end=23, gap_cols=["Bx"])
        result = interpolate_with_limits(df)
        assert result["Bx"].iloc[20:23].notna().all()

    def test_rejects_large_bx_gap(self):
        df = _synthetic_sat_df_with_gap(n=60, gap_start=20, gap_end=35, gap_cols=["Bx"])
        result = interpolate_with_limits(df)
        assert result["Bx"].iloc[25:30].isna().any()

    def test_fills_60min_plasma_gap(self):
        df = _synthetic_sat_df_with_gap(n=120, gap_start=30, gap_end=60, gap_cols=["Ux"])
        result = interpolate_with_limits(df)
        assert result["Ux"].iloc[30:60].notna().all()

    def test_rejects_large_plasma_gap(self):
        df = _synthetic_sat_df_with_gap(n=200, gap_start=30, gap_end=100, gap_cols=["Ux"])
        result = interpolate_with_limits(df)
        assert result["Ux"].iloc[90:100].isna().any()

    def test_leading_nan_not_filled(self):
        df = _synthetic_sat_df(n=60)
        df.iloc[:5, df.columns.get_loc("Bx")] = np.nan
        result = interpolate_with_limits(df)
        assert result["Bx"].iloc[0:5].isna().all()

    def test_trailing_nan_not_filled(self):
        df = _synthetic_sat_df(n=60)
        df.iloc[-5:, df.columns.get_loc("Bx")] = np.nan
        result = interpolate_with_limits(df)
        assert result["Bx"].iloc[-5:].isna().all()

    def test_custom_limits(self):
        df = _synthetic_sat_df_with_gap(n=60, gap_start=20, gap_end=30, gap_cols=["Bx"])
        result = interpolate_with_limits(df, limits={"Bx": 20})
        assert result["Bx"].iloc[20:30].notna().all()

    def test_missing_col_in_limits(self):
        df = _synthetic_sat_df(n=20).drop(columns=["Bx"])
        result = interpolate_with_limits(df)
        assert "Bx" not in result.columns

    def test_does_not_mutate_input(self):
        df = _synthetic_sat_df_with_gap(n=60, gap_start=20, gap_end=23, gap_cols=["Bx"])
        assert df["Bx"].iloc[21] != df["Bx"].iloc[21]  # NaN != NaN
        interpolate_with_limits(df)
        assert pd.isna(df["Bx"].iloc[21])


# ---------------------------------------------------------------------------
# _jump_magnitude
# ---------------------------------------------------------------------------

class TestJumpMagnitude:
    def test_pct_symmetric(self):
        assert _jump_magnitude(100, 200, "pct") == _jump_magnitude(200, 100, "pct")
        assert abs(_jump_magnitude(100, 200, "pct") - 100.0) < 1e-10

    def test_pct_zero_safe(self):
        assert _jump_magnitude(0, 100, "pct") == 0.0

    def test_abs_mode(self):
        assert _jump_magnitude(10, 30, "abs") == 20.0

    def test_pct_same_value(self):
        assert _jump_magnitude(5, 5, "pct") == 0.0

    def test_negative_values(self):
        assert _jump_magnitude(-100, -200, "pct") == _jump_magnitude(100, 200, "pct")


# ---------------------------------------------------------------------------
# smooth_transitions
# ---------------------------------------------------------------------------

class TestSmoothTransitions:
    def _make_step_df(self, n=60, step_index=30, col="Ux", v1=-400.0, v2=-250.0):
        """Create a df with a sharp step in one column."""
        df = _synthetic_sat_df(n=n, ux=v1)
        df.iloc[step_index:, df.columns.get_loc(col)] = v2
        return df

    def _make_source_changed(self, df, step_index=30, col="Ux"):
        changed = pd.Series(False, index=df.index)
        changed.iloc[step_index] = True
        return {col: changed}

    def test_no_smooth_below_cmax(self):
        df = self._make_step_df(v1=-400.0, v2=-395.0)
        sc = self._make_source_changed(df)
        result = smooth_transitions(df, source_changed=sc)
        np.testing.assert_array_equal(result["Ux"].values, df["Ux"].values)

    def test_smooth_above_cmax(self):
        df = self._make_step_df(v1=-400.0, v2=-200.0)
        sc = self._make_source_changed(df)
        result = smooth_transitions(df, source_changed=sc)
        assert result["Ux"].iloc[30] != df["Ux"].iloc[30]

    def test_window_scales_with_jump(self):
        df_small = self._make_step_df(v1=-400.0, v2=-270.0)
        sc_small = self._make_source_changed(df_small)
        result_small = smooth_transitions(df_small, source_changed=sc_small)

        df_large = self._make_step_df(v1=-400.0, v2=-100.0)
        sc_large = self._make_source_changed(df_large)
        result_large = smooth_transitions(df_large, source_changed=sc_large)

        diff_small = (result_small["Ux"] - df_small["Ux"]).abs()
        diff_large = (result_large["Ux"] - df_large["Ux"]).abs()
        assert (diff_large > 0).sum() >= (diff_small > 0).sum()

    def test_window_capped_at_wmax(self):
        df = self._make_step_df(v1=-400.0, v2=-10.0)
        sc = self._make_source_changed(df)
        result = smooth_transitions(df, source_changed=sc, wmax=5)
        diff = (result["Ux"] - df["Ux"]).abs()
        assert (diff > 0).sum() <= 10

    def test_source_changed_none_all_candidates(self):
        df = self._make_step_df()
        result = smooth_transitions(df, source_changed=None)
        assert result["Ux"].iloc[30] != df["Ux"].iloc[30]

    def test_unchanged_minutes_skipped(self):
        df = self._make_step_df(v1=-400.0, v2=-200.0)
        sc = {"Ux": pd.Series(False, index=df.index)}
        result = smooth_transitions(df, source_changed=sc)
        np.testing.assert_array_equal(result["Ux"].values, df["Ux"].values)

    def test_b_fields_untouched(self):
        df = _synthetic_sat_df(n=60, bx=5.0)
        df.iloc[30:, df.columns.get_loc("Bx")] = 50.0
        result = smooth_transitions(df, source_changed=None)
        np.testing.assert_array_equal(result["Bx"].values, df["Bx"].values)

    def test_does_not_mutate_input(self):
        df = self._make_step_df()
        original = df["Ux"].copy()
        smooth_transitions(df, source_changed=None)
        pd.testing.assert_series_equal(df["Ux"], original)
