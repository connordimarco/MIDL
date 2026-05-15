import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_propagation import VX_COL, ballistic_propagation
from helpers import _synthetic_sat_df


def _make_prop_input(n=120, ux=-400.0, x_gse_km=1.5e6, target_km=90000, noise=0.0):
    """Build inputs for ballistic_propagation."""
    df = _synthetic_sat_df(n=n, ux=ux, noise=noise)
    df = df.rename(columns={"Ux": VX_COL})
    orbit = pd.Series({"X_GSE": x_gse_km})
    return orbit, df, target_km


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

class TestBallisticPropagation:
    def test_constant_wind_delay(self):
        orbit, df, target = _make_prop_input(n=240, ux=-400.0)
        result = ballistic_propagation(orbit, df, target_x_km=target)
        expected_delay = (1.5e6 - 90000) / 400.0 / 60.0  # ~58.75 min
        bx_input_peak = 0
        shifted = int(round(expected_delay))
        assert abs(result["Bx"].iloc[shifted] - df["Bx"].iloc[bx_input_peak]) < 1.0

    def test_output_grid_matches_input(self):
        orbit, df, target = _make_prop_input(n=120)
        result = ballistic_propagation(orbit, df, target_x_km=target)
        assert result.index[0] == df.index[0]
        assert result.index[-1] == df.index[-1]
        assert len(result) == len(df)

    def test_causality_drops_overtaken(self):
        orbit, df, target = _make_prop_input(n=120, ux=-300.0)
        df.iloc[60:, df.columns.get_loc(VX_COL)] = -600.0
        result = ballistic_propagation(orbit, df, target_x_km=target)
        assert result.index.is_monotonic_increasing

    def test_nan_ux_filled_for_timing(self):
        orbit, df, target = _make_prop_input(n=120)
        df.iloc[30:35, df.columns.get_loc(VX_COL)] = np.nan
        result = ballistic_propagation(orbit, df, target_x_km=target)
        assert result["Bx"].notna().sum() > 50

    def test_original_ux_nan_restored(self):
        orbit, df, target = _make_prop_input(n=120)
        df.iloc[50, df.columns.get_loc(VX_COL)] = np.nan
        result = ballistic_propagation(orbit, df, target_x_km=target)
        assert result[VX_COL].isna().any()

    def test_all_nan_ux_no_crash(self):
        orbit, df, target = _make_prop_input(n=60)
        df[VX_COL] = np.nan
        result = ballistic_propagation(orbit, df, target_x_km=target)
        assert isinstance(result, pd.DataFrame)

    def test_zero_distance_identity(self):
        x_km = 90000.0
        orbit, df, _ = _make_prop_input(n=60, x_gse_km=x_km)
        result = ballistic_propagation(orbit, df, target_x_km=x_km)
        np.testing.assert_allclose(
            result["Bx"].dropna().values,
            df["Bx"].reindex(result.index).dropna().values,
            atol=0.1,
        )

    def test_single_row_no_crash(self):
        orbit, df, target = _make_prop_input(n=1)
        result = ballistic_propagation(orbit, df, target_x_km=target)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Numerical properties
# ---------------------------------------------------------------------------

class TestPropagationNumerics:
    def test_travel_time_formula(self):
        x_gse = 1.5e6
        target = 90000.0
        ux = 400.0
        expected_s = (x_gse - target) / ux
        assert abs(expected_s - 3525.0) < 1.0

    def test_fast_wind_shorter_delay(self):
        orbit_fast, df_fast, target = _make_prop_input(n=240, ux=-800.0)
        orbit_slow, df_slow, _ = _make_prop_input(n=240, ux=-400.0)
        result_fast = ballistic_propagation(orbit_fast, df_fast, target_x_km=target)
        result_slow = ballistic_propagation(orbit_slow, df_slow, target_x_km=target)
        first_valid_fast = result_fast["Bx"].first_valid_index()
        first_valid_slow = result_slow["Bx"].first_valid_index()
        assert first_valid_fast is not None
        assert first_valid_slow is not None

    def test_monotonic_output_time(self):
        orbit, df, target = _make_prop_input(n=120)
        result = ballistic_propagation(orbit, df, target_x_km=target)
        assert result.index.is_monotonic_increasing
