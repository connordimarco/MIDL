import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_filters import despike, interpolate_with_limits, smooth_transitions
from midl_pipeline.l1_combine import combine_data_priority, combine_temperature, SAT_CODE
from midl_pipeline.l1_midl import MIDLResult, _compute_source_changed
from helpers import (
    _synthetic_sat_df,
    _synthetic_sat_df_with_gap,
    _synthetic_sat_df_with_spike,
    _synthetic_data_map,
)


@pytest.mark.integration
class TestDespikeInterpolateChain:
    def test_spike_then_interpolate(self):
        df = _synthetic_sat_df(n=60, bx=5.0)
        df.iloc[10, df.columns.get_loc("Bx")] = 999.0
        df.iloc[30:33, df.columns.get_loc("Bx")] = np.nan

        result = despike(df)
        assert abs(result["Bx"].iloc[10] - 5.0) < 1.0

        result = interpolate_with_limits(result)
        assert result["Bx"].iloc[30:33].notna().all()

    def test_nan_spike_recovery(self):
        df = _synthetic_sat_df(n=60, bx=5.0)
        df.iloc[19, df.columns.get_loc("Bx")] = np.nan
        df.iloc[20, df.columns.get_loc("Bx")] = 999.0
        df.iloc[21, df.columns.get_loc("Bx")] = np.nan

        result = despike(df)
        result = interpolate_with_limits(result)
        assert result["Bx"].iloc[18:23].notna().sum() >= 3


@pytest.mark.integration
class TestCombineAndSmooth:
    def test_source_change_smoothed(self):
        grid = pd.date_range("2024-05-01", periods=120, freq="min")
        data_map = {
            "ace": pd.DataFrame({
                "Bx": 5.0, "By": 0.0, "Bz": 0.0,
                "Ux": -400.0, "Uy": 0.0, "Uz": 0.0, "rho": 5.0, "T": 1e5,
            }, index=grid),
            "dscovr": pd.DataFrame({
                "Bx": 5.0, "By": 0.0, "Bz": 0.0,
                "Ux": -400.0, "Uy": 0.0, "Uz": 0.0, "rho": 5.0, "T": 1e5,
            }, index=grid),
            "wind": pd.DataFrame({
                "Bx": 5.0, "By": 0.0, "Bz": 0.0,
                "Ux": -400.0, "Uy": 0.0, "Uz": 0.0, "rho": 5.0, "T": 1e5,
            }, index=grid),
        }
        data_map["wind"].iloc[60:, data_map["wind"].columns.get_loc("Ux")] = -200.0

        df_combined, source_map = combine_data_priority(data_map, grid)
        source_changed = _compute_source_changed(source_map)
        smoothed = smooth_transitions(df_combined, source_changed=source_changed)

        assert isinstance(smoothed, pd.DataFrame)
        assert len(smoothed) == 120


@pytest.mark.integration
class TestFullSyntheticPipeline:
    def test_three_agreeing_sats(self):
        n = 120
        data_map = _synthetic_data_map(n_sats=3, n=n, noise=0.5)
        grid = pd.date_range("2024-05-01", periods=n, freq="min")

        for sat in data_map:
            data_map[sat] = despike(data_map[sat])
            data_map[sat] = interpolate_with_limits(data_map[sat])

        df_combined, source_map = combine_data_priority(data_map, grid)
        df_combined["T"], _ = combine_temperature(data_map, grid)

        source_changed = _compute_source_changed(source_map)
        smoothed = smooth_transitions(df_combined, source_changed=source_changed)

        assert smoothed["Bx"].notna().sum() > n * 0.9
        assert abs(smoothed["Bx"].median() - 5.0) < 2.0

    def test_single_sat_passthrough(self):
        n = 120
        data_map = {"ace": _synthetic_sat_df(n=n, noise=0.5)}
        grid = pd.date_range("2024-05-01", periods=n, freq="min")

        data_map["ace"] = despike(data_map["ace"])
        data_map["ace"] = interpolate_with_limits(data_map["ace"])

        df_combined, source_map = combine_data_priority(data_map, grid)
        assert df_combined["Bx"].notna().sum() > n * 0.9

    def test_nan_propagation(self):
        n = 120
        df = _synthetic_sat_df(n=n)
        df.iloc[40:80] = np.nan
        data_map = {"ace": df}
        grid = pd.date_range("2024-05-01", periods=n, freq="min")

        df_combined, _ = combine_data_priority(data_map, grid)
        assert df_combined["Bx"].iloc[50:70].isna().all()

    def test_output_covers_full_range(self):
        n = 120
        data_map = _synthetic_data_map(n_sats=3, n=n, noise=0.5)
        grid = pd.date_range("2024-05-01", periods=n, freq="min")

        df_combined, _ = combine_data_priority(data_map, grid)
        assert len(df_combined) == n
        assert df_combined.index[0] == grid[0]
        assert df_combined.index[-1] == grid[-1]


@pytest.mark.integration
class TestMIDLResultSchema:
    def test_dataclass_fields(self):
        result = MIDLResult(
            unpropagated=pd.DataFrame(columns=["Bx", "By", "Bz"]),
            propagated={14: pd.DataFrame(), 32: pd.DataFrame()},
            ref_x_re={},
            source_map={},
            mhd_profile=None,
        )
        assert isinstance(result.unpropagated, pd.DataFrame)
        assert isinstance(result.propagated, dict)
        assert 14 in result.propagated
        assert result.mhd_profile is None
