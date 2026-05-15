import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_quality import (
    check_flat_plateau,
    check_near_zero,
    check_outlier_satellite,
    score_all_plasma,
)
from helpers import _synthetic_sat_df


# ---------------------------------------------------------------------------
# Flat-plateau detection
# ---------------------------------------------------------------------------

class TestFlatPlateau:
    def test_constant_ux_flagged(self):
        df = _synthetic_sat_df(n=60, ux=-400.0, noise=0.0)
        masks = check_flat_plateau(df, variables=["Ux"])
        assert masks["Ux"].sum() > 0

    def test_varying_ux_clean(self):
        df = _synthetic_sat_df(n=60, ux=-400.0, noise=5.0)
        masks = check_flat_plateau(df, variables=["Ux"])
        assert masks["Ux"].sum() == 0

    def test_uy_tight_threshold(self):
        df = _synthetic_sat_df(n=60, uy=0.0, noise=0.0)
        masks = check_flat_plateau(df, variables=["Uy"])
        assert masks["Uy"].sum() > 0

    def test_all_nan_returns_false(self):
        df = _synthetic_sat_df(n=60)
        df["Ux"] = np.nan
        masks = check_flat_plateau(df, variables=["Ux"])
        assert masks["Ux"].sum() == 0

    def test_partial_plateau(self):
        df = _synthetic_sat_df(n=100, ux=-400.0, noise=5.0)
        df.iloc[50:80, df.columns.get_loc("Ux")] = -400.0
        masks = check_flat_plateau(df, variables=["Ux"])
        assert masks["Ux"].iloc[50:80].any()
        assert not masks["Ux"].iloc[0:30].any()

    def test_window_boundary(self):
        df = _synthetic_sat_df(n=60, ux=-400.0, noise=0.0)
        masks = check_flat_plateau(df, variables=["Ux"])
        interior_flagged = masks["Ux"].iloc[10:50].any()
        assert interior_flagged


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class TestOutlierSatellite:
    def test_three_agree_no_outlier(self):
        sat_dfs = {
            "ace": _synthetic_sat_df(n=60, ux=-400.0, noise=1.0, seed=1),
            "dscovr": _synthetic_sat_df(n=60, ux=-400.0, noise=1.0, seed=2),
            "wind": _synthetic_sat_df(n=60, ux=-400.0, noise=1.0, seed=3),
        }
        result = check_outlier_satellite(sat_dfs, variables=["Ux"])
        for name in sat_dfs:
            assert result[name]["Ux"].sum() == 0

    def test_one_outlier_flagged(self):
        sat_dfs = {
            "ace": _synthetic_sat_df(n=60, ux=-400.0, noise=1.0, seed=1),
            "dscovr": _synthetic_sat_df(n=60, ux=-600.0, noise=1.0, seed=2),
            "wind": _synthetic_sat_df(n=60, ux=-400.0, noise=1.0, seed=3),
        }
        result = check_outlier_satellite(sat_dfs, variables=["Ux"])
        assert result["dscovr"]["Ux"].sum() > 0
        assert result["ace"]["Ux"].sum() == 0
        assert result["wind"]["Ux"].sum() == 0

    def test_fewer_than_3_returns_false(self):
        sat_dfs = {
            "ace": _synthetic_sat_df(n=60, ux=-400.0, seed=1),
            "dscovr": _synthetic_sat_df(n=60, ux=-600.0, seed=2),
        }
        result = check_outlier_satellite(sat_dfs, variables=["Ux"])
        for name in sat_dfs:
            assert result[name]["Ux"].sum() == 0

    def test_rho_ratio_mode(self):
        sat_dfs = {
            "ace": _synthetic_sat_df(n=60, rho=5.0, noise=0.1, seed=1),
            "dscovr": _synthetic_sat_df(n=60, rho=20.0, noise=0.1, seed=2),
            "wind": _synthetic_sat_df(n=60, rho=5.5, noise=0.1, seed=3),
        }
        result = check_outlier_satellite(sat_dfs, variables=["rho"])
        assert result["dscovr"]["rho"].sum() > 0

    def test_all_nan_sat_excluded(self):
        sat_dfs = {
            "ace": _synthetic_sat_df(n=60, ux=-400.0, seed=1),
            "dscovr": _synthetic_sat_df(n=60, ux=-400.0, seed=2),
            "wind": _synthetic_sat_df(n=60, ux=-400.0, seed=3),
        }
        sat_dfs["wind"]["Ux"] = np.nan
        result = check_outlier_satellite(sat_dfs, variables=["Ux"])
        for name in sat_dfs:
            assert result[name]["Ux"].sum() == 0


# ---------------------------------------------------------------------------
# Near-zero
# ---------------------------------------------------------------------------

class TestNearZero:
    def test_near_zero_flagged(self):
        df = _synthetic_sat_df(n=60, uy=0.3)
        masks = check_near_zero(df, variables=["Uy"])
        assert masks["Uy"].all()

    def test_above_threshold_clean(self):
        df = _synthetic_sat_df(n=60, uy=1.0)
        masks = check_near_zero(df, variables=["Uy"])
        assert masks["Uy"].sum() == 0

    def test_nan_not_flagged(self):
        df = _synthetic_sat_df(n=60, uy=0.0)
        df["Uy"] = np.nan
        masks = check_near_zero(df, variables=["Uy"])
        assert masks["Uy"].sum() == 0

    def test_negative_values(self):
        df = _synthetic_sat_df(n=60, uy=-0.3)
        masks = check_near_zero(df, variables=["Uy"])
        assert masks["Uy"].all()


# ---------------------------------------------------------------------------
# score_all_plasma (composite)
# ---------------------------------------------------------------------------

class TestScoreAllPlasma:
    def test_composite_mask(self):
        sat_dfs = {
            "ace": _synthetic_sat_df(n=60, ux=-400.0, noise=5.0, seed=1),
            "dscovr": _synthetic_sat_df(n=60, ux=-400.0, noise=5.0, seed=2),
            "wind": _synthetic_sat_df(n=60, ux=-400.0, noise=5.0, seed=3),
        }
        result = score_all_plasma(sat_dfs)
        for name in sat_dfs:
            assert "Ux" in result[name]

    def test_dscovr_only_gets_near_zero(self):
        sat_dfs = {
            "ace": _synthetic_sat_df(n=60, uy=0.0, noise=0.0, seed=1),
            "dscovr": _synthetic_sat_df(n=60, uy=0.0, noise=0.0, seed=2),
            "wind": _synthetic_sat_df(n=60, uy=0.0, noise=0.0, seed=3),
        }
        result = score_all_plasma(sat_dfs)
        dscovr_uy_bad = result["dscovr"]["Uy"].sum()
        ace_uy_bad = result["ace"]["Uy"].sum()
        assert dscovr_uy_bad >= ace_uy_bad

    def test_empty_input(self):
        sat_dfs = {
            "ace": pd.DataFrame(),
            "dscovr": pd.DataFrame(),
            "wind": pd.DataFrame(),
        }
        result = score_all_plasma(sat_dfs)
        for name in sat_dfs:
            assert result[name] == {}
