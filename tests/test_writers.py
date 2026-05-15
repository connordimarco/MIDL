import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_writers import _frozenset_to_str, write_monthly_outputs
from midl_pipeline.l1_midl import MIDLResult
from helpers import _synthetic_sat_df


def _make_result(n=1440, start="2024-05-01", boundaries=(14, 32)):
    """Build a minimal MIDLResult for writer tests."""
    df = _synthetic_sat_df(n=n, start=start)
    propagated = {}
    for b in boundaries:
        propagated[b] = _synthetic_sat_df(n=n, start=start)

    grid = df.index
    source_map = {
        "Bx": pd.Series([frozenset({1, 3})] * n, index=grid),
        "By": pd.Series([frozenset({1, 3})] * n, index=grid),
        "Bz": pd.Series([frozenset({1, 3})] * n, index=grid),
        "Ux": pd.Series([frozenset({1})] * n, index=grid),
        "Uy": pd.Series([frozenset({2, 3})] * n, index=grid),
        "Uz": pd.Series([frozenset({2, 3})] * n, index=grid),
        "rho": pd.Series([frozenset({1})] * n, index=grid),
        "T": pd.Series([frozenset({1, 2, 3})] * n, index=grid),
    }

    from datetime import date
    ref_x_re = {date(2024, 5, 1): 220.0}

    return MIDLResult(
        unpropagated=df,
        propagated=propagated,
        ref_x_re=ref_x_re,
        source_map=source_map,
        mhd_profile=None,
    )


# ---------------------------------------------------------------------------
# _frozenset_to_str
# ---------------------------------------------------------------------------

class TestFrozensetToStr:
    def test_single(self):
        assert _frozenset_to_str(frozenset({1})) == "1"

    def test_multi_sorted(self):
        assert _frozenset_to_str(frozenset({3, 1})) == "13"

    def test_none_empty(self):
        assert _frozenset_to_str(None) == ""

    def test_empty_set(self):
        assert _frozenset_to_str(frozenset()) == ""

    def test_all_four(self):
        assert _frozenset_to_str(frozenset({1, 2, 3, 4})) == "1234"


# ---------------------------------------------------------------------------
# write_monthly_outputs
# ---------------------------------------------------------------------------

class TestWriteMonthlyOutputs:
    def test_creates_dirs(self, tmp_output_dir):
        result = _make_result(n=60)
        write_monthly_outputs(result, str(tmp_output_dir))
        assert (tmp_output_dir / "2024" / "05").exists()

    def test_l1_written(self, tmp_output_dir):
        result = _make_result(n=60)
        write_monthly_outputs(result, str(tmp_output_dir))
        assert (tmp_output_dir / "2024" / "05" / "202405_L1.csv").exists()

    def test_propagated_written(self, tmp_output_dir):
        result = _make_result(n=60)
        write_monthly_outputs(result, str(tmp_output_dir))
        assert (tmp_output_dir / "2024" / "05" / "202405_14Re.csv").exists()
        assert (tmp_output_dir / "2024" / "05" / "202405_32Re.csv").exists()

    def test_source_columns(self, tmp_output_dir):
        result = _make_result(n=60)
        write_monthly_outputs(result, str(tmp_output_dir))
        df = pd.read_csv(
            tmp_output_dir / "2024" / "05" / "202405_L1.csv", index_col=0)
        for col in ["B_source", "Ux_source", "Uyz_source", "rho_source", "T_source"]:
            assert col in df.columns

    def test_timestamp_format(self, tmp_output_dir):
        result = _make_result(n=60)
        write_monthly_outputs(result, str(tmp_output_dir))
        df = pd.read_csv(
            tmp_output_dir / "2024" / "05" / "202405_L1.csv", index_col=0)
        ts = df.index[0]
        assert "T" in ts
        assert len(ts) == 19  # YYYY-MM-DDTHH:MM:SS

    def test_csv_precision(self, tmp_output_dir):
        result = _make_result(n=60)
        result.unpropagated["Bx"] = 1.23456
        result.unpropagated["Ux"] = -400.56789
        result.unpropagated["rho"] = 5.123456
        result.unpropagated["T"] = 100123.7
        write_monthly_outputs(result, str(tmp_output_dir))
        df = pd.read_csv(
            tmp_output_dir / "2024" / "05" / "202405_L1.csv", index_col=0)
        assert df["Bx"].iloc[0] == 1.23
        assert df["Ux"].iloc[0] == -400.6
        assert df["rho"].iloc[0] == 5.123
        assert df["T"].iloc[0] == 100124.0

    def test_cross_month(self, tmp_output_dir):
        n = 2880  # 2 days
        result = _make_result(n=n, start="2024-05-31")
        write_monthly_outputs(result, str(tmp_output_dir))
        assert (tmp_output_dir / "2024" / "05" / "202405_L1.csv").exists()
        assert (tmp_output_dir / "2024" / "06" / "202406_L1.csv").exists()

    def test_roundtrip_read_write(self, tmp_output_dir):
        result = _make_result(n=60)
        write_monthly_outputs(result, str(tmp_output_dir))
        df = pd.read_csv(
            tmp_output_dir / "2024" / "05" / "202405_L1.csv",
            index_col=0, parse_dates=True)
        assert len(df) == 60
        assert df["Bx"].notna().all()
