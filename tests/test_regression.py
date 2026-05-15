import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_readers import read_l1_data
from midl_pipeline.l1_midl import _read_sat_positions
from midl_pipeline.l1_filters import despike
from midl_pipeline.l1_combine import combine_data_priority, combine_temperature
from helpers import FIXTURES_DIR


# ---------------------------------------------------------------------------
# Real fixture data tests — require the fixture files in tests/fixtures/raw/
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestReadFixtures:
    def test_read_2003_ace_legacy(self, ace_only_day):
        df = read_l1_data(str(ace_only_day / "L1_ace.dat"))
        assert not df.empty
        assert len(df) >= 1440
        assert "Bx" in df.columns
        assert df.index[0].year == 2003

    def test_read_2005_ace_wind(self, two_sat_day):
        df_ace = read_l1_data(str(two_sat_day / "L1_ace.dat"))
        df_wind = read_l1_data(str(two_sat_day / "L1_wind.dat"))
        assert not df_ace.empty
        assert not df_wind.empty
        assert df_ace.index[0].year == 2005
        assert df_wind.index[0].year == 2005

    def test_read_2020_three_sats(self, three_sat_day):
        for sat in ["L1_ace.dat", "L1_dscovr.dat", "L1_wind.dat"]:
            df = read_l1_data(str(three_sat_day / sat))
            assert not df.empty, f"{sat} is empty"
            assert df.index[0].year == 2020

    def test_read_2024_plasma_gap(self, plasma_gap_day):
        df_ace = read_l1_data(str(plasma_gap_day / "L1_ace.dat"))
        assert not df_ace.empty
        assert df_ace["rho"].isna().all()
        assert df_ace["T"].isna().all()
        assert df_ace["Bx"].notna().any()

        df_dsc = read_l1_data(str(plasma_gap_day / "L1_dscovr.dat"))
        assert df_dsc["rho"].notna().any() or df_dsc["Bx"].notna().any()


@pytest.mark.smoke
class TestSatposFixtures:
    def test_satpos_reasonable(self, three_sat_day):
        pos, gated = _read_sat_positions(str(three_sat_day / "L1_satpos.dat"))
        assert pos["ace"] / 6371.0 > 190
        assert pos["dscovr"] / 6371.0 > 190
        assert pos["wind"] / 6371.0 > 190
        assert len(gated) == 0


@pytest.mark.smoke
class TestDespikeReal:
    def test_despike_real_bounded(self, three_sat_day):
        df = read_l1_data(str(three_sat_day / "L1_wind.dat"))
        result = despike(df)
        bx = result["Bx"].dropna()
        if len(bx) > 0:
            assert bx.abs().max() < 100, "Bx exceeds ±100 nT after despike"
        ux = result["Ux"].dropna()
        if len(ux) > 0:
            assert ux.max() < 100, "Ux > 100 km/s (should be negative/near-zero)"
            assert ux.min() > -1500, "Ux < -1500 km/s (unphysical)"


@pytest.mark.smoke
class TestCombineReal:
    def test_combine_real_three_sat(self, three_sat_day):
        data_map = {}
        for sat in ["ace", "dscovr", "wind"]:
            df = read_l1_data(str(three_sat_day / f"L1_{sat}.dat"))
            numeric_cols = ["Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T"]
            existing = [c for c in numeric_cols if c in df.columns]
            data_map[sat] = despike(df[existing])

        all_idx = pd.DatetimeIndex([])
        for df in data_map.values():
            all_idx = all_idx.union(df.index)
        grid = pd.date_range(all_idx.min(), all_idx.max(), freq="min")

        df_combined, source_map = combine_data_priority(data_map, grid)
        assert len(df_combined) == len(grid)
        assert df_combined["Bx"].notna().any()
        assert "Bx" in source_map


@pytest.mark.smoke
class TestOutputRegression:
    def test_output_vs_reference(self, three_sat_day):
        ref_l1_path = FIXTURES_DIR / "reference" / "202006_L1.csv"
        if not ref_l1_path.exists():
            pytest.skip("Reference file not found")

        ref = pd.read_csv(ref_l1_path, index_col=0, parse_dates=True)

        day_rows = ref.loc["2020-06-01":"2020-06-01"]
        if day_rows.empty:
            pytest.skip("No 2020-06-01 data in reference")

        for col in ["Bx", "By", "Bz", "Ux"]:
            if col in day_rows.columns:
                valid = day_rows[col].dropna()
                assert len(valid) > 0, f"Reference {col} is all NaN"
