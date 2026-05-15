import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_midl import (
    MIDLResult,
    _compute_source_changed,
    _day_range,
    _read_sat_positions,
)
from helpers import _synthetic_sat_df


# ---------------------------------------------------------------------------
# _day_range
# ---------------------------------------------------------------------------

class TestDayRange:
    def test_single_day(self):
        days = list(_day_range(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-01")))
        assert days == ["2024-05-01"]

    def test_multi_day(self):
        days = list(_day_range(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-03")))
        assert len(days) == 3
        assert days == ["2024-05-01", "2024-05-02", "2024-05-03"]

    def test_month_boundary(self):
        days = list(_day_range(pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-02")))
        assert len(days) == 3
        assert "2024-02-01" in days

    def test_leap_day(self):
        days = list(_day_range(pd.Timestamp("2024-02-28"), pd.Timestamp("2024-03-01")))
        assert "2024-02-29" in days
        assert len(days) == 3

    def test_year_boundary(self):
        days = list(_day_range(pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01")))
        assert len(days) == 2
        assert "2025-01-01" in days


# ---------------------------------------------------------------------------
# _read_sat_positions
# ---------------------------------------------------------------------------

class TestReadSatPositions:
    def _write_satpos(self, path, data_line):
        with open(path, "w") as f:
            f.write("Multi-Satellite Position File (GSM Coordinates, Re)\n")
            f.write("year  mo  dy  hr  mn  sc  Ax  Ay  Az  Dx  Dy  Dz  Wx  Wy  Wz  Sx  Sy  Sz  Ix  Iy  Iz\n")
            f.write("#START\n")
            f.write(data_line + "\n")

    def test_standard_file(self, tmp_path):
        path = tmp_path / "L1_satpos.dat"
        self._write_satpos(str(path),
            "2020  6 15 12  0  0    227.8    -29.6     20.0    250.7      6.1     -8.8    213.9     94.9     -2.9      nan      nan      nan      nan      nan      nan")
        pos, gated = _read_sat_positions(str(path))
        assert abs(pos["ace"] / 6371.0 - 227.8) < 0.1
        assert abs(pos["dscovr"] / 6371.0 - 250.7) < 0.1
        assert abs(pos["wind"] / 6371.0 - 213.9) < 0.1
        assert np.isnan(pos["solar1"])
        assert len(gated) == 0

    def test_gating_190re(self, tmp_path):
        path = tmp_path / "L1_satpos.dat"
        self._write_satpos(str(path),
            "2003  1  1 12  0  0    227.8    -29.6     20.0    250.7      6.1     -8.8     50.0     94.9     -2.9      nan      nan      nan      nan      nan      nan")
        pos, gated = _read_sat_positions(str(path))
        assert "wind" in gated
        assert np.isnan(pos["wind"])

    def test_missing_file_nan(self):
        pos, gated = _read_sat_positions("/nonexistent/file.dat")
        for sat in pos:
            assert np.isnan(pos[sat])

    def test_nan_position(self, tmp_path):
        path = tmp_path / "L1_satpos.dat"
        self._write_satpos(str(path),
            "2020  6 15 12  0  0      nan      nan      nan    250.7      6.1     -8.8    213.9     94.9     -2.9      nan      nan      nan      nan      nan      nan")
        pos, gated = _read_sat_positions(str(path))
        assert np.isnan(pos["ace"])


# ---------------------------------------------------------------------------
# _compute_source_changed
# ---------------------------------------------------------------------------

class TestComputeSourceChanged:
    def test_constant_source(self):
        idx = pd.date_range("2024-05-01", periods=60, freq="min")
        source_map = {"Bx": pd.Series([frozenset({1})] * 60, index=idx)}
        result = _compute_source_changed(source_map)
        assert result["Bx"].sum() == 0

    def test_transition_detected(self):
        idx = pd.date_range("2024-05-01", periods=60, freq="min")
        sources = [frozenset({1})] * 30 + [frozenset({3})] * 30
        source_map = {"Bx": pd.Series(sources, index=idx)}
        result = _compute_source_changed(source_map)
        assert result["Bx"].iloc[30] is True or result["Bx"].iloc[30] == True
        assert result["Bx"].iloc[29] == False


# ---------------------------------------------------------------------------
# MIDLResult dataclass
# ---------------------------------------------------------------------------

class TestMIDLResult:
    def test_dataclass_fields(self):
        result = MIDLResult(
            unpropagated=pd.DataFrame(),
            propagated={},
            ref_x_re={},
            source_map={},
            mhd_profile=None,
        )
        assert hasattr(result, "unpropagated")
        assert hasattr(result, "propagated")
        assert hasattr(result, "ref_x_re")
        assert hasattr(result, "source_map")
        assert hasattr(result, "mhd_profile")
        assert result.mhd_profile is None
