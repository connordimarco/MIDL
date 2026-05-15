import struct

import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_mhd import (
    _fill_for_mhd,
    _prepend_spinup_pad,
    _read_idl_record_file,
    _read_runtime_from_header,
    _write_l1_dat,
    _NUMERIC_COLS,
    _IDL_NDOUBLE,
)
from helpers import _synthetic_sat_df, _synthetic_sat_df_with_gap


# ---------------------------------------------------------------------------
# _fill_for_mhd
# ---------------------------------------------------------------------------

class TestFillForMhd:
    def test_no_nan_output(self):
        df = _synthetic_sat_df_with_gap(n=60, gap_start=20, gap_end=30)
        result = _fill_for_mhd(df)
        for col in _NUMERIC_COLS:
            assert result[col].isna().sum() == 0, f"{col} still has NaN"

    def test_defaults(self):
        idx = pd.date_range("2024-05-01", periods=60, freq="min")
        df = pd.DataFrame(np.nan, index=idx, columns=_NUMERIC_COLS)
        result = _fill_for_mhd(df)
        assert result["Ux"].iloc[0] == -400.0
        assert result["rho"].iloc[0] == 5.0
        assert result["T"].iloc[0] == 1e5
        assert result["Bx"].iloc[0] == 0.0

    def test_interior_interpolated(self):
        df = _synthetic_sat_df(n=60, bx=10.0)
        df.iloc[25:30, df.columns.get_loc("Bx")] = np.nan
        df.iloc[24, df.columns.get_loc("Bx")] = 10.0
        df.iloc[30, df.columns.get_loc("Bx")] = 20.0
        result = _fill_for_mhd(df)
        assert 10.0 < result["Bx"].iloc[27] < 20.0

    def test_leading_bfilled(self):
        df = _synthetic_sat_df(n=60, bx=5.0)
        df.iloc[:10, df.columns.get_loc("Bx")] = np.nan
        result = _fill_for_mhd(df)
        assert result["Bx"].iloc[0] == 5.0

    def test_missing_column_raises(self):
        df = _synthetic_sat_df(n=60).drop(columns=["T"])
        with pytest.raises(KeyError):
            _fill_for_mhd(df)


# ---------------------------------------------------------------------------
# _prepend_spinup_pad
# ---------------------------------------------------------------------------

class TestPrependSpinupPad:
    def test_pad_length_60(self):
        df = _synthetic_sat_df(n=60)
        padded, _ = _prepend_spinup_pad(df, pd.Timedelta(hours=1))
        assert len(padded) == 60 + 60

    def test_pad_values_first_row(self):
        df = _synthetic_sat_df(n=60, bx=7.0)
        padded, _ = _prepend_spinup_pad(df, pd.Timedelta(hours=1))
        np.testing.assert_array_equal(padded["Bx"].iloc[:60].values, 7.0)

    def test_pad_timestamps(self):
        df = _synthetic_sat_df(n=60)
        padded, real_start = _prepend_spinup_pad(df, pd.Timedelta(hours=1))
        assert padded.index[0] == df.index[0] - pd.Timedelta(hours=1)
        assert real_start == df.index[0]


# ---------------------------------------------------------------------------
# _write_l1_dat
# ---------------------------------------------------------------------------

class TestWriteL1Dat:
    def test_header_format(self, tmp_path):
        df = _synthetic_sat_df(n=10)
        path = tmp_path / "L1.dat"
        from datetime import date
        ref_x = {date(2024, 5, 1): 1.5e6}
        _write_l1_dat(df, ref_x, str(path))

        with open(path) as f:
            lines = f.readlines()
        assert "MIDL" in lines[0]
        assert "#COORD" in lines[3]
        assert "GSM" in lines[4]
        assert "#TIMEDELAY" in lines[6]
        assert "#START" in lines[10]

    def test_data_row_count(self, tmp_path):
        df = _synthetic_sat_df(n=10)
        path = tmp_path / "L1.dat"
        from datetime import date
        ref_x = {date(2024, 5, 1): 1.5e6}
        _write_l1_dat(df, ref_x, str(path))

        with open(path) as f:
            lines = f.readlines()
        start_idx = next(i for i, l in enumerate(lines) if "#START" in l)
        data_lines = lines[start_idx + 1:]
        assert len(data_lines) == 10
        fields = data_lines[0].split()
        assert len(fields) == 15


# ---------------------------------------------------------------------------
# _read_idl_record_file
# ---------------------------------------------------------------------------

class TestReadIdlRecordFile:
    def test_synthetic_binary(self, tmp_path):
        path = tmp_path / "test.idl"
        n_cells = 3
        reclen = 8 * _IDL_NDOUBLE
        with open(path, "wb") as f:
            for c in range(n_cells):
                vals = [float(c * _IDL_NDOUBLE + j) for j in range(_IDL_NDOUBLE)]
                f.write(struct.pack("<i", reclen))
                f.write(struct.pack(f"<{_IDL_NDOUBLE}d", *vals))
                f.write(struct.pack("<i", reclen))

        result = _read_idl_record_file(str(path))
        assert len(result["x"]) == n_cells
        assert result["x"][0] == 1.0  # col index 1
        assert result["Rho"][0] == 4.0  # col index 4


# ---------------------------------------------------------------------------
# _read_runtime_from_header
# ---------------------------------------------------------------------------

class TestReadRuntimeFromHeader:
    def test_reads_timesimulation(self, tmp_path):
        idl_path = tmp_path / "1d__mhd_1_t00000100_n00000100_pe0000.idl"
        idl_path.write_bytes(b"")  # dummy
        h_path = tmp_path / "1d__mhd_1_t00000100_n00000100.h"
        h_path.write_text(
            "some header\n"
            "#TIMESIMULATION\n"
            " 3600.0\n"
            "other stuff\n"
        )
        result = _read_runtime_from_header(str(idl_path))
        assert result == 3600.0
