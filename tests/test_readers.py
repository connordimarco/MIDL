import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_readers import hapi_csv_to_df, read_l1_data


# ---------------------------------------------------------------------------
# read_l1_data — synthetic files
# ---------------------------------------------------------------------------

class TestReadL1Data:
    def _write_dat(self, path, header_line, rows):
        with open(path, "w") as f:
            f.write("Test provenance line\n")
            f.write(header_line + "\n")
            f.write("#START\n")
            for row in rows:
                f.write(row + "\n")

    def test_new_format(self, tmp_path):
        path = tmp_path / "test.dat"
        header = "year month day hour minute Bx By Bz Ux Uy Uz rho T"
        rows = [
            "2024  1  1  0  0     1.0     2.0     3.0   -400.0      1.0      2.0    5.0000    100000.0",
            "2024  1  1  0  1     1.1     2.1     3.1   -401.0      1.1      2.1    5.1000    100100.0",
        ]
        self._write_dat(str(path), header, rows)
        df = read_l1_data(str(path))
        assert len(df) == 2
        assert "Bx" in df.columns
        assert df.index[0] == pd.Timestamp("2024-01-01 00:00:00")

    def test_legacy_format(self, tmp_path):
        path = tmp_path / "test.dat"
        header = "year mo dy hr mn sc msc Bx By Bz Ux Uy Uz rho T"
        rows = [
            "2020  6 15  0  0  0   0    -2.99    -2.28     0.69   -301.22      7.90     30.42    5.0000    100000.0",
            "2020  6 15  0  1  0   0    -2.82    -2.53     0.69   -300.89      5.13     25.62    5.1000    100100.0",
        ]
        self._write_dat(str(path), header, rows)
        df = read_l1_data(str(path))
        assert len(df) == 2
        assert df.index[0] == pd.Timestamp("2020-06-15 00:00:00")

    def test_999999_to_nan(self, tmp_path):
        path = tmp_path / "test.dat"
        header = "year month day hour minute Bx By Bz Ux Uy Uz rho T"
        rows = [
            "2024  1  1  0  0     1.0     2.0     3.0   -400.0      1.0      2.0    999999    100000.0",
        ]
        self._write_dat(str(path), header, rows)
        df = read_l1_data(str(path))
        assert pd.isna(df["rho"].iloc[0])

    def test_missing_file_empty(self):
        df = read_l1_data("/nonexistent/path/file.dat")
        assert df.empty

    def test_timestamp_construction(self, tmp_path):
        path = tmp_path / "test.dat"
        header = "year month day hour minute Bx By Bz Ux Uy Uz rho T"
        rows = [
            "2024  3 15 12 30     1.0     2.0     3.0   -400.0      1.0      2.0    5.0000    100000.0",
        ]
        self._write_dat(str(path), header, rows)
        df = read_l1_data(str(path))
        assert df.index[0] == pd.Timestamp("2024-03-15 12:30:00")

    def test_correct_columns(self, tmp_path):
        path = tmp_path / "test.dat"
        header = "year month day hour minute Bx By Bz Ux Uy Uz rho T"
        rows = [
            "2024  1  1  0  0     1.0     2.0     3.0   -400.0      1.0      2.0    5.0000    100000.0",
        ]
        self._write_dat(str(path), header, rows)
        df = read_l1_data(str(path))
        for col in ["Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T"]:
            assert col in df.columns

    def test_empty_file_after_header(self, tmp_path):
        path = tmp_path / "test.dat"
        self._write_dat(str(path), "year month day hour minute Bx By Bz Ux Uy Uz rho T", [])
        df = read_l1_data(str(path))
        assert df.empty


# ---------------------------------------------------------------------------
# hapi_csv_to_df
# ---------------------------------------------------------------------------

class TestHapiCsvToDf:
    def test_reads_simple_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text(
            "timestamp,b_x,b_y,b_z\n"
            "2026-04-15T00:00:00,1.0,2.0,3.0\n"
            "2026-04-15T00:01:00,1.1,2.1,3.1\n"
        )
        col_map = {"b_x": "Bx", "b_y": "By", "b_z": "Bz"}
        df = hapi_csv_to_df(str(path), col_map)
        assert len(df) == 2
        assert "Bx" in df.columns

    def test_fill_value_nan(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text(
            "timestamp,val\n"
            "2026-04-15T00:00:00,-9999\n"
            "2026-04-15T00:01:00,5.0\n"
        )
        df = hapi_csv_to_df(str(path), {})
        assert pd.isna(df["val"].iloc[0])
        assert df["val"].iloc[1] == 5.0

    def test_col_rename(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text(
            "timestamp,old_name\n"
            "2026-04-15T00:00:00,1.0\n"
        )
        df = hapi_csv_to_df(str(path), {"old_name": "Bx"})
        assert "Bx" in df.columns

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("timestamp,val\n")
        df = hapi_csv_to_df(str(path), {})
        assert df.empty
