import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_readers import (
    hapi_csv_to_df,
    imap_mag_cdf_to_df,
    imap_swapi_cdf_to_df,
    read_l1_data,
    swips_l2_to_df,
)


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

    def test_headerless_csv(self, tmp_path):
        # HAPI-spec CSV: no header row (NCEI dropped its header ~June 2026).
        path = tmp_path / "test.csv"
        path.write_text(
            "2026-07-01T00:00:00.000Z,0.47,-14.6,9.83\n"
            "2026-07-01T00:01:00.000Z,0.34,-14.7,9.39\n"
        )
        col_map = {"b_gsm_min_x": "Bx", "b_gsm_min_y": "By", "b_gsm_min_z": "Bz"}
        df = hapi_csv_to_df(str(path), col_map)
        assert len(df) == 2
        assert list(df.columns) == ["Bx", "By", "Bz"]
        assert df["Bx"].iloc[0] == 0.47

    def test_headerless_csv_column_count_mismatch(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("2026-07-01T00:00:00.000Z,0.47,-14.6\n")
        col_map = {"b_gsm_min_x": "Bx", "b_gsm_min_y": "By", "b_gsm_min_z": "Bz"}
        df = hapi_csv_to_df(str(path), col_map)
        assert df.empty

    def test_fully_empty_file(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("")
        df = hapi_csv_to_df(str(path), {"b_x": "Bx"})
        assert df.empty

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("timestamp,val\n")
        df = hapi_csv_to_df(str(path), {})
        assert df.empty


# ---------------------------------------------------------------------------
# swips_l2_to_df — synthetic SOLAR-1 SWiPS L2 NetCDF
# ---------------------------------------------------------------------------

class TestSwipsL2ToDf:
    # 2026-09-01T00:00:30 UTC as SWiPS counts it: microseconds since
    # 1958-01-01 *including* the 37 leap seconds.
    _T0_US = ((pd.Timestamp("2026-09-01T00:00:30") - pd.Timestamp("1958-01-01"))
              / pd.Timedelta(microseconds=1) + 37e6)

    def _write(self, tmp_path, v, n, t, flags=None):
        import gzip
        from netCDF4 import Dataset
        nc = tmp_path / "swips.nc"
        ds = Dataset(str(nc), "w")
        ds.createDimension("sweep", len(n))
        ds.createDimension("coordinate", 3)
        mid = ds.createVariable("sweep_mid_time", "f8", ("sweep",))
        mid[:] = self._T0_US + 60e6 * np.arange(len(n))
        vv = ds.createVariable("proton_v_uncorr_gsm", "f4", ("sweep", "coordinate"),
                               fill_value=-9999)
        vv[:] = np.asarray(v, dtype=float)
        for name, vals in (("proton_n_uncorr", n), ("proton_t_uncorr", t)):
            var = ds.createVariable(name, "f4", ("sweep",), fill_value=-9999)
            var[:] = np.asarray(vals, dtype=float)
        if flags is not None:
            fl = ds.createVariable("flags", "i4", ("sweep",))
            fl[:] = np.asarray(flags)
        ds.close()
        gz = tmp_path / "swips.nc.gz"
        gz.write_bytes(gzip.compress(nc.read_bytes()))
        return str(gz)

    def test_reads_moments_and_leap_second_time(self, tmp_path):
        path = self._write(tmp_path, [[-400.0, 5.0, -3.0]], [5.0], [1e5])
        df = swips_l2_to_df(path)
        assert list(df.columns) == ["Ux", "Uy", "Uz", "rho", "T"]
        assert df.index[0] == pd.Timestamp("2026-09-01T00:00:30")
        assert df["Ux"].iloc[0] == pytest.approx(-400.0)
        assert df["rho"].iloc[0] == pytest.approx(5.0)

    def test_drops_whole_sweep_with_gpa_fill(self, tmp_path):
        # -9999.99 slips past the -9999 _FillValue; density/T come as zeros.
        path = self._write(tmp_path,
                           [[-9999.99, -9999.99, -9999.99], [-400.0, 1.0, 1.0],
                            [-410.0, 1.0, 1.0]],
                           [0.0, 0.0, 4.0], [0.0, 9e4, 9e4])
        df = swips_l2_to_df(path)
        assert len(df) == 1
        assert df["Ux"].iloc[0] == pytest.approx(-410.0)

    def test_nonzero_flags_dropped(self, tmp_path):
        path = self._write(tmp_path, [[-400.0, 0, 0], [-410.0, 0, 0]],
                           [5.0, 5.0], [1e5, 1e5], flags=[0, 1])
        df = swips_l2_to_df(path)
        assert len(df) == 1
        assert df["Ux"].iloc[0] == pytest.approx(-400.0)

    def test_missing_file_empty(self, tmp_path):
        assert swips_l2_to_df(str(tmp_path / "nope.nc.gz")).empty


# ---------------------------------------------------------------------------
# IMAP CDF readers — synthetic CDFs
# ---------------------------------------------------------------------------

def _write_imap_cdf(path, times, variables):
    """Write a minimal CDF with a TT2000 'epoch' plus `variables`
    ({name: (array, attrs)})."""
    import cdflib
    from cdflib.cdfwrite import CDF
    tt = cdflib.cdfepoch.compute_tt2000(
        [[t.year, t.month, t.day, t.hour, t.minute, t.second, 0, 0, 0]
         for t in times])
    cdf = CDF(str(path), delete=True)
    cdf.write_var({"Variable": "epoch", "Data_Type": 33, "Num_Elements": 1,
                   "Rec_Vary": True, "Dim_Sizes": []},
                  var_attrs={}, var_data=np.asarray(tt, dtype=np.int64))
    for name, (data, attrs) in variables.items():
        data = np.asarray(data)
        dtype = 45 if data.dtype.kind == "f" else 4   # CDF_DOUBLE / CDF_INT4
        cdf.write_var({"Variable": name, "Data_Type": dtype, "Num_Elements": 1,
                       "Rec_Vary": True, "Dim_Sizes": list(data.shape[1:])},
                      var_attrs=attrs, var_data=data)
    cdf.close()
    return str(path)


class TestImapSwapiCdfToDf:
    def test_reads_and_screens(self, tmp_path):
        times = pd.date_range("2026-07-15T00:00:30", periods=3, freq="1min")
        fill = -1e31
        path = _write_imap_cdf(tmp_path / "swapi.cdf", times, {
            "proton_sw_velocity_rtn_sun": (
                [[400.0, 5.0, -2.0], [410.0, 1.0, 1.0], [fill, fill, fill]],
                {"FILLVAL": fill, "VALIDMIN": -10000.0, "VALIDMAX": 10000.0}),
            "proton_sw_density": ([5.0, fill, 4.0],
                                  {"FILLVAL": fill, "VALIDMIN": 0.0,
                                   "VALIDMAX": 150.0}),
            "proton_sw_temperature": ([1e5, 2e5, 9e4], {"FILLVAL": fill}),
            # record 1: bit 2 (bad fit) -> dropped; record 2: bit 4 caution
            "swp_flags": (np.array([0, 4, 16], dtype=np.int32), {}),
        })
        df = imap_swapi_cdf_to_df(path)
        assert list(df.columns) == ["VR", "VT", "VN", "rho", "T"]
        assert len(df) == 2
        assert df.index[0] == times[0]
        assert df["VR"].iloc[0] == pytest.approx(400.0)
        assert np.isnan(df["VR"].iloc[1])          # fill -> NaN, row kept

    def test_missing_file_empty(self, tmp_path):
        assert imap_swapi_cdf_to_df(str(tmp_path / "nope.cdf")).empty


class TestImapMagCdfToDf:
    def test_reads_and_drops_bad_quality(self, tmp_path):
        times = pd.date_range("2026-07-15", periods=3, freq="1s")
        path = _write_imap_cdf(tmp_path / "mag.cdf", times, {
            "b_gsm": ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                      {"FILLVAL": -1e31}),
            "quality_flags": (np.array([0, 1, 0], dtype=np.int32), {}),
        })
        df = imap_mag_cdf_to_df(path)
        assert list(df.columns) == ["Bx", "By", "Bz"]
        assert len(df) == 2
        assert df["Bz"].iloc[1] == pytest.approx(9.0)
