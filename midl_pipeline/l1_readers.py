"""
l1_readers.py
-------------
Low-level file readers that convert raw L1 source files to DataFrames.

Four entry points:
  - cdf_to_df()      — NASA CDF files (ACE, WIND, DSCOVR orbit via CDAWeb).
  - nc_gz_to_df()    — Gzipped NetCDF files (DSCOVR 1-min products from NGDC).
  - hapi_csv_to_df() — HAPI CSV files (SOLAR-1 mag + orbit from NCEI).
  - read_l1_data()   — Custom ASCII .dat files produced by this pipeline.

All return a DataFrame indexed by a DatetimeIndex at 1-min cadence, or an
empty DataFrame on failure.  Fill/valid-range masking is applied so callers
receive NaN where the source file marks data as missing or out-of-range.
"""
import gzip
import os

import cdflib
import numpy as np
import pandas as pd
from netCDF4 import Dataset


def cdf_to_df(cdf_path, time_var, data_vars):
    """Read a CDF file into a DataFrame.

    Parameters
    ----------
    cdf_path : str
        Path to the CDF file.
    time_var : str
        Name of the CDF epoch variable (e.g. 'Epoch').
    data_vars : dict[str, list[str]]
        Mapping of CDF variable name -> output column name(s).  Vector
        variables are split into as many columns as entries in the list.

    Returns
    -------
    pd.DataFrame  (empty on error)
    """
    try:
        # Open source CDF and convert its time variable to datetimes.
        cdf = cdflib.CDF(cdf_path)
        epoch = cdf.varget(time_var)
        time_dt = cdflib.cdfepoch.to_datetime(epoch)
        data = {'timestamp': time_dt}

        for var_name, col_names in data_vars.items():
            # Pull each requested variable and apply metadata-based masking.
            val = cdf.varget(var_name).astype(np.float64)
            fillval = cdf.varattsget(var_name).get('FILLVAL', None)
            validmin = cdf.varattsget(var_name).get('VALIDMIN', None)
            validmax = cdf.varattsget(var_name).get('VALIDMAX', None)

            if val.ndim > 1:
                # Split vector-valued variables into named columns.
                for i, col in enumerate(col_names):
                    col_data = val[:, i]
                    if fillval is not None:
                        col_data = np.where(
                            col_data == fillval, np.nan, col_data)
                    if validmin is not None:
                        vmin = np.atleast_1d(validmin)
                        col_data = np.where(
                            col_data < (vmin[i] if len(vmin) > 1 else vmin[0]),
                            np.nan,
                            col_data,
                        )
                    if validmax is not None:
                        vmax = np.atleast_1d(validmax)
                        col_data = np.where(
                            col_data > (vmax[i] if len(vmax) > 1 else vmax[0]),
                            np.nan,
                            col_data,
                        )
                    data[col] = col_data
            else:
                # Scalar variable maps to a single output column.
                if fillval is not None:
                    val = np.where(val == fillval, np.nan, val)
                if validmin is not None:
                    val = np.where(val < validmin, np.nan, val)
                if validmax is not None:
                    val = np.where(val > validmax, np.nan, val)
                data[col_names[0]] = val

        return pd.DataFrame(data).set_index('timestamp')
    except Exception as e:
        print(f"Error reading CDF {cdf_path}: {e}")
        return pd.DataFrame()


def nc_gz_to_df(nc_gz_path, time_var, data_vars):
    """Read a gzipped NetCDF file into a DataFrame.

    Designed for NOAA NGDC DSCOVR products (*.nc.gz).

    Parameters
    ----------
    nc_gz_path : str
        Path to the .nc.gz file.
    time_var : str
        Name of the time variable (expected units: milliseconds since epoch).
    data_vars : dict[str, list[str]]
        Same convention as cdf_to_df().

    Returns
    -------
    pd.DataFrame  (empty on error)
    """
    try:
        # NGDC files are gzipped NetCDF, so decompress first.
        with gzip.open(nc_gz_path, 'rb') as gz_f:
            raw = gz_f.read()

        # Read from memory so we do not need temp files on disk.
        ds = Dataset('inmemory', memory=raw)
        t_raw = np.array(ds.variables[time_var][:], dtype=np.float64)
        timestamps = pd.to_datetime(t_raw, unit='ms')

        data = {'timestamp': timestamps}

        for var_name, col_names in data_vars.items():
            if var_name not in ds.variables:
                # Keep shape consistent even when a variable is missing.
                print(f"  WARNING: Variable '{var_name}' not in {nc_gz_path}")
                for col in col_names:
                    data[col] = np.full(len(timestamps), np.nan)
                continue

            var = ds.variables[var_name]
            # Turn masked arrays into float arrays with NaNs.
            val = np.ma.filled(np.ma.array(var[:], dtype=np.float64), np.nan)
            vmin = getattr(var, 'valid_min', None)
            vmax = getattr(var, 'valid_max', None)

            if val.ndim > 1:
                # Split vector variables into separate columns.
                for i, col in enumerate(col_names):
                    col_data = val[:, i]
                    if vmin is not None:
                        mn = np.atleast_1d(vmin)
                        col_data = np.where(
                            col_data < (mn[i] if len(mn) > 1 else mn[0]),
                            np.nan,
                            col_data,
                        )
                    if vmax is not None:
                        mx = np.atleast_1d(vmax)
                        col_data = np.where(
                            col_data > (mx[i] if len(mx) > 1 else mx[0]),
                            np.nan,
                            col_data,
                        )
                    data[col] = col_data
            else:
                # Scalar variable maps to one output column.
                if vmin is not None:
                    val = np.where(val < vmin, np.nan, val)
                if vmax is not None:
                    val = np.where(val > vmax, np.nan, val)
                data[col_names[0]] = val

        ds.close()
        return pd.DataFrame(data).set_index('timestamp')

    except Exception as e:
        print(f"Error reading {nc_gz_path}: {e}")
        return pd.DataFrame()


def hapi_csv_to_df(csv_path, col_map, fill_value=-9999):
    """Read a HAPI-format CSV file into a DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file downloaded from a HAPI endpoint.
    col_map : dict[str, str]
        Mapping of HAPI parameter name -> output column name.
    fill_value : float
        Value treated as missing data (replaced with NaN).

    Returns
    -------
    pd.DataFrame  (empty on error)
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        df.index.name = 'timestamp'
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.rename(columns=col_map)
        df = df.replace(fill_value, np.nan)
        return df
    except Exception as e:
        print(f'Error reading HAPI CSV {csv_path}: {e}')
        return pd.DataFrame()


def read_l1_data(filepath):
    """Read one of the pipeline's per-satellite L1 .dat ASCII files.

    The expected header format (3 lines, then data):
        Line 1: provenance comment
        Line 2: column names
        Line 3: '#START'
        Data:   space-delimited columns matching col_names below

    Parameters
    ----------
    filepath : str
        Path to the .dat file.

    Returns
    -------
    pd.DataFrame  (empty if the file is missing or unreadable)
    """
    # Column layouts: new format (5 time cols) and legacy (7 time cols).
    col_names_new = [
        'year', 'mo', 'dy', 'hr', 'mn',
        'Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T',
    ]
    col_names_legacy = [
        'year', 'mo', 'dy', 'hr', 'mn', 'sc', 'msc',
        'Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T',
    ]

    if not os.path.exists(filepath):
        # Missing daily files are expected in sparse-data cases.
        return pd.DataFrame()

    try:
        # Detect format by reading the header line (line 2, 0-indexed).
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in range(1):
                f.readline()
            header_line = f.readline()
        header_fields = header_line.split()
        has_sc_msc = 'sc' in header_fields

        if has_sc_msc:
            col_names = col_names_legacy
        else:
            col_names = col_names_new

        # Read only canonical physics columns so files can carry extra metadata.
        # Treat 999999 as NaN (fill value for missing data in DAT output).
        df = pd.read_csv(filepath, sep=r'\s+', names=col_names,
                         comment='#', skiprows=3, usecols=range(len(col_names)),
                         na_values=['999999'])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Rebuild timestamp from date/time columns.
    if has_sc_msc:
        dt_cols = df[['year', 'mo', 'dy', 'hr', 'mn', 'sc']].copy()
        dt_cols.columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
        df['timestamp'] = pd.to_datetime(
            dt_cols) + pd.to_timedelta(df['msc'], unit='ms')
    else:
        dt_cols = df[['year', 'mo', 'dy', 'hr', 'mn']].copy()
        dt_cols.columns = ['year', 'month', 'day', 'hour', 'minute']
        df['timestamp'] = pd.to_datetime(dt_cols)

    return df.set_index('timestamp')
