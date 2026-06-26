"""
l1_pipeline.py
--------------
Per-satellite download, processing, and L1 file creation.

Downloads raw CDF/NC data, resamples to 1-min, rotates to GSM, converts
units, and writes per-satellite files to L1_raw/. Skipped if L1_raw
already contains data for the day.

Public entry points:
  download_day(day, cda)            -- Download + write L1_raw.
  create_position_file(day, cda)    -- Write L1_satpos.dat (noon positions).
"""
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import spacepy.coordinates as sc
from spacepy.time import Ticktock

from .l1_downloaders import (
    download_cdaweb_files,
    download_dscovr_ngdc,
    download_position_cdaweb_files,
    download_solar1_hapi,
    download_solar1_position_hapi,
)
from .l1_readers import cdf_to_df, hapi_csv_to_df, nc_gz_to_df


def gse_to_gsm(df, cols):
    """Rotate vector columns from GSE to GSM in-place."""
    if df.empty:
        return df
    vec_gse = df[cols].values
    times = df.index.to_pydatetime()
    t = Ticktock(times, 'UTC')
    c_gse = sc.Coords(vec_gse, 'GSE', 'car', ticks=t)
    c_gsm = c_gse.convert('GSM', 'car')
    vec_gsm = c_gsm.data
    df[cols[0]] = vec_gsm[:, 0]
    df[cols[1]] = vec_gsm[:, 1]
    df[cols[2]] = vec_gsm[:, 2]
    return df


_NGDC_F1M_VARS = {
    'proton_vx_gsm': ['Ux'],
    'proton_vy_gsm': ['Uy'],
    'proton_vz_gsm': ['Uz'],
    'proton_density': ['rho'],
    'proton_temperature': ['T'],
}

_NGDC_M1M_VARS = {
    'bx_gsm': ['Bx'],
    'by_gsm': ['By'],
    'bz_gsm': ['Bz'],
}


def _write_l1_dat(df, output_file, source_label):
    """Write one L1-format ASCII file from a 1-minute DataFrame."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'{source_label} (nT, km/s, cm^-3, K)\n')
        f.write('year month day hour minute Bx By Bz Ux Uy Uz rho T\n')
        f.write('#START\n')

        for t, row in df.iterrows():
            f.write(
                f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} "
                f"{row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} "
                f"{row['Ux']:9.2f} {row['Uy']:9.2f} {row['Uz']:9.2f} "
                f"{row['rho']:9.4f} {row['T']:10.1f}\n"
            )


def process_satellite(
    sat_name,
    mag_file_pattern,
    plasma_file_pattern,
    var_map,
    data_dir,
    trange_start,
    trange_end,
    cleanup_cdfs=True,
    raw_base='L1_raw',
):
    """Download-phase processing for one CDAWeb-sourced satellite.

    Reads separate MAG and plasma CDF files, resamples both to 1-minute,
    applies coordinate rotation and temperature conversion, then writes
    a raw (pre-despike) L1_<sat_name>.dat file to L1_raw/.

    Does NOT despike or write to L1/ -- that is handled by
    process_raw_to_filtered().

    Parameters
    ----------
    sat_name : str  ('ace' | 'wind')
    mag_file_pattern : str  Glob prefix for the magnetometer CDF (e.g. 'ac_h0_mfi').
    plasma_file_pattern : str  Glob prefix for the plasma CDF (e.g. 'ac_h0_swe').
    var_map : dict  Variable-name mapping (see download_day for examples).
    data_dir : str  Root scratch directory where CDFs were downloaded.
    trange_start, trange_end : str  'YYYY-MM-DD' day boundaries for the output grid.
    cleanup_cdfs : bool  Remove raw CDF files after writing output (default True).
    """
    print(f'\nProcessing {sat_name}...')

    # Find downloaded MAG/PLASMA CDF files for this satellite/day.
    mag_files = glob.glob(
        f'{data_dir}/**/{mag_file_pattern}*.cdf', recursive=True)
    plasma_files = glob.glob(
        f'{data_dir}/**/{plasma_file_pattern}*.cdf', recursive=True)

    if not mag_files and not plasma_files:
        print(f'Missing both MAG and Plasma files for {sat_name}. Skipping.')
        return

    # Read ALL matching CDFs and concatenate.  glob.glob() returns files
    # in arbitrary filesystem order, so reading only [0] can silently pick
    # the wrong day's CDF when multiple files exist for the time range.
    if mag_files:
        mag_frames = [cdf_to_df(f, var_map['mag_time'], var_map['mag_vars'])
                      for f in sorted(mag_files)]
        df_mag = pd.concat([f for f in mag_frames if not f.empty]).sort_index()
        df_mag = df_mag[~df_mag.index.duplicated(keep='first')]
    else:
        print(f'Missing MAG files for {sat_name}. Filling with NaNs.')
        mag_cols = []
        for cols in var_map['mag_vars'].values():
            mag_cols.extend(cols)
        df_mag = pd.DataFrame(columns=mag_cols)

    if plasma_files:
        plasma_frames = [cdf_to_df(f, var_map['plasma_time'], var_map['plasma_vars'])
                         for f in sorted(plasma_files)]
        df_plasma = pd.concat(
            [f for f in plasma_frames if not f.empty]).sort_index()
        df_plasma = df_plasma[~df_plasma.index.duplicated(keep='first')]
    else:
        print(f'Missing Plasma files for {sat_name}. Filling with NaNs.')
        plasma_cols = []
        for cols in var_map['plasma_vars'].values():
            plasma_cols.extend(cols)
        df_plasma = pd.DataFrame(columns=plasma_cols)

    # Resample both streams onto one 1-minute master grid.
    grid = pd.date_range(start=trange_start, end=trange_end, freq='1min')
    df_master = pd.DataFrame(index=grid)

    if not df_mag.empty:
        df_mag_res = df_mag.resample(
            '1min').mean().interpolate(method='time', limit=1)
    else:
        df_mag_res = pd.DataFrame(columns=df_mag.columns)

    if not df_plasma.empty:
        df_plasma_res = df_plasma.resample(
            '1min').mean().interpolate(method='time', limit=1)
    else:
        df_plasma_res = pd.DataFrame(columns=df_plasma.columns)

    df_final = df_master.join(df_mag_res).join(df_plasma_res)

    # Rotate vectors only where source coordinates require it.
    if sat_name == 'dscovr':
        print('Rotating DSCOVR data from GSE to GSM...')
        gse_to_gsm(df_final, ['Bx', 'By', 'Bz'])
        gse_to_gsm(df_final, ['Ux', 'Uy', 'Uz'])
    elif sat_name == 'wind':
        print('Rotating WIND Plasma data from GSE to GSM...')
        gse_to_gsm(df_final, ['Ux', 'Uy', 'Uz'])

    # WIND temperature comes in as thermal speed; convert to Kelvin.
    if 'v_th' in df_final.columns:
        df_final['T'] = df_final['v_th'] ** 2 * 60.5
    elif 'T' not in df_final.columns:
        df_final['T'] = np.nan

    # Keep obviously bad temperature fill values out of output.
    df_final.loc[df_final['T'] > 1e9, 'T'] = np.nan

    # T and rho come from the same plasma file: if rho is NaN the thermal
    # speed is either absent or a fill value, so T must also be NaN.
    if 'rho' in df_final.columns:
        df_final.loc[df_final['rho'].isna(), 'T'] = np.nan

    # Skip writing if the satellite had no actual data for this day.
    check_cols = [c for c in ('Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho')
                  if c in df_final.columns]
    if check_cols and df_final[check_cols].isna().all().all():
        print(f'  {sat_name}: all data is NaN for this day, skipping L1_raw write.')
    else:
        dt_start = datetime.strptime(trange_start, '%Y-%m-%d')
        raw_output_dir = os.path.join(raw_base, dt_start.strftime('%Y/%m/%d'))
        os.makedirs(raw_output_dir, exist_ok=True)
        raw_output_file = os.path.join(raw_output_dir, f'L1_{sat_name}.dat')
        _write_l1_dat(
            df_final,
            raw_output_file,
            f'Produced from {sat_name} CDAWeb CDFs (raw)',
        )
        print(f'Saved {raw_output_file}')

    # Remove raw CDFs once we have daily output files.
    if cleanup_cdfs:
        print(f'Cleaning up {sat_name} CDFs...')
        for fpath in mag_files + plasma_files:
            try:
                os.remove(fpath)
            except Exception as e:
                print(f'Could not remove {fpath}: {e}')


def process_satellite_ngdc(day, data_dir, trange_start, trange_end, cleanup=True,
                           raw_base='L1_raw'):
    """Download-phase processing for DSCOVR using NOAA NGDC 1-minute products.

    Downloads f1m (plasma) and m1m (magnetometer) gzipped NetCDF files,
    resamples to 1-minute, and writes raw L1_dscovr.dat to L1_raw/.
    NGDC data is already in GSM so no rotation is needed.

    Does NOT despike or write to L1/ -- that is handled by
    process_raw_to_filtered().

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    data_dir : str  Scratch directory for temporary downloads.
    trange_start, trange_end : str  Day boundaries for the output grid.
    cleanup : bool  Remove downloaded .nc.gz files after writing (default True).
    """
    print('\nProcessing DSCOVR (NGDC)...')

    # Download 1-minute plasma + mag products from NGDC.
    try:
        paths = download_dscovr_ngdc(day, data_dir)
    except RuntimeError as e:
        print(f'  WARNING: DSCOVR unavailable -- {e}')
        paths = {}

    if 'f1m' in paths:
        df_plasma = nc_gz_to_df(paths['f1m'], 'time', _NGDC_F1M_VARS)
    else:
        print('  Missing f1m (plasma) -- DSCOVR plasma will be NaN.')
        df_plasma = pd.DataFrame(columns=['Ux', 'Uy', 'Uz', 'rho', 'T'])

    if 'm1m' in paths:
        df_mag = nc_gz_to_df(paths['m1m'], 'time', _NGDC_M1M_VARS)
    else:
        print('  Missing m1m (mag) -- DSCOVR mag will be NaN.')
        df_mag = pd.DataFrame(columns=['Bx', 'By', 'Bz'])

    # Resample to the same 1-minute master grid as other satellites.
    grid = pd.date_range(start=trange_start, end=trange_end, freq='1min')
    df_master = pd.DataFrame(index=grid)

    if not df_mag.empty:
        df_mag_res = df_mag.resample(
            '1min').mean().interpolate(method='time', limit=1)
    else:
        df_mag_res = pd.DataFrame(columns=['Bx', 'By', 'Bz'])

    if not df_plasma.empty:
        df_plasma_res = df_plasma.resample(
            '1min').mean().interpolate(method='time', limit=1)
    else:
        df_plasma_res = pd.DataFrame(columns=['Ux', 'Uy', 'Uz', 'rho', 'T'])

    df_final = df_master.join(df_mag_res).join(df_plasma_res)

    # NGDC temperature is already Kelvin; just guard bad outliers.
    if 'T' not in df_final.columns:
        df_final['T'] = np.nan
    df_final.loc[df_final['T'] > 1e9, 'T'] = np.nan

    # Skip writing if DSCOVR had no actual data for this day.
    check_cols = [c for c in ('Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho')
                  if c in df_final.columns]
    if check_cols and df_final[check_cols].isna().all().all():
        print('  dscovr: all data is NaN for this day, skipping L1_raw write.')
    else:
        dt_start = datetime.strptime(trange_start, '%Y-%m-%d')
        raw_output_dir = os.path.join(raw_base, dt_start.strftime('%Y/%m/%d'))
        os.makedirs(raw_output_dir, exist_ok=True)
        raw_output_file = os.path.join(raw_output_dir, 'L1_dscovr.dat')
        _write_l1_dat(
            df_final,
            raw_output_file,
            'Produced from DSCOVR NOAA NGDC NetCDF (raw)',
        )
        print(f'Saved {raw_output_file}')

    # Delete temporary NGDC downloads when requested.
    if cleanup:
        print('  Cleaning up DSCOVR NGDC files...')
        for fpath in paths.values():
            try:
                os.remove(fpath)
            except Exception as e:
                print(f'  Could not remove {fpath}: {e}')


_HAPI_MAG_COL_MAP = {
    'b_gsm_min_x': 'Bx',
    'b_gsm_min_y': 'By',
    'b_gsm_min_z': 'Bz',
}


def process_satellite_hapi(day, data_dir, trange_start, trange_end,
                           cleanup=True, raw_base='L1_raw'):
    """Download-phase processing for SOLAR-1 using NCEI HAPI 1-minute MAG.

    Downloads the mag-l3 dataset (GSM, 1-min averaged), resamples to the
    standard 1-minute grid, and writes L1_solar1.dat to L1_raw/.
    Plasma columns are filled with NaN (not yet available from SOLAR-1).

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    data_dir : str  Scratch directory for temporary downloads.
    trange_start, trange_end : str  Day boundaries for the output grid.
    cleanup : bool  Remove downloaded CSV after writing (default True).
    """
    print('\nProcessing SOLAR-1 (HAPI)...')

    csv_path = download_solar1_hapi(day, data_dir)

    if csv_path is None:
        print('  SOLAR-1 mag unavailable for this day.')
        return

    df_mag = hapi_csv_to_df(csv_path, _HAPI_MAG_COL_MAP)

    if df_mag.empty:
        print('  SOLAR-1: empty mag DataFrame, skipping.')
        return

    grid = pd.date_range(start=trange_start, end=trange_end, freq='1min')
    df_master = pd.DataFrame(index=grid)

    df_mag_res = df_mag.resample('1min').mean().interpolate(
        method='time', limit=1)
    df_final = df_master.join(df_mag_res)

    for col in ('Ux', 'Uy', 'Uz', 'rho', 'T'):
        df_final[col] = np.nan

    check_cols = [c for c in ('Bx', 'By', 'Bz') if c in df_final.columns]
    if check_cols and df_final[check_cols].isna().all().all():
        print('  solar1: all mag data is NaN for this day, skipping.')
    else:
        dt_start = datetime.strptime(trange_start, '%Y-%m-%d')
        raw_output_dir = os.path.join(raw_base, dt_start.strftime('%Y/%m/%d'))
        os.makedirs(raw_output_dir, exist_ok=True)
        raw_output_file = os.path.join(raw_output_dir, 'L1_solar1.dat')
        _write_l1_dat(
            df_final,
            raw_output_file,
            'Produced from SOLAR-1 NCEI HAPI mag-l3 (raw)',
        )
        print(f'Saved {raw_output_file}')

    if cleanup and csv_path:
        try:
            os.remove(csv_path)
        except Exception as e:
            print(f'  Could not remove {csv_path}: {e}')


_SENTINEL_NAME = '.download_complete'


def download_day(day, cda, raw_dir='L1_raw'):
    """Phase 1: download raw data for all satellites and write to raw_dir/.

    Checks for a sentinel file to skip entirely on re-runs.  Per-satellite
    checks avoid re-downloading satellites whose raw file already exists.
    Also creates the position file in the same directory.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    cda : pyspedas.CDAWeb
    raw_dir : str
        Root of the data directory tree. All files (per-satellite .dat
        and L1_satpos.dat) are written to raw_dir/YYYY/MM/DD/.
    """
    dt = datetime.strptime(day, '%Y-%m-%d')
    day_raw_dir = os.path.join(raw_dir, dt.strftime('%Y/%m/%d'))
    sentinel = os.path.join(day_raw_dir, _SENTINEL_NAME)

    if os.path.exists(sentinel):
        print(f'[download_day] {day}: sentinel exists, skipping download.')
        return

    trange_start = day
    trange_end = (pd.to_datetime(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    data_dir = 'cdf_temp'

    # Determine which satellites still need downloading.
    need_ace = not os.path.exists(os.path.join(day_raw_dir, 'L1_ace.dat'))
    need_wind = not os.path.exists(os.path.join(day_raw_dir, 'L1_wind.dat'))
    need_dscovr = not os.path.exists(os.path.join(day_raw_dir, 'L1_dscovr.dat'))
    need_solar1 = not os.path.exists(os.path.join(day_raw_dir, 'L1_solar1.dat'))

    # Download CDAWeb datasets for ACE + WIND in a single API call.
    if need_ace or need_wind:
        datasets = []
        if need_ace:
            datasets.extend(['AC_H0_MFI', 'AC_H0_SWE'])
        if need_wind:
            datasets.extend(['WI_H0_MFI', 'WI_H1_SWE'])
        download_cdaweb_files(cda, datasets, trange_start, trange_end, data_dir)

    ace_map = {
        'mag_time': 'Epoch',
        'mag_vars': {'BGSM': ['Bx', 'By', 'Bz']},
        'plasma_time': 'Epoch',
        'plasma_vars': {'V_GSM': ['Ux', 'Uy', 'Uz'], 'Np': ['rho'], 'Tpr': ['T']},
    }
    win_map = {
        'mag_time': 'Epoch',
        'mag_vars': {'BGSM': ['Bx', 'By', 'Bz']},
        'plasma_time': 'Epoch',
        # Non-linear bi-Maxwellian fit (Proton_*_nonlin), not the SWE moments:
        # it better isolates the proton core. The moment temperature is biased
        # high (validated against ACE; >1e5 K in fast/transient wind) while
        # velocity and density are nearly unchanged. Proton_W_nonlin is the
        # scalar thermal speed sqrt((2*Wperp^2 + Wpar^2)/3), i.e. T=(2*Tperp+Tpar)/3.
        # OMNI uses the same non-linear fit (King & Papitashvili, 2005).
        'plasma_vars': {
            'Proton_VX_nonlin': ['Ux'],
            'Proton_VY_nonlin': ['Uy'],
            'Proton_VZ_nonlin': ['Uz'],
            'Proton_Np_nonlin': ['rho'],
            'Proton_W_nonlin': ['v_th'],
        },
    }

    if need_ace:
        process_satellite('ace', 'ac_h0_mfi', 'ac_h0_swe',
                          ace_map, data_dir, trange_start, trange_end,
                          raw_base=raw_dir)
    if need_dscovr:
        process_satellite_ngdc(day, data_dir, trange_start, trange_end,
                               raw_base=raw_dir)
    if need_wind:
        process_satellite('wind', 'wi_h0_mfi', 'wi_h1_swe',
                          win_map, data_dir, trange_start, trange_end,
                          raw_base=raw_dir)
    if need_solar1:
        process_satellite_hapi(day, data_dir, trange_start, trange_end,
                               raw_base=raw_dir)

    # Position file (always recreate -- cheap and needed by combine step).
    create_position_file(day, cda, pos_dir=raw_dir)

    # Write sentinel so future runs skip this day entirely.
    os.makedirs(day_raw_dir, exist_ok=True)
    with open(sentinel, 'w') as f:
        f.write(f'Downloaded {day}\n')
    print(f'[download_day] {day}: done.')


def create_position_file(day, cda, cleanup_cdfs=True, pos_dir='L1_raw'):
    """Write L1_satpos.dat containing mean noon GSM positions for all satellites.

    Downloads a narrow 11:00-13:00 UT window of orbit data, averages the
    X/Y/Z positions, and writes a single-row file used by the propagator to
    determine ballistic travel time.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    cda : pyspedas.CDAWeb
    cleanup_cdfs : bool  Remove CDF files after writing (default True).
    """
    print(f'\n--- Generating L1_satpos.dat for {day} ---')

    # Position output is one noon-time row per day.
    dt_day = pd.to_datetime(day)
    data_dir = 'cdf_temp'

    # Bug 2 fix: purge stale position CDFs from any previous run so that
    # glob does not pick up wrong-day files left behind by a crash.
    for pattern in ('ac_h0_mfi_*.cdf', 'wi_h0_mfi_*.cdf',
                    'dscovr_orbit_pre_*.cdf'):
        for stale in glob.glob(f'{data_dir}/**/{pattern}', recursive=True):
            try:
                os.remove(stale)
            except Exception:
                pass

    dt_start = datetime.strptime(day, '%Y-%m-%d')
    output_dir = os.path.join(pos_dir, dt_start.strftime('%Y/%m/%d'))
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, 'L1_satpos.dat')

    # Pull a narrow 11:00-13:00 UT window for stable mean position.
    urllist = download_position_cdaweb_files(cda, day, data_dir)

    if not urllist:
        print('No position files found.')
        return

    # ACE position is GSM in km, so convert to Re.
    ace_files = glob.glob(f'{data_dir}/**/ac_h0_mfi_*.cdf', recursive=True)
    ace_mean = pd.Series({'Ax': np.nan, 'Ay': np.nan, 'Az': np.nan})
    if ace_files:
        df_ace = cdf_to_df(ace_files[0], 'Epoch', {
                           'SC_pos_GSM': ['Ax', 'Ay', 'Az']})
        if not df_ace.empty and {'Ax', 'Ay', 'Az'}.issubset(df_ace.columns):
            df_ace[['Ax', 'Ay', 'Az']] /= 6371.0
            ace_mean = df_ace[['Ax', 'Ay', 'Az']].mean()

    # WIND position is already GSM in Re.
    wind_files = glob.glob(f'{data_dir}/**/wi_h0_mfi_*.cdf', recursive=True)
    wind_mean = pd.Series({'Wx': np.nan, 'Wy': np.nan, 'Wz': np.nan})
    if wind_files:
        df_wind = cdf_to_df(wind_files[0], 'Epoch', {
                            'PGSM': ['Wx', 'Wy', 'Wz']})
        if not df_wind.empty and {'Wx', 'Wy', 'Wz'}.issubset(df_wind.columns):
            wind_mean = df_wind[['Wx', 'Wy', 'Wz']].mean()

    # DSCOVR position comes in GSE km; convert to Re then rotate to GSM.
    dsc_files = glob.glob(
        f'{data_dir}/**/dscovr_orbit_pre_*.cdf', recursive=True)
    dsc_mean = pd.Series({'Dx': np.nan, 'Dy': np.nan, 'Dz': np.nan})
    if dsc_files:
        df_dsc = cdf_to_df(dsc_files[0], 'Epoch', {
                           'GSE_POS': ['Dx', 'Dy', 'Dz']})
        if not df_dsc.empty and {'Dx', 'Dy', 'Dz'}.issubset(df_dsc.columns):
            df_dsc[['Dx', 'Dy', 'Dz']] /= 6371.0
            gse_to_gsm(df_dsc, ['Dx', 'Dy', 'Dz'])
            dsc_mean = df_dsc[['Dx', 'Dy', 'Dz']].mean()

    # SOLAR-1 position from HAPI (already GSM in km).
    sol_csv = download_solar1_position_hapi(day, data_dir)
    sol_mean = pd.Series({'Sx': np.nan, 'Sy': np.nan, 'Sz': np.nan})
    if sol_csv is not None:
        df_sol = hapi_csv_to_df(
            sol_csv,
            {'sat_x_gsm': 'Sx', 'sat_y_gsm': 'Sy', 'sat_z_gsm': 'Sz'},
            fill_value=-99999,
        )
        if not df_sol.empty and {'Sx', 'Sy', 'Sz'}.issubset(df_sol.columns):
            df_sol[['Sx', 'Sy', 'Sz']] /= 6371.0
            sol_mean = df_sol[['Sx', 'Sy', 'Sz']].mean()

    # Write one merged row for downstream propagation logic.
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(
            f'Multi-Satellite Position File (GSM Coordinates, Re) for {day}\n')
        f.write('year  mo  dy  hr  mn  sc  '
                'Ax  Ay  Az  Dx  Dy  Dz  Wx  Wy  Wz  '
                'Sx  Sy  Sz  Ix  Iy  Iz\n')
        f.write('#START\n')

        def fmt(val):
            return f'{val:8.1f}' if pd.notna(val) else '     nan'

        line = (
            f"{dt_day.year:4d} {dt_day.month:2d} {dt_day.day:2d} 12  0  0 "
            f"{fmt(ace_mean['Ax'])} {fmt(ace_mean['Ay'])} {fmt(ace_mean['Az'])} "
            f"{fmt(dsc_mean['Dx'])} {fmt(dsc_mean['Dy'])} {fmt(dsc_mean['Dz'])} "
            f"{fmt(wind_mean['Wx'])} {fmt(wind_mean['Wy'])} {fmt(wind_mean['Wz'])} "
            f"{fmt(sol_mean['Sx'])} {fmt(sol_mean['Sy'])} {fmt(sol_mean['Sz'])} "
            f"{fmt(np.nan)} {fmt(np.nan)} {fmt(np.nan)}\n"
        )
        f.write(line)

    print(f'Saved {output_filepath}')

    # Cleanup raw position files after writing output.
    cleanup_files = ace_files + wind_files + dsc_files
    if sol_csv is not None:
        cleanup_files.append(sol_csv)
    if cleanup_cdfs:
        for fpath in cleanup_files:
            try:
                os.remove(fpath)
            except Exception:
                pass


