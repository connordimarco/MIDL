"""
l1_writers.py
-------------
Output formatters for MIDLResult objects.

Writes monthly CSV files to:
    output_dir/YYYY/MM/YYYYMM_{L1,14Re,32Re}.csv

Usage:
    from midl_pipeline import midl, write_monthly_outputs

    result = midl('2024-05-09', '2024-05-11')
    write_monthly_outputs(result, output_dir='data')
"""
import os

import pandas as pd


def write_monthly_outputs(result, output_dir='data'):
    """Write MIDLResult to monthly CSV files.

    Creates directory structure:
        output_dir/YYYY/MM/YYYYMM_{L1,14Re,32Re}.csv

    For unpropagated data, an X_Re column is added containing the X_GSM
    distance (in Earth radii) of the reference satellite for each day.

    When `result.mhd_profile` is not None, also writes per-Re MHD CSVs
    to output_dir/YYYY/MM/mhd/YYYYMM_mhd_RRRRe.csv at x = -70, -69,
    ..., 69, 70 Re (141 files per month).

    Parameters
    ----------
    result : MIDLResult
    output_dir : str
    """
    datasets = _prepare_datasets(result)

    for label, df in datasets.items():
        if df.empty:
            continue

        df = df.copy()
        df['_ym'] = df.index.to_period('M')

        n_months = 0
        for period, group in df.groupby('_ym'):
            group = group.drop(columns='_ym')
            group.index.name = 'timestamp'

            year_month_dir = os.path.join(
                output_dir, f'{period.year:04d}', f'{period.month:02d}')

            ym_prefix = f'{period.year:04d}{period.month:02d}'
            _write_csv(group, label, year_month_dir, ym_prefix)
            n_months += 1

        print(f'Wrote {n_months} monthly files for {label} to {output_dir}/')

    # MHD profile (optional): per-Re CSVs at x = -70..70 Re.
    if getattr(result, 'mhd_profile', None) is not None:
        _write_mhd_monthly(result.mhd_profile, output_dir)


def _frozenset_to_str(fs):
    """Convert frozenset of satellite codes to a sorted digit string.

    1=ACE, 2=DSCOVR, 3=WIND, 4=SOLAR-1, 5=IMAP (reserved).
    e.g. frozenset({1,3}) -> '13'.
    """
    if fs is None or (isinstance(fs, frozenset) and not fs):
        return ''
    return ''.join(str(c) for c in sorted(fs))


# Map from source_map variable keys to output column names.
_SOURCE_COLUMNS = {
    'B_source':   'Bx',    # Bx/By/Bz all share the same source
    'Ux_source':  'Ux',
    'Uyz_source': 'Uy',    # Uy/Uz share the same source
    'rho_source': 'rho',
    'T_source':   'T',
}

# Interpolation-provenance flag columns.  Mirror the source-column naming
# (`<group>_interp`) and map each group to the data column whose presence
# decides whether the flag is written (blank where that value is missing).
#   output_col -> (flag-group key in result.interp_flags, gating data column)
_INTERP_COLUMNS = {
    'B_interp':   ('B',   'Bx'),
    'Ux_interp':  ('Ux',  'Ux'),
    'Uyz_interp': ('Uyz', 'Uy'),
    'rho_interp': ('rho', 'rho'),
    'T_interp':   ('T',   'T'),
}


def _attach_interp_flags(df, flag_map):
    """Add integer `<group>_interp` columns to *df* from *flag_map*.

    flag_map : dict[group -> pd.Series of int]  (provenance levels:
        0 all-direct, 1 mixed, 2 all-interpolated, 3 Stage-6 fill).
    A flag cell is left blank (NaN) where the corresponding data value is
    missing, so provenance is only asserted where there is a value.
    """
    if not flag_map:
        return df
    for out_col, (group, gate_col) in _INTERP_COLUMNS.items():
        flag = flag_map.get(group)
        if flag is None:
            continue
        flag = flag.reindex(df.index)
        if gate_col in df.columns:
            # Blank the flag where the value itself is absent.
            flag = flag.where(df[gate_col].notna())
        # Nullable integer so blanks stay blank (no trailing .0) in CSV.
        df[out_col] = flag.astype('Int64')
    return df


def _prepare_datasets(result):
    """Build dict of {label: DataFrame} from MIDLResult.

    For 'unpropagated', adds X_Re and source columns using result metadata.
    """
    datasets = {}

    df_unp = result.unpropagated.copy()
    if result.ref_x_re:
        df_unp['X'] = df_unp.index.map(
            lambda t: result.ref_x_re.get(t.date(), float('nan')))

    # Add source provenance columns.
    if result.source_map:
        for out_col, src_key in _SOURCE_COLUMNS.items():
            if src_key in result.source_map:
                src = result.source_map[src_key].reindex(df_unp.index)
                df_unp[out_col] = src.map(
                    lambda fs: _frozenset_to_str(fs) or float('nan'))

    # Add interpolation-provenance flag columns (additive).
    df_unp = _attach_interp_flags(df_unp, getattr(result, 'interp_flags', None))

    datasets['L1'] = df_unp

    prop_flags = getattr(result, 'propagated_interp_flags', None) or {}
    for b_re, df in result.propagated.items():
        df = df.copy()
        df = _attach_interp_flags(df, prop_flags.get(b_re))
        datasets[f'{b_re}Re'] = df

    return datasets


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_CSV_PRECISION = {
    'Bx': '%.2f', 'By': '%.2f', 'Bz': '%.2f',
    'Ux': '%.1f', 'Uy': '%.2f', 'Uz': '%.2f',
    'rho': '%.3f', 'T': '%.0f', 'X': '%.1f',
}


_MHD_FLOAT_VARS = ('Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T')
_MHD_RE_SLICES = tuple(range(-70, 71))


def _write_mhd_monthly(mhd_ds, output_dir):
    """Split an xr.Dataset by calendar month and write per-Re MHD CSVs.

    Writes output_dir/YYYY/MM/mhd/YYYYMM_mhd_RRRRe.csv for each Re in
    _MHD_RE_SLICES (-70, -69, ..., 69, 70).  Each file has the same
    8-column schema and precision as the existing ballistic _14Re/_32Re
    CSVs.
    """
    if mhd_ds is None or len(mhd_ds.time) == 0:
        return

    times = pd.DatetimeIndex(mhd_ds.time.values)
    ym_key = times.strftime('%Y-%m')
    unique_ym = sorted(set(ym_key))

    n_months = 0
    n_files = 0
    for ym in unique_ym:
        year = int(ym[:4])
        month = int(ym[5:7])
        sel_mask = (ym_key == ym)
        group = mhd_ds.isel(time=sel_mask.nonzero()[0])
        group_times = pd.DatetimeIndex(group.time.values)

        mhd_dir = os.path.join(
            output_dir, f'{year:04d}', f'{month:02d}', 'mhd')
        os.makedirs(mhd_dir, exist_ok=True)
        ym_prefix = f'{year:04d}{month:02d}'

        for re_val in _MHD_RE_SLICES:
            sl = group.sel(x=re_val, method='nearest')
            df = pd.DataFrame(
                {v: sl[v].values for v in _MHD_FLOAT_VARS if v in sl.data_vars},
                index=group_times)
            df.index.name = 'timestamp'
            path = os.path.join(
                mhd_dir, f'{ym_prefix}_mhd_{re_val:03d}Re.csv')
            rounded = df.copy()
            for col, spec in _CSV_PRECISION.items():
                if col in rounded.columns:
                    decimals = int(spec.split('.')[1].rstrip('f'))
                    rounded[col] = rounded[col].round(decimals)
            rounded.to_csv(path, date_format='%Y-%m-%dT%H:%M:%S')
            n_files += 1
        n_months += 1

    print(f'Wrote {n_files} MHD CSVs across {n_months} months to {output_dir}/')


def _write_csv(df, label, year_month_dir, ym_prefix):
    """Write one monthly CSV file."""
    os.makedirs(year_month_dir, exist_ok=True)
    path = os.path.join(year_month_dir, f'{ym_prefix}_{label}.csv')
    rounded = df.copy()
    for col, spec in _CSV_PRECISION.items():
        if col in rounded.columns:
            decimals = int(spec.split('.')[1].rstrip('f'))
            rounded[col] = rounded[col].round(decimals)
    rounded.to_csv(path, date_format='%Y-%m-%dT%H:%M:%S')
