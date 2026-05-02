"""
l1_plot.py
----------
Plotting utilities for MIDLResult objects.

Usage:
    from midl_pipeline import midl, plot_day, plot_variable

    result = midl('2024-05-09', '2024-05-11')
    plot_day(result, '2024-05-10')
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

SAT_COLORS = {
    'ace': ('#1f77b4', 'ACE'),
    'dscovr': ('#ff7f0e', 'DSCOVR'),
    'wind': ('#2ca02c', 'WIND'),
    'solar1': ('#9467bd', 'SOLAR-1'),
}

VAR_LABELS = {
    'Bx': ('Bx', 'nT'),
    'By': ('By', 'nT'),
    'Bz': ('Bz', 'nT'),
    'Ux': ('Vx', 'km/s'),
    'Uy': ('Vy', 'km/s'),
    'Uz': ('Vz', 'km/s'),
    'rho': ('n', 'cm^-3'),
    'T': ('T', 'K'),
}

VARIABLES = list(VAR_LABELS.keys())

COMBINED_STYLE = {'color': '#222', 'ls': '-', 'lw': 1.0}
PROP_STYLES = {
    14: {'color': '#d62728', 'ls': ':', 'lw': 0.9, 'label': '14 Re'},
    32: {'color': '#1f77b4', 'ls': ':', 'lw': 0.9, 'label': '32 Re'},
}


def _fmt_xaxis(ax):
    """Format x-axis as HH:MM for a single-day plot."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))


def _set_shared_ylim(axes, series_list, log_scale=False):
    """Set identical y-axis limits across axes."""
    all_vals = pd.concat([s.dropna() for s in series_list if not s.empty],
                         ignore_index=True)
    if all_vals.empty:
        return
    if log_scale:
        all_vals = all_vals[all_vals > 0]
        if all_vals.empty:
            return
        lo, hi = all_vals.min(), all_vals.max()
        for ax in axes:
            ax.set_ylim(lo * 0.9, hi * 1.1)
            ax.set_yscale('log')
    else:
        lo, hi = all_vals.min(), all_vals.max()
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        for ax in axes:
            ax.set_ylim(lo - pad, hi + pad)


def _slice_day(df, day_str):
    """Extract one calendar day from a DataFrame."""
    day_start = pd.Timestamp(day_str)
    day_end = day_start + pd.Timedelta(days=1)
    return df.loc[(df.index >= day_start) & (df.index < day_end)]


def _load_day_from_csv(data_dir, day_str):
    """Load one day's data from monthly CSV files.

    Reads from data_dir/YYYY/MM/YYYYMM_{L1,14Re,32Re}.csv

    Returns (df_unpropagated, {b_re: df_propagated}).
    """
    dt = pd.Timestamp(day_str)
    ym_prefix = f'{dt.year:04d}{dt.month:02d}'
    month_dir = os.path.join(data_dir, f'{dt.year:04d}', f'{dt.month:02d}')

    df_comb = pd.DataFrame()
    prop_dfs = {}

    for fname in ('L1', '14Re', '32Re'):
        path = os.path.join(month_dir, f'{ym_prefix}_{fname}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df = _slice_day(df, day_str)

        if fname == 'L1':
            df_comb = df
        else:
            b_re = int(fname.replace('Re', ''))
            prop_dfs[b_re] = df

    return df_comb, prop_dfs


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_day(result, day_str, output_dir='plots'):
    """Plot one day from a MIDLResult: 8 rows (variables) x 3 columns.

    Columns: Combined (unpropagated), 14 Re, 32 Re.

    Parameters
    ----------
    result : MIDLResult
    day_str : str
        Day to plot, e.g. '2024-05-10'.
    output_dir : str
        Directory for output PNG.
    """
    df_comb = _slice_day(result.unpropagated, day_str)
    prop_dfs = {b: _slice_day(df, day_str) for b, df in result.propagated.items()}

    n_rows = len(VARIABLES)
    n_cols = 1 + len(prop_dfs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 2.2 * n_rows),
                             sharex=True)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    col_titles = ['Combined (unpropagated)'] + [
        f'Propagated to {b} Re' for b in sorted(prop_dfs.keys())]

    for row_idx, var in enumerate(VARIABLES):
        label, unit = VAR_LABELS[var]
        log_scale = (var == 'T')
        series_list = []

        # Column 0: combined.
        ax = axes[row_idx, 0]
        if var in df_comb.columns:
            ax.plot(df_comb.index.to_numpy(), df_comb[var].to_numpy(), **COMBINED_STYLE)
            series_list.append(df_comb[var])
        ax.set_ylabel(f'{label} ({unit})', fontsize=8)

        # Propagated columns.
        for col_idx, b_re in enumerate(sorted(prop_dfs.keys()), start=1):
            ax = axes[row_idx, col_idx]
            df_p = prop_dfs[b_re]
            style = PROP_STYLES.get(b_re, {'color': 'gray', 'ls': '-', 'lw': 0.9})
            if var in df_p.columns:
                ax.plot(df_p.index.to_numpy(), df_p[var].to_numpy(), color=style['color'],
                        ls='-', lw=style['lw'])
                series_list.append(df_p[var])

        # Shared y limits and formatting.
        row_axes = [axes[row_idx, c] for c in range(n_cols)]
        if series_list:
            _set_shared_ylim(row_axes, series_list, log_scale=log_scale)

        for ax in row_axes:
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)
            if not log_scale and var != 'T':
                ax.axhline(0, color='k', lw=0.3, alpha=0.3)

    # Column titles.
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9)

    # Format x-axis on bottom row.
    for col_idx in range(n_cols):
        _fmt_xaxis(axes[-1, col_idx])

    fig.suptitle(day_str, fontsize=12, y=1.0)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir,
                            day_str.replace('-', '_') + '.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_variable(result, var, day_str, output_dir='plots'):
    """Plot a single variable across all products for one day.

    Parameters
    ----------
    result : MIDLResult
    var : str
        Variable name (e.g. 'Bz', 'Ux', 'rho').
    day_str : str
    output_dir : str
    """
    df_comb = _slice_day(result.unpropagated, day_str)

    fig, ax = plt.subplots(figsize=(10, 3))
    label, unit = VAR_LABELS.get(var, (var, ''))

    if var in df_comb.columns:
        ax.plot(df_comb.index.to_numpy(), df_comb[var].to_numpy(), label='Combined',
                **COMBINED_STYLE)

    for b_re in sorted(result.propagated.keys()):
        df_p = _slice_day(result.propagated[b_re], day_str)
        style = PROP_STYLES.get(b_re, {'color': 'gray', 'ls': '-', 'lw': 0.9})
        if var in df_p.columns:
            ax.plot(df_p.index.to_numpy(), df_p[var].to_numpy(),
                    label=style.get('label', f'{b_re} Re'),
                    color=style['color'], ls=style['ls'], lw=style['lw'])

    ax.set_ylabel(f'{label} ({unit})')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    _fmt_xaxis(ax)
    if var == 'T':
        ax.set_yscale('log')

    fig.suptitle(f'{var} - {day_str}', fontsize=11)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir,
                            f'{day_str.replace("-", "_")}_{var}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_day_from_csv(data_dir, day_str, output_dir='plots',
                      raw_dir='L1_raw'):
    """Plot one day: left column = per-satellite raw, right column = combined.

    Parameters
    ----------
    data_dir : str
        Path to data directory (e.g. 'data') containing YYYY/MM/ files.
    day_str : str
        Day to plot, e.g. '2024-05-10'.
    output_dir : str
        Directory for output PNG.
    raw_dir : str
        Path to L1_raw directory with per-satellite .dat files.
    """
    from .l1_readers import read_l1_data

    df_comb, _ = _load_day_from_csv(data_dir, day_str)

    # Load per-satellite raw data for this day.
    dt = pd.Timestamp(day_str)
    day_path = os.path.join(raw_dir, dt.strftime('%Y/%m/%d'))
    sat_dfs = {}
    for sat in SAT_COLORS:
        fpath = os.path.join(day_path, f'L1_{sat}.dat')
        df = read_l1_data(fpath)
        if not df.empty:
            sat_dfs[sat] = _slice_day(df, day_str)

    if df_comb.empty and not sat_dfs:
        print(f'  No data for {day_str}, skipping plot.')
        return

    n_rows = len(VARIABLES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.2 * n_rows),
                             sharex=True)

    for row_idx, var in enumerate(VARIABLES):
        label, unit = VAR_LABELS[var]
        log_scale = (var == 'T')
        series_list = []

        # Left column: per-satellite raw data (overlaid).
        ax_left = axes[row_idx, 0]
        for sat in SAT_COLORS:
            if sat in sat_dfs and var in sat_dfs[sat].columns:
                color, sat_label = SAT_COLORS[sat]
                df_s = sat_dfs[sat]
                ax_left.plot(df_s.index.to_numpy(), df_s[var].to_numpy(),
                             color=color, lw=0.6, alpha=0.8)
                series_list.append(df_s[var])
        ax_left.set_ylabel(f'{label} ({unit})', fontsize=8)

        # Right column: MIDL combined.
        ax_right = axes[row_idx, 1]
        if not df_comb.empty and var in df_comb.columns:
            ax_right.plot(df_comb.index.to_numpy(), df_comb[var].to_numpy(),
                          **COMBINED_STYLE)
            series_list.append(df_comb[var])

        # Shared y limits and formatting.
        if series_list:
            _set_shared_ylim([ax_left, ax_right], series_list,
                             log_scale=log_scale)

        for ax in (ax_left, ax_right):
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)
            if not log_scale and var != 'T':
                ax.axhline(0, color='k', lw=0.3, alpha=0.3)

    # Column titles.
    sat_title = ' / '.join(v[1] for v in SAT_COLORS.values())
    axes[0, 0].set_title(sat_title, fontsize=9)
    axes[0, 1].set_title('MIDL Combined', fontsize=9)

    # Legend on top-left panel.
    legend_handles = [Line2D([0], [0], color=SAT_COLORS[s][0], lw=1.0,
                             label=SAT_COLORS[s][1])
                      for s in SAT_COLORS]
    axes[0, 0].legend(handles=legend_handles, fontsize=7, loc='upper right',
                      ncol=3)

    # Format x-axis on bottom row.
    for col_idx in range(2):
        _fmt_xaxis(axes[-1, col_idx])

    fig.suptitle(day_str, fontsize=12, y=1.0)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir,
                            day_str.replace('-', '_') + '.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')
