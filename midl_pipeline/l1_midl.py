"""
l1_midl.py
----------
Main entry point for the MIDL pipeline.

Processes an arbitrary date range of L1 solar wind data from L1_raw/
into merged, quality-screened, propagated output.

Public API: midl(start, end) -> MIDLResult
"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .l1_combine import combine_data_priority, combine_temperature
from .l1_filters import despike, interpolate_with_limits, smooth_transitions, INTERP_LIMITS
from .l1_propagation import ballistic_propagation
from .l1_readers import read_l1_data


_NUMERIC_COLS = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']

SATELLITES = ('ace', 'dscovr', 'wind', 'solar1')
# When IMAP data becomes available, add 'imap' here.


@dataclass
class MIDLResult:
    """Return value from midl().

    Attributes
    ----------
    unpropagated : pd.DataFrame
        Combined data at reference satellite position.
        Columns: Bx, By, Bz, Ux, Uy, Uz, rho, T.
        Index: DatetimeIndex at 1-minute cadence.
    propagated : dict[int, pd.DataFrame]
        Ballistically propagated data keyed by boundary distance in Re.
        Default keys: 14, 32. Same columns as unpropagated.
    ref_x_re : dict[datetime.date, float]
        X_GSM position (in Earth radii) of the reference satellite for each
        calendar day.  The reference satellite is the one closest to Earth.
    source_map : dict[str, pd.Series]
        Per-variable source provenance. Each Series contains frozenset of
        satellite codes (1=ACE, 2=DSCOVR, 3=WIND, 4=SOLAR-1) at each
        minute.  Code 5 reserved for IMAP.
        Keys: Bx, By, Bz, Ux, Uy, Uz, rho, T.
    mhd_profile : xr.Dataset or None
        1D MHD-propagated solar wind profile produced by BATSRUS when
        'mhd' is enabled in the `propagation` kwarg of midl().  Has dims
        (time, x) with x spanning roughly 31..235 Re (native BATSRUS
        grid), data vars Bx/By/Bz/Ux/Uy/Uz/rho/T, plus a
        No NaN masking — BATSRUS output is kept everywhere.  None when
        MHD is disabled.
    """
    unpropagated: pd.DataFrame
    propagated: dict
    ref_x_re: dict
    source_map: dict
    mhd_profile: "object" = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day_range(start, end):
    """Yield 'YYYY-MM-DD' strings for each day in [start, end] inclusive."""
    current = start
    while current <= end:
        yield current.strftime('%Y-%m-%d')
        current += timedelta(days=1)


def _load_raw_range(raw_dir, start, end):
    """Load L1_raw .dat files for [start, end] into per-satellite DataFrames.

    Returns dict: {'ace': df, 'dscovr': df, 'wind': df}.
    Satellites with no data are omitted.
    """
    data_map = {}
    for sat in SATELLITES:
        frames = []
        for day_str in _day_range(start, end):
            dt = datetime.strptime(day_str, '%Y-%m-%d')
            path = os.path.join(raw_dir, dt.strftime('%Y/%m/%d'),
                                f'L1_{sat}.dat')
            df = read_l1_data(path)
            if not df.empty:
                frames.append(df[_NUMERIC_COLS])
        if frames:
            combined = pd.concat(frames).sort_index()
            data_map[sat] = combined[~combined.index.duplicated(keep='first')]
    return data_map


def _read_sat_positions(pos_file):
    """Read per-satellite noon X positions in km from L1_satpos.dat."""
    result = {'ace': np.nan, 'dscovr': np.nan, 'wind': np.nan, 'solar1': np.nan}
    if not os.path.exists(pos_file):
        return result
    try:
        with open(pos_file, 'r', encoding='utf-8') as f:
            data_started = False
            for line in f:
                if line.strip().startswith('#START'):
                    data_started = True
                    continue
                if not data_started:
                    continue
                parts = line.split()
                if len(parts) >= 15:
                    result['ace']    = float(parts[6])  * 6371.0
                    result['dscovr'] = float(parts[9])  * 6371.0
                    result['wind']   = float(parts[12]) * 6371.0
                    if len(parts) >= 18:
                        result['solar1'] = float(parts[15]) * 6371.0
                    break
    except Exception as e:
        print(f'  Warning: Could not read position file ({e}).')
    return result


def _load_positions_range(raw_dir, start, end):
    """Load satellite positions for [start, end] into a dict indexed by date.

    Reads L1_satpos.dat from the same raw_dir tree as the satellite data.
    Returns dict: {date -> {'ace': x_km, 'dscovr': x_km, 'wind': x_km}}.
    """
    positions = {}
    for day_str in _day_range(start, end):
        dt = datetime.strptime(day_str, '%Y-%m-%d')
        pos_file = os.path.join(raw_dir, dt.strftime('%Y/%m/%d'),
                                'L1_satpos.dat')
        positions[dt.date()] = _read_sat_positions(pos_file)
    return positions


def _propagate_to_reference(data_map, positions):
    """Shift satellites to daily reference position (closest to Earth).

    Modifies data_map in place. Returns a dict {date -> x_ref_km} for use
    in final propagation to boundary.
    """
    ref_x_daily = {}

    # Collect all dates that have data.
    all_dates = set()
    for sat_df in data_map.values():
        all_dates.update(sat_df.index.date)
    all_dates = sorted(all_dates)

    # Forward-fill positions for days with missing satpos files.
    last_good_pos = None

    for date in all_dates:
        pos = positions.get(date)
        if pos is None:
            pos = last_good_pos if last_good_pos else {
                sat: np.nan for sat in SATELLITES}
        if any(np.isfinite(v) for v in pos.values()):
            last_good_pos = pos

        available_x = {sat: pos[sat] for sat in data_map
                       if np.isfinite(pos.get(sat, np.nan))}

        if not available_x:
            ref_x_daily[date] = 1.5e6
            continue

        ref_sat = min(available_x, key=lambda s: available_x[s])
        x_ref_km = available_x[ref_sat]
        ref_x_daily[date] = x_ref_km

        # Shift non-reference satellites for this day's data.
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)

        for sat in list(data_map.keys()):
            x_sat = available_x.get(sat, np.nan)
            if not np.isfinite(x_sat) or x_sat <= x_ref_km:
                continue

            day_mask = ((data_map[sat].index >= day_start) &
                        (data_map[sat].index < day_end))
            if not day_mask.any():
                continue

            df_day = data_map[sat].loc[day_mask].copy()

            # If Ux is entirely NaN, borrow from another satellite
            # (or use -400 km/s default) so B data isn't lost.
            ux_was_all_nan = df_day['Ux'].isna().all()
            if ux_was_all_nan:
                donor_ux = None
                for other in data_map:
                    if other == sat:
                        continue
                    other_day = data_map[other].loc[
                        (data_map[other].index >= day_start) &
                        (data_map[other].index < day_end), 'Ux']
                    if other_day.notna().any():
                        donor_ux = other_day.reindex(df_day.index).interpolate(
                            method='time')
                        break
                if donor_ux is not None:
                    df_day['Ux'] = donor_ux
                else:
                    df_day['Ux'] = -400.0

            df_day = df_day.rename(
                columns={'Ux': 'Vx Velocity, km/s, GSE'})
            orbit = pd.Series({'X_GSE': x_sat})
            df_prop = ballistic_propagation(
                orbit, df_day, target_x_km=x_ref_km)
            df_prop = df_prop.rename(
                columns={'Vx Velocity, km/s, GSE': 'Ux'})

            # Donor Ux was only for timing — erase it from output.
            if ux_was_all_nan:
                df_prop['Ux'] = np.nan

            # Replace this day's slice in the full DataFrame.
            data_map[sat] = pd.concat([
                data_map[sat].loc[~day_mask],
                df_prop
            ]).sort_index()

    return ref_x_daily


def _compute_source_changed(source_map):
    """Build per-column boolean mask: True where satellite source changed."""
    source_changed = {}
    for col, src in source_map.items():
        vals = src.values
        changed = np.zeros(len(vals), dtype=bool)
        for k in range(1, len(vals)):
            if vals[k] is not None and vals[k - 1] is not None:
                changed[k] = vals[k] != vals[k - 1]
        source_changed[col] = pd.Series(changed, index=src.index)
    return source_changed


def _propagate_to_boundary(df_combined, ref_x_daily, target_km):
    """Propagate combined data to a fixed boundary using per-day reference X.

    Each day is propagated with a 3-hour pad from the previous day so that
    the ballistic time-shift doesn't leave NaN gaps at the start of each day.
    """
    _PAD = pd.Timedelta(hours=3)
    all_dates = sorted(set(df_combined.index.date))
    frames = []

    for date in all_dates:
        x_ref = ref_x_daily.get(date, 1.5e6)
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)

        # Include a pad before the day so interpolation has context.
        pad_start = day_start - _PAD
        pad_mask = ((df_combined.index >= pad_start) &
                    (df_combined.index < day_end))
        df_padded = df_combined.loc[pad_mask].copy()

        if df_padded.empty or df_padded['Ux'].isna().all():
            day_mask = ((df_combined.index >= day_start) &
                        (df_combined.index < day_end))
            frames.append(df_combined.loc[day_mask].copy())
            continue

        df_padded = df_padded.rename(
            columns={'Ux': 'Vx Velocity, km/s, GSE'})
        orbit = pd.Series({'X_GSE': x_ref})
        df_prop = ballistic_propagation(
            orbit, df_padded, target_x_km=target_km)
        df_prop = df_prop.rename(
            columns={'Vx Velocity, km/s, GSE': 'Ux'})

        # Slice back to just the target day.
        day_mask = ((df_prop.index >= day_start) &
                    (df_prop.index < day_end))
        frames.append(df_prop.loc[day_mask])

    if not frames:
        return df_combined.copy()
    return pd.concat(frames).sort_index()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def midl(start, end, raw_dir='L1_raw', boundaries_re=(14, 32),
         propagation=('ballistic',), batsrus_dir=None, mhd_work_dir=None):
    """Process L1 solar wind data for [start, end].

    Reads raw satellite data and position files from raw_dir/, applies
    the full pipeline (despike, interpolate, propagate-to-reference,
    quality-score, source-select, combine, smooth, propagate-to-boundary),
    and returns
    a MIDLResult.

    Parameters
    ----------
    start, end : str or pd.Timestamp
        Date range to process (inclusive), e.g. '2024-05-09', '2024-05-11'.
    raw_dir : str
        Path to directory tree containing per-day satellite data and
        L1_satpos.dat files (raw_dir/YYYY/MM/DD/).
    boundaries_re : tuple of int
        Propagation target distances in Earth radii. Default (14, 32).
    propagation : tuple of str
        Propagation methods to run.  'ballistic' runs the existing
        per-boundary ballistic time-shift.  'mhd' runs BATSRUS 1D and
        stores the full profile in MIDLResult.mhd_profile.  Defaults to
        ('ballistic',) for backwards compatibility.
    batsrus_dir : str or None
        Path to the built BATSRUS install used by the 'mhd' method.
        Defaults to `MIDL-Pipeline/BATSRUS` resolved relative to the
        midl_pipeline package.  Ignored when 'mhd' not in `propagation`.
    mhd_work_dir : str or None
        Scratch directory for the BATSRUS run.  A fresh tempdir is
        allocated if None.  Ignored when 'mhd' not in `propagation`.

    Returns
    -------
    MIDLResult
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Pad by 1 day for boundary context (replaces old 3-day window).
    load_start = (start - pd.Timedelta(days=1)).normalize()
    load_end = (end + pd.Timedelta(days=1)).normalize()

    # Stage 0: Load raw data.
    print(f'Loading L1_raw data for {load_start.date()} to {load_end.date()}...')
    data_map = _load_raw_range(raw_dir, load_start, load_end)

    if not data_map:
        print('No satellite data found.')
        empty = pd.DataFrame(columns=_NUMERIC_COLS)
        return MIDLResult(
            unpropagated=empty,
            propagated={b: empty.copy() for b in boundaries_re},
            ref_x_re={},
            source_map={},
            mhd_profile=None,
        )

    # Stage 1: Despike.
    print('Despiking...')
    for sat in data_map:
        data_map[sat] = despike(data_map[sat])

    # Stage 2: Interpolate per-satellite gaps.
    print('Interpolating gaps...')
    for sat in data_map:
        data_map[sat] = interpolate_with_limits(data_map[sat], INTERP_LIMITS)

    # Stage 3: Propagate to reference position.
    print('Propagating to reference positions...')
    positions = _load_positions_range(raw_dir, load_start, load_end)
    ref_x_daily = _propagate_to_reference(data_map, positions)

    # Deduplicate after propagation — time-shifting can create collisions at
    # day boundaries. Keep the fastest parcel (most negative Ux) per minute.
    for sat in data_map:
        df = data_map[sat]
        if df.index.duplicated().any():
            df = df.sort_values('Ux', ascending=True)
            data_map[sat] = df[~df.index.duplicated(keep='first')]

    # Build master grid spanning the full padded window.
    grid_start = load_start
    grid_end = load_end + pd.Timedelta(days=1)
    n_minutes = int((grid_end - grid_start).total_seconds() / 60)
    master_grid = pd.date_range(start=grid_start, periods=n_minutes,
                                freq='1min')

    # Stage 4: Quality score + source select.
    print('Running quality scoring and source selection...')
    df_combined, source_map = combine_data_priority(
        data_map, master_grid)

    print('Combining temperature...')
    df_combined['T'], t_source = combine_temperature(data_map, master_grid)
    source_map['T'] = t_source

    # Stage 5: Smooth transitions.
    print('Smoothing transitions...')
    source_changed = _compute_source_changed(source_map)
    df_combined = smooth_transitions(
        df_combined, source_changed=source_changed)

    # Stage 6: Propagate to boundaries.
    propagated = {}
    for b_re in boundaries_re:
        target_km = b_re * 6371.0
        print(f'Propagating to {b_re} Re ({target_km:.0f} km)...')
        propagated[b_re] = _propagate_to_boundary(
            df_combined, ref_x_daily, target_km)
        propagated[b_re] = interpolate_with_limits(
            propagated[b_re], INTERP_LIMITS)

    # Stage 6b: 1D MHD propagation (optional).
    # Uses a restart loop: if BATSRUS crashes mid-run, recover whatever
    # plot files were written, skip past the crashing minute, and relaunch
    # on the remaining tail.  Segments are concatenated at the end.
    mhd_profile = None
    if 'mhd' in propagation:
        print('Running 1D MHD propagation (BATSRUS)...')
        from .l1_mhd import mhd_propagation
        import xarray as xr

        _MAX_RESTART = 10
        _RESTART_SKIP = pd.Timedelta(minutes=2)
        _MIN_REMAINING = pd.Timedelta(hours=2)

        segments = []
        crash_infos = []
        remaining_start = df_combined.index[0]
        tail_end = df_combined.index[-1]

        for attempt in range(_MAX_RESTART):
            if remaining_start > tail_end:
                break
            if (tail_end - remaining_start) < _MIN_REMAINING:
                break

            sub = df_combined.loc[remaining_start:]
            ref_sub = {d: v for d, v in ref_x_daily.items()
                       if d >= remaining_start.date()}
            if not ref_sub:
                ref_sub = ref_x_daily

            try:
                ds_seg = mhd_propagation(
                    sub, ref_sub,
                    work_dir=mhd_work_dir, batsrus_dir=batsrus_dir,
                    allow_partial=True)
            except RuntimeError as e:
                crash_infos.append(f'unrecoverable: {e}')
                print(f'  MHD unrecoverable crash: {e}')
                break

            crashed = bool(ds_seg.attrs.get('batsrus_crashed', 0))
            if crashed:
                info = ds_seg.attrs.get('batsrus_crash_info', '')
                crash_infos.append(info)
                print(f'  MHD crash at attempt {attempt+1}, '
                      f'recovered {len(ds_seg.time)} minutes')

            if len(ds_seg.time) > 0:
                segments.append(ds_seg)
                t_last = pd.Timestamp(ds_seg.time.values[-1])
            else:
                break

            if not crashed:
                break

            remaining_start = t_last + _RESTART_SKIP

        if segments:
            if len(segments) == 1:
                mhd_profile = segments[0]
            else:
                mhd_profile = xr.concat(segments, dim='time')
                _, uniq_idx = np.unique(
                    mhd_profile.time.values, return_index=True)
                mhd_profile = mhd_profile.isel(time=np.sort(uniq_idx))

            # Reindex onto full 1-min grid so gaps appear as NaN.
            full_index = pd.date_range(
                df_combined.index[0], df_combined.index[-1], freq='1min')
            mhd_profile = mhd_profile.reindex(time=full_index)

            if crash_infos:
                print(f'  MHD completed with {len(crash_infos)} crash(es)')
        else:
            print('  WARNING: MHD produced no recoverable output.')

    # Stage 7: Slice to requested range and return.
    result_start = start.normalize()
    result_end = (end + pd.Timedelta(days=1)).normalize()
    mask = ((df_combined.index >= result_start) &
            (df_combined.index < result_end))

    result_propagated = {}
    for b_re, df_prop in propagated.items():
        prop_mask = ((df_prop.index >= result_start) &
                     (df_prop.index < result_end))
        result_propagated[b_re] = df_prop.loc[prop_mask].copy()

    # Convert reference positions from km back to Re.
    ref_x_re = {date: x_km / 6371.0 for date, x_km in ref_x_daily.items()}

    # Slice source_map to requested range.
    result_source_map = {}
    for col, src in source_map.items():
        src_mask = ((src.index >= result_start) & (src.index < result_end))
        result_source_map[col] = src.loc[src_mask].copy()

    # Slice MHD profile to requested range (same window as ballistic).
    if mhd_profile is not None:
        mhd_profile = mhd_profile.sel(
            time=slice(result_start, result_end - pd.Timedelta(minutes=1)))

    print('Done.')
    return MIDLResult(
        unpropagated=df_combined.loc[mask].copy(),
        propagated=result_propagated,
        ref_x_re=ref_x_re,
        source_map=result_source_map,
        mhd_profile=mhd_profile,
    )
