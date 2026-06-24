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
from .l1_filters import (despike, interpolate_with_limits, smooth_transitions,
                         drop_isolated_minutes, bracketing_or_fill, INTERP_LIMITS)
from .l1_propagation import ballistic_propagation
from .l1_readers import read_l1_data


_NUMERIC_COLS = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']

# Suffix for per-satellite interpolation-provenance companion columns.
# These ride through propagation as float 0.0/1.0 and are binarized back.
_INTERP_SUFFIX = '_interp'
_INTERP_VARS = _NUMERIC_COLS  # one flag column per numeric variable

# Plasma variables for which single-minute gap fills are NOT flagged as
# interpolated (the fill sits below the native plasma cadence and under the
# despike median-filter resolution).  The magnetic field (Bx/By/Bz) keeps
# single-minute fills flagged, since B varies fast and is measured at high
# cadence.
_PLASMA_INTERP_VARS = {'Ux', 'Uy', 'Uz', 'rho', 'T'}

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
    interp_flags : dict[str, pd.Series] or None
        Interpolation-provenance flags for the unpropagated (L1) product,
        keyed by output group ('B', 'Ux', 'Uyz', 'rho', 'T').  Each Series
        is integer-valued on the L1 index:
        0 = all contributing values are direct observations;
        1 = mixed — at least one contributing satellite value was produced
            by gap interpolation AND at least one is direct;
        2 = ALL contributing values were produced by interpolation.
        Single-minute plasma fills are not flagged (treated as direct); the
        magnetic field keeps single-minute fills flagged.
    propagated_interp_flags : dict[int, dict[str, pd.Series]] or None
        Same 0/1/2 flag groups for each propagated boundary, keyed by boundary
        Re then group.  A minute filled by the post-propagation pass inherits
        the worse (max) of the two bracketing minutes' levels (a gap between
        two direct minutes is therefore flagged direct, 0), so no separate
        "post-propagation fill" level is emitted.
    """
    unpropagated: pd.DataFrame
    propagated: dict
    ref_x_re: dict
    source_map: dict
    mhd_profile: "object" = None
    interp_flags: dict = None
    propagated_interp_flags: dict = None


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


_MIN_L1_X_RE = 190.0


def _read_sat_positions(pos_file):
    """Read per-satellite noon X positions in km from L1_satpos.dat.

    Satellites with X_GSM < 190 Re are treated as not at L1 (e.g. WIND
    in the magnetotail pre-2004, DSCOVR in transit Feb 2015) and their
    position is set to NaN so they are excluded from propagation.
    """
    result = {'ace': np.nan, 'dscovr': np.nan, 'wind': np.nan, 'solar1': np.nan}
    if not os.path.exists(pos_file):
        return result, set()
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

    gated = set()
    for sat in result:
        x_re = result[sat] / 6371.0
        if np.isfinite(x_re) and x_re < _MIN_L1_X_RE:
            result[sat] = np.nan
            gated.add(sat)

    return result, gated


def _load_positions_range(raw_dir, start, end):
    """Load satellite positions for [start, end] into a dict indexed by date.

    Reads L1_satpos.dat from the same raw_dir tree as the satellite data.
    Returns (positions, gated_by_day) where:
        positions: {date -> {'ace': x_km, 'dscovr': x_km, 'wind': x_km}}
        gated_by_day: {date -> set of satellite names gated by L1 threshold}
    """
    positions = {}
    gated_by_day = {}
    for day_str in _day_range(start, end):
        dt = datetime.strptime(day_str, '%Y-%m-%d')
        pos_file = os.path.join(raw_dir, dt.strftime('%Y/%m/%d'),
                                'L1_satpos.dat')
        pos, gated = _read_sat_positions(pos_file)
        positions[dt.date()] = pos
        gated_by_day[dt.date()] = gated
    return positions, gated_by_day


def _propagate_to_reference(data_map, positions, gated_by_day=None):
    """Shift satellites to daily reference position (closest to Earth).

    Modifies data_map in place. Returns a dict {date -> x_ref_km} for use
    in final propagation to boundary.
    """
    if gated_by_day is None:
        gated_by_day = {}
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
        if pos is None or not any(np.isfinite(v) for v in pos.values()):
            pos = last_good_pos if last_good_pos else {
                sat: np.nan for sat in SATELLITES}
        if any(np.isfinite(v) for v in pos.values()):
            last_good_pos = pos

        available_x = {sat: pos[sat] for sat in data_map
                       if np.isfinite(pos.get(sat, np.nan))}

        # Drop data only for satellites explicitly gated (X < 190 Re),
        # not for satellites with NaN positions from missing satpos data.
        gated_today = gated_by_day.get(date, set())
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)
        for sat in list(data_map.keys()):
            if sat in gated_today:
                day_mask = ((data_map[sat].index >= day_start) &
                            (data_map[sat].index < day_end))
                if day_mask.any():
                    data_map[sat] = data_map[sat].loc[~day_mask]

        if not available_x:
            ref_x_daily[date] = 1.5e6
            continue

        ref_sat = min(available_x, key=lambda s: available_x[s])
        x_ref_km = available_x[ref_sat]
        ref_x_daily[date] = x_ref_km

        # Shift non-reference satellites for this day's data.

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


# Output flag groups -> the per-satellite variable flags that feed them.
# Mirrors the five source groups (B, Ux, Uyz, rho, T).
_FLAG_GROUPS = {
    'B':   ('Bx', 'By', 'Bz'),
    'Ux':  ('Ux',),
    'Uyz': ('Uy', 'Uz'),
    'rho': ('rho',),
    'T':   ('T',),
}

# source_map keys that carry the contributing-satellite frozensets for each
# output flag group.
_FLAG_GROUP_SOURCE_KEY = {
    'B':   'Bx',    # Bx/By/Bz share the same source decision
    'Ux':  'Ux',
    'Uyz': 'Uy',    # Uy/Uz share the same source decision
    'rho': 'rho',
    'T':   'T',
}


def _build_stage2_flags(data_map, source_map, master_grid):
    """Merge per-satellite Stage-2 interp flags into per-group output levels.

    For each output flag group and each minute, counts how many of the
    *contributing* satellites (source_map frozenset) had an interpolated
    value for that group at that minute:

        0 — all contributing values are direct observations
        1 — mixed: >=1 contributing value interpolated AND >=1 direct
        2 — ALL contributing values interpolated

    A satellite's group value counts as interpolated when any of its
    component variables in the group was gap-filled (e.g. B: Bx|By|Bz).
    Returns dict[group -> pd.Series of int (0/1/2)] on master_grid.

    Purely diagnostic — does not affect any numeric output column.
    """
    from .l1_combine import SAT_CODE
    code_to_sat = {v: k for k, v in SAT_CODE.items()}

    # Per-satellite, per-variable flag series aligned to the master grid.
    # Missing satellites / columns default to all-False.
    sat_flag = {}
    for sat in SAT_CODE:
        df = data_map.get(sat)
        sat_flag[sat] = {}
        for var in _INTERP_VARS:
            col = f'{var}{_INTERP_SUFFIX}'
            if df is not None and col in df.columns:
                s = df[col].reindex(master_grid)
                sat_flag[sat][var] = s.fillna(False).astype(bool)
            else:
                sat_flag[sat][var] = pd.Series(False, index=master_grid)

    group_flags = {}
    for group, vars_in_group in _FLAG_GROUPS.items():
        src_key = _FLAG_GROUP_SOURCE_KEY[group]
        src = source_map.get(src_key)
        out_vals = np.zeros(len(master_grid), dtype=np.int64)
        if src is None:
            group_flags[group] = pd.Series(out_vals, index=master_grid)
            continue
        src = src.reindex(master_grid)
        src_vals = src.values
        # Pre-extract per-sat, per-var boolean arrays for speed.
        arrs = {sat: {v: sat_flag[sat][v].values for v in vars_in_group}
                for sat in SAT_CODE}
        for i in range(len(master_grid)):
            codes = src_vals[i]
            if codes is None:
                continue
            n_contrib = 0
            n_flagged = 0
            for c in codes:
                sat = code_to_sat.get(c)
                if sat is None:
                    continue
                n_contrib += 1
                for v in vars_in_group:
                    if arrs[sat][v][i]:
                        n_flagged += 1
                        break
            if n_contrib == 0 or n_flagged == 0:
                continue            # 0: all direct (or nothing to assess)
            elif n_flagged == n_contrib:
                out_vals[i] = 2     # all contributing values interpolated
            else:
                out_vals[i] = 1     # mixed: interpolated + direct
        group_flags[group] = pd.Series(out_vals, index=master_grid)

    return group_flags


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

    # Stage 1: Despike.  Capture the median filter's NaN->finite bridges (it
    # reconstructs isolated single-minute dropouts), so they count as
    # interpolated, not direct, when the provenance flag is built in Stage 2.
    print('Despiking...')
    despike_bridge = {}
    for sat in data_map:
        data_map[sat], despike_bridge[sat] = despike(
            data_map[sat], return_mask=True)

    # Stage 2: Interpolate per-satellite gaps.
    #
    # Provenance flags (additive only — does not change numeric output):
    # interpolate_with_limits returns a boolean mask of which cells were
    # gap-filled.  We attach those masks as companion `<var>_interp` columns
    # on each satellite's frame so they ride through the ballistic time-shift
    # with their parent rows and can be merged with the same source selection
    # as the data in Stage 4.
    print('Interpolating gaps...')
    for sat in data_map:
        filled, mask = interpolate_with_limits(
            data_map[sat], INTERP_LIMITS, return_mask=True)
        bridge = despike_bridge[sat]
        for col in mask.columns:
            # A cell is interpolated if Stage-1 despike bridged it OR Stage-2
            # gap-fill filled it (T has no despike bridge).  Store as float
            # 0.0/1.0 so the flags survive the numeric-only regridding inside
            # ballistic_propagation (which drops non-numeric columns).
            # Binarized back to bool after propagation.
            cell = mask[col]
            if col in bridge.columns:
                cell = cell | bridge[col]
            # Plasma: a single isolated minute fill is not flagged (below the
            # native plasma cadence / median-filter resolution).  B keeps it.
            if col in _PLASMA_INTERP_VARS:
                cell = drop_isolated_minutes(cell)
            filled[f'{col}{_INTERP_SUFFIX}'] = cell.astype(float)
        data_map[sat] = filled

    # Stage 3: Propagate to reference position.
    print('Propagating to reference positions...')
    positions, gated_by_day = _load_positions_range(raw_dir, load_start, load_end)
    ref_x_daily = _propagate_to_reference(data_map, positions, gated_by_day)

    # Deduplicate after propagation — time-shifting can create collisions at
    # day boundaries. Keep the fastest parcel (most negative Ux) per minute.
    for sat in data_map:
        df = data_map[sat]
        if df.index.duplicated().any():
            df = df.sort_values('Ux', ascending=True)
            data_map[sat] = df[~df.index.duplicated(keep='first')]

    # Binarize the interpolation-flag companion columns.  The ballistic
    # regridding interpolates numerically (reindex + interpolate(limit=2)
    # jitter snapping), so a 0/1 flag can land as a fractional value at a
    # snapped minute.  Treat any positive fraction as "flagged" (a flagged
    # parent row contributed to this minute) — conservative by design.
    for sat in data_map:
        df = data_map[sat]
        flag_cols = [c for c in df.columns if c.endswith(_INTERP_SUFFIX)]
        for c in flag_cols:
            df[c] = (df[c].fillna(0.0) > 0.0)
        data_map[sat] = df

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
    # Per-satellite Stage-2 T interp flags, for the merged T provenance flag.
    t_interp_flags = {}
    t_flag_col = f'T{_INTERP_SUFFIX}'
    for sat in data_map:
        if t_flag_col in data_map[sat].columns:
            t_interp_flags[sat] = data_map[sat][t_flag_col]
    df_combined['T'], t_source, t_interp_flag = combine_temperature(
        data_map, master_grid, t_interp_flags=t_interp_flags, return_flag=True)
    source_map['T'] = t_source

    # Build merged Stage-2 interp flags for the five output groups.
    # (Additive provenance only — df_combined values are unchanged.)
    stage2_flags = _build_stage2_flags(data_map, source_map, master_grid)
    stage2_flags['T'] = t_interp_flag

    # Stage 5: Smooth transitions.
    print('Smoothing transitions...')
    source_changed = _compute_source_changed(source_map)
    df_combined = smooth_transitions(
        df_combined, source_changed=source_changed)

    # Representative data column per group, used to look up the per-group
    # gap-fill limit from INTERP_LIMITS (B groups -> 5 min, plasma -> 60 min)
    # when carrying the provenance carriers across a propagation gap.
    _GROUP_REPR_COL_FOR_LIMIT = {'B': 'Bx', 'Ux': 'Ux', 'Uyz': 'Uy',
                                 'rho': 'rho', 'T': 'T'}
    # Internal float carrier suffixes for the merged Stage-2 flag levels so
    # they survive ballistic_propagation's numeric-only regridding.  The
    # 0/1/2 level cannot be carried as one number (regridding blends
    # numerically), so it is decomposed into two monotone booleans:
    #   any  = level >= 1  (some contributing value interpolated)
    #   all  = level == 2  (every contributing value interpolated)
    # After regridding, a snapped minute blends <=2 parent rows:
    #   any  := blend > 0   (any flagged parent contributed)
    #   all  := blend == 1  (every contributing parent was all-interpolated)
    _CARRY_ANY = '__interp_any'
    _CARRY_ALL = '__interp_all'

    # Attach the carriers to the combined frame as floats so they ride
    # through _propagate_to_boundary with their parent rows.
    df_combined_f = df_combined.copy()
    for group, flag in stage2_flags.items():
        lev = flag.reindex(df_combined_f.index).fillna(0).astype(int)
        df_combined_f[f'{group}{_CARRY_ANY}'] = (lev >= 1).astype(float)
        df_combined_f[f'{group}{_CARRY_ALL}'] = (lev >= 2).astype(float)

    # Per-group provenance flag series for the unpropagated (L1) product:
    # exactly the Stage-2 levels (L1 has no Stage-6 boundary pass, so no 3).
    unprop_flags = {g: stage2_flags[g].reindex(df_combined.index)
                    for g in _FLAG_GROUPS}

    # Stage 6: Propagate to boundaries.
    propagated = {}
    propagated_flags = {}
    for b_re in boundaries_re:
        target_km = b_re * 6371.0
        print(f'Propagating to {b_re} Re ({target_km:.0f} km)...')
        prop = _propagate_to_boundary(
            df_combined_f, ref_x_daily, target_km)

        prop = interpolate_with_limits(prop, INTERP_LIMITS)

        # Carry the provenance carriers across each propagation gap.  The
        # carriers are not in INTERP_LIMITS, so the fill above left them NaN
        # across every gap the data fill bridged.  A gap-filled minute should
        # inherit the WORSE (max) of the two bracketing levels: physically the
        # ballistic fill is a rarefaction/deceleration profile, not missing
        # data, so it carries the observation/fill status of its neighbors
        # rather than a distinct "fill" label.  For a monotone boolean carrier,
        # "worse of the two endpoints" is the OR of its forward- and back-fill,
        # bounded to the same per-group gap limit the data fill used.
        for g in _FLAG_GROUPS:
            limit = INTERP_LIMITS.get(_GROUP_REPR_COL_FOR_LIMIT[g])
            for suffix in (_CARRY_ANY, _CARRY_ALL):
                c = f'{g}{suffix}'
                if c not in prop.columns:
                    continue
                s = prop[c]
                inherited = bracketing_or_fill(s, limit)
                # Only overwrite the propagation-gap cells (NaN carriers);
                # leave the regridded 0.0/1.0 values untouched.
                prop[c] = s.where(s.notna(), inherited.astype(float))

        # Build the per-group boundary flags.
        b_flags = {}
        for g in _FLAG_GROUPS:
            c_any = f'{g}{_CARRY_ANY}'
            c_all = f'{g}{_CARRY_ALL}'
            # Binarize the carriers (regridding blends <=2 parent rows, and
            # propagation gaps now carry the inherited worse-of-two level):
            #   any: any positive fraction means a flagged parent row
            #        contributed (conservative toward "interpolated").
            #   all: only an exact 1.0 blend means every contributing parent
            #        was all-interpolated (a blend with any direct-backed
            #        parent is by construction mixed).
            if c_any in prop.columns:
                any_f = prop[c_any].fillna(0.0) > 0.0
            else:
                any_f = pd.Series(False, index=prop.index)
            if c_all in prop.columns:
                all_f = prop[c_all].fillna(0.0) >= 1.0 - 1e-9
            else:
                all_f = pd.Series(False, index=prop.index)
            all_f = all_f & any_f   # 'all' cannot hold without 'any'

            # Level: 0 all-direct, 1 mixed, 2 all-interpolated; blank handled
            # later where the value is NaN (gaps beyond the limit stay blank).
            flag = pd.Series(0, index=prop.index, dtype='int64')
            flag = flag.mask(any_f, 1)
            flag = flag.mask(all_f, 2)
            b_flags[g] = flag

        # Drop the internal carrier columns from the numeric output.
        prop = prop.drop(columns=[c for c in prop.columns
                                  if c.endswith((_CARRY_ANY, _CARRY_ALL))])
        propagated[b_re] = prop
        propagated_flags[b_re] = b_flags

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
    result_propagated_flags = {}
    for b_re, df_prop in propagated.items():
        prop_mask = ((df_prop.index >= result_start) &
                     (df_prop.index < result_end))
        result_propagated[b_re] = df_prop.loc[prop_mask].copy()
        result_propagated_flags[b_re] = {
            g: f.loc[(f.index >= result_start) & (f.index < result_end)].copy()
            for g, f in propagated_flags[b_re].items()
        }

    # Convert reference positions from km back to Re.
    ref_x_re = {date: x_km / 6371.0 for date, x_km in ref_x_daily.items()}

    # Slice source_map to requested range.
    result_source_map = {}
    for col, src in source_map.items():
        src_mask = ((src.index >= result_start) & (src.index < result_end))
        result_source_map[col] = src.loc[src_mask].copy()

    # Slice unpropagated (L1) provenance flags to requested range.
    result_unprop_flags = {}
    for g, f in unprop_flags.items():
        f_mask = ((f.index >= result_start) & (f.index < result_end))
        result_unprop_flags[g] = f.loc[f_mask].copy()

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
        interp_flags=result_unprop_flags,
        propagated_interp_flags=result_propagated_flags,
    )
