"""
l1_combine.py
-------------
Combines per-satellite data into a single merged output.

Core responsibilities:
  - Calls score_all_plasma() to obtain per-satellite quality bad-masks.
  - Selects the best available source for every variable/minute using
    agreement-first rules plus previous-value continuity fallback.
  - Combines temperature via geometric median in log-space.

Public functions: combine_data_priority(), combine_temperature()
"""
from itertools import combinations as _combinations

import numpy as np
import pandas as pd

from .l1_filters import median_filter_3
from .l1_quality import score_all_plasma


SAT_CODE = {'ace': 1, 'dscovr': 2, 'wind': 3, 'solar1': 4}
# Code 5 reserved for IMAP (pending data availability).
# To add IMAP when data is available:
#   1. pip install imap-data-access
#   2. Add 'imap': 5 to SAT_CODE
#   3. Add download function in l1_downloaders.py (query api.imap-mission.com for
#      instrument=mag, data_level=l2; download CDF; read b_gsm variable)
#   4. Add process function in l1_pipeline.py
#   5. Add 'imap' to SATELLITES in l1_midl.py
#   6. Populate Ix/Iy/Iz position columns in create_position_file()
#   7. Add color entry in l1_plot.py SAT_COLORS
#   8. For plasma: SWAPI L3 (bulk moments) needed — not yet in imap-processing

# Variables for which DSCOVR is deprioritized in the 2-satellite fallback.
# When only DSCOVR + one other satellite are available and they disagree,
# the non-DSCOVR satellite is chosen instead of using continuity logic.
_DSCOVR_DEPRIORITIZE_VARS = {'rho', 'T'}
_DSCOVR_CODE = SAT_CODE['dscovr']  # 2


def _switch_threshold(col):
    thresholds = {
        'Bx': 8.0,
        'By': 8.0,
        'Bz': 8.0,
        '|B|': 8.0,
        'Ux': 80.0,
        'Uy': 40.0,
        'Uz': 40.0,
        '|Vt|': 40.0,
        'rho': 2.0,
    }
    return thresholds.get(col, np.inf)


def _agree(v1, v2, col):
    """Return True when two values are close enough to be considered agreeing."""
    return abs(v1 - v2) <= _switch_threshold(col)


def _fallback_source(values, available_codes, prev_value,
                     deprioritize_code=None):
    """Fallback source when no pair agrees.

    Startup (no previous value): prefer WIND when available.
    Steady-state: choose source closest to previous output value.

    If *deprioritize_code* is set and a non-deprioritized alternative
    exists, the deprioritized satellite is never selected.
    """
    if deprioritize_code is not None and deprioritize_code in available_codes:
        alternatives = [c for c in available_codes if c != deprioritize_code]
        if alternatives:
            if len(alternatives) == 1:
                return alternatives[0]
            available_codes = alternatives

    if np.isfinite(prev_value):
        return min(available_codes, key=lambda c: abs(values[c] - prev_value))
    if 3 in available_codes:
        return 3
    return available_codes[0]


def _select_column_with_continuity(col, sat_series, bad_masks=None,
                                   deprioritize_code=None):
    """Merge one variable across satellites with continuity and quality logic.

    Parameters
    ----------
    col : str
        Variable name.
    sat_series : dict[str, pd.Series]
        Per-satellite series for this variable.
    bad_masks : dict[int, dict[str, pd.Series]] or None
        Per-satellite, per-variable boolean masks from the quality scorer.
        ``bad_masks[sat_code][var]`` is True where that satellite is bad.
    """
    # Minimum consecutive minutes the alternative satellite must be closer
    # before the fallback source switches.  Prevents single-minute noise
    # from triggering a source change in the 2-sat no-agreement path.
    _SWITCH_MIN = 3

    index = next(iter(sat_series.values())).index
    out_vals = np.full(len(index), np.nan, dtype=float)
    out_nsat = np.zeros(len(index), dtype=int)
    out_source = [None] * len(index)   # frozenset of sat codes contributing each minute

    prev_value = np.nan
    locked_source = None   # satellite code currently committed in fallback path
    switch_count = 0       # consecutive minutes the preferred candidate != locked

    for i, _ in enumerate(index):
        values = {
            code: sat_series[sat].iloc[i]
            for sat, code in SAT_CODE.items()
            if pd.notna(sat_series[sat].iloc[i])
        }

        # --- Quality gate: drop ANY satellite flagged bad ---
        if bad_masks is not None:
            for code in list(values.keys()):
                sat_masks = bad_masks.get(code)
                if sat_masks is not None and col in sat_masks:
                    if bool(sat_masks[col].iloc[i]):
                        values.pop(code, None)

        available = sorted(values.keys())
        n_sat = len(available)
        out_nsat[i] = n_sat

        if n_sat == 0:
            continue

        if n_sat == 1:
            out_vals[i] = values[available[0]]
            out_source[i] = frozenset(available)
            prev_value = out_vals[i]
            continue

        # Agreement-first logic:
        #  - all agree -> median of all
        #  - largest agreeing subset >= 3 -> median of that subset
        #  - any agreeing pair -> mean of that pair
        #  - none agree -> fallback source (WIND at startup, else closest to prev)
        pairs = []
        pairs_set = set()
        for p_idx, c1 in enumerate(available):
            for c2 in available[p_idx + 1:]:
                if _agree(values[c1], values[c2], col):
                    pairs.append((c1, c2))
                    pairs_set.add((c1, c2))

        max_pairs = n_sat * (n_sat - 1) // 2
        if len(pairs) == max_pairs:
            out_vals[i] = np.median([values[c] for c in available])
            out_source[i] = frozenset(available)
            prev_value = out_vals[i]
            continue

        # Check for a subset of 3+ that all agree pairwise.
        best_clique = None
        if n_sat >= 4 and len(pairs) >= 3:
            for size in range(n_sat - 1, 2, -1):
                for subset in _combinations(available, size):
                    if all((a, b) in pairs_set
                           for a, b in _combinations(subset, 2)):
                        best_clique = subset
                        break
                if best_clique:
                    break

        if best_clique:
            out_vals[i] = np.median([values[c] for c in best_clique])
            out_source[i] = frozenset(best_clique)
            prev_value = out_vals[i]
            continue

        if pairs:
            best_pair = min(pairs, key=lambda p: abs(
                values[p[0]] - values[p[1]]))
            out_vals[i] = 0.5 * (values[best_pair[0]] + values[best_pair[1]])
            out_source[i] = frozenset(best_pair)
            prev_value = out_vals[i]
            continue

        # Fallback: no agreeing pair.  Use hysteresis to prevent oscillation
        # when two satellites alternate which is slightly closer to prev_value.
        candidate = _fallback_source(values, available, prev_value,
                                     deprioritize_code=deprioritize_code)

        if locked_source is None or locked_source not in available:
            # Cold-start or locked satellite disappeared — accept immediately.
            locked_source = candidate
            switch_count = 0
        elif candidate == locked_source:
            switch_count = 0
        else:
            switch_count += 1
            if switch_count >= _SWITCH_MIN:
                locked_source = candidate
                switch_count = 0

        out_vals[i] = values[locked_source]
        out_source[i] = frozenset([locked_source])
        prev_value = out_vals[i]

    return (pd.Series(out_vals, index=index),
            pd.Series(out_nsat, index=index),
            pd.Series(out_source, index=index))


def _apply_source_to_components(source_series, component_sat_series, index):
    """Pick component values using a pre-computed source decision.

    Used for coupled vector selection: the source decision comes from
    running the magnitude (|B| or |Vt|) through the agreement logic,
    then this function applies that decision to the original components.

    Mirrors the aggregation in _select_column_with_continuity:
      1 satellite  -> use its value
      2 satellites -> mean
      3 satellites -> median
    """
    CODE_TO_SAT = {v: k for k, v in SAT_CODE.items()}
    out = np.full(len(index), np.nan, dtype=float)

    for i in range(len(index)):
        src = source_series.iloc[i]
        if src is None:
            continue
        codes = sorted(src)
        vals = [component_sat_series[CODE_TO_SAT[c]].iloc[i]
                for c in codes
                if pd.notna(component_sat_series[CODE_TO_SAT[c]].iloc[i])]
        if not vals:
            continue
        if len(vals) == 1:
            out[i] = vals[0]
        elif len(vals) == 2:
            out[i] = 0.5 * (vals[0] + vals[1])
        else:
            out[i] = np.median(vals)

    return pd.Series(out, index=index)


def combine_data_priority(data_map, master_grid):
    # Align each satellite to the same timeline before merging.
    def _dedup_and_reindex(df, grid):
        return df.reindex(grid)

    sat_dfs = {name: _dedup_and_reindex(
        data_map.get(name, pd.DataFrame(index=master_grid)), master_grid)
        for name in SAT_CODE}

    # --- Run quality scorer across all satellites ---
    print('  Running plasma quality assessment for all satellites...')
    all_bad_by_name = score_all_plasma(sat_dfs)
    all_bad_masks = {SAT_CODE[name]: masks
                     for name, masks in all_bad_by_name.items()}

    # T is excluded here — handled separately by combine_temperature().
    cols = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho']
    df_combined = pd.DataFrame(index=master_grid, columns=cols)
    source_map = {}   # col -> pd.Series of frozenset(sat_codes)

    def _sat_series_for(col):
        return {
            name: sat_dfs[name][col] if col in sat_dfs[name] else pd.Series(
                np.nan, index=master_grid)
            for name in SAT_CODE
        }

    # --- Block A: Magnetic field (Bx, By, Bz) coupled via |B| ---
    b_series = {comp: _sat_series_for(comp) for comp in ('Bx', 'By', 'Bz')}
    mag_b_series = {}
    for sat in SAT_CODE:
        mag_b_series[sat] = np.sqrt(
            b_series['Bx'][sat]**2 +
            b_series['By'][sat]**2 +
            b_series['Bz'][sat]**2)

    _, _, b_source = _select_column_with_continuity(
        '|B|', mag_b_series, bad_masks=None)

    for comp in ('Bx', 'By', 'Bz'):
        df_combined[comp] = _apply_source_to_components(
            b_source, b_series[comp], master_grid)
        source_map[comp] = b_source

    # --- Block B: Transverse velocity (Uy, Uz) coupled via |Vt| ---
    vt_series = {comp: _sat_series_for(comp) for comp in ('Uy', 'Uz')}
    mag_vt_series = {}
    for sat in SAT_CODE:
        mag_vt_series[sat] = np.sqrt(
            vt_series['Uy'][sat]**2 +
            vt_series['Uz'][sat]**2)

    vt_bad_masks = {}
    for name, code in SAT_CODE.items():
        name_masks = all_bad_by_name.get(name)
        if name_masks is not None:
            uy_bad = name_masks.get('Uy', pd.Series(False, index=master_grid))
            uz_bad = name_masks.get('Uz', pd.Series(False, index=master_grid))
            vt_bad_masks[code] = {'|Vt|': uy_bad | uz_bad}

    _, _, vt_source = _select_column_with_continuity(
        '|Vt|', mag_vt_series, bad_masks=vt_bad_masks)

    for comp in ('Uy', 'Uz'):
        df_combined[comp] = _apply_source_to_components(
            vt_source, vt_series[comp], master_grid)
        source_map[comp] = vt_source

    # --- Block C: Independent variables (Ux, rho) ---
    for col in ('Ux', 'rho'):
        sat_series = _sat_series_for(col)
        depri = _DSCOVR_CODE if col in _DSCOVR_DEPRIORITIZE_VARS else None
        values, _, source = _select_column_with_continuity(
            col, sat_series, bad_masks=all_bad_masks,
            deprioritize_code=depri)
        df_combined[col] = values
        source_map[col] = source

    return df_combined, source_map


# ---------------------------------------------------------------------------
# Temperature combiner
# ---------------------------------------------------------------------------
# T is handled separately from B and plasma because:
#   - It spans orders of magnitude, making threshold-based quality checks unreliable.
#   - Real propagation delays between spacecraft look identical to sensor disagreement.
#   - ACE SWE proton T, WIND thermal-speed-derived T, and DSCOVR PLASMAG T have
#     different calibrations that routinely disagree by 2-3x.
#
# Strategy:
#   1. Per-satellite 3-point median to remove single-minute spikes.
#   2. Per-satellite spikiness filter: if a satellite's T has high rolling log-std
#      over an 11-minute window (> 0.5), those minutes are excluded.
#   3. Per-minute geometric median across available satellites:
#        exp(median(log(T_values)))
#   4. Final 3-point rolling median on the combined result.

# Rolling log-std threshold above which a satellite's T is considered too noisy
# to contribute to the combination. Evaluated over an 11-minute window.
_T_SPIKY_LOG_STD = 0.5
_T_SPIKY_WINDOW = 11


def combine_temperature(data_map, master_grid, t_interp_flags=None,
                        return_flag=False):
    """Merge proton temperature across all L1 satellites.

    Parameters
    ----------
    data_map : dict[str, pd.DataFrame]
        Per-satellite DataFrames keyed by satellite name.
        Each must contain a 'T' column.
    master_grid : pd.DatetimeIndex
        Target 1-minute time grid.
    t_interp_flags : dict[str, pd.Series] or None
        Optional per-satellite boolean Stage-2 interpolation flags for T
        (True where that satellite's T at that minute was gap-filled before
        this function ran).  Used to compute the merged T provenance flag.
    return_flag : bool
        When True, also return an integer Series ``t_interp`` on master_grid
        scoring the *contributing* satellites' T provenance at each minute
        (interpolated = upstream Stage-2 fill via ``t_interp_flags`` OR this
        function's internal 2-minute pre-fill):

            0 — all contributing values direct,
            1 — mixed (some interpolated, some direct),
            2 — ALL contributing values interpolated.

    Returns
    -------
    combined : pd.Series
        Combined temperature on master_grid.
    t_source : pd.Series
        Per-minute frozenset of contributing satellite codes.
    t_interp : pd.Series of int (0/1/2)
        Only when ``return_flag`` is True.

    Notes
    -----
    The numeric outputs (``combined``, ``t_source``) are identical whether
    or not ``return_flag`` is set — the flag machinery is additive only.
    """
    sat_T = {}
    # Per-satellite boolean: True where the value used here came from
    # interpolation (upstream Stage-2 fill OR this routine's 2-min pre-fill).
    sat_T_interp = {}
    for sat in SAT_CODE:
        if sat in data_map and 'T' in data_map[sat].columns:
            raw = data_map[sat]['T'].reindex(master_grid)
            s = raw.interpolate(method='time', limit=2, limit_area='inside')
            # Cells the 2-min pre-fill turned from NaN into finite values.
            prefill = raw.isna() & s.notna()
            # Upstream Stage-2 flag for this satellite's T, aligned here.
            upstream = (t_interp_flags.get(sat) if t_interp_flags else None)
            if upstream is not None:
                upstream = upstream.reindex(master_grid).fillna(False).astype(bool)
            else:
                upstream = pd.Series(False, index=master_grid)
            sat_T_interp[sat] = (prefill | upstream)
            s = pd.Series(median_filter_3(s.values), index=master_grid)
            log_std = (
                np.log(s.clip(lower=1))
                .rolling(_T_SPIKY_WINDOW, center=True, min_periods=5)
                .std()
            )
            s = s.where(log_std <= _T_SPIKY_LOG_STD, other=np.nan)
            sat_T[sat] = s
        else:
            sat_T[sat] = pd.Series(np.nan, index=master_grid)
            sat_T_interp[sat] = pd.Series(False, index=master_grid)

    df = pd.DataFrame(sat_T)
    log_df = np.log(df.clip(lower=1))
    log_median = log_df.median(axis=1, skipna=True)
    out = np.where(df.notna().any(axis=1), np.exp(log_median), np.nan)

    if 'T' in _DSCOVR_DEPRIORITIZE_VARS:
        n_available = df.notna().sum(axis=1)
        dscovr_present = df['dscovr'].notna() if 'dscovr' in df else False
        mask_2sat_dscovr = (n_available == 2) & dscovr_present
        non_dscovr_cols = [s for s in SAT_CODE if s != 'dscovr']
        non_dscovr = df[non_dscovr_cols].bfill(axis=1).iloc[:, 0]
        out = np.where(mask_2sat_dscovr, non_dscovr.values, out)

    t_source = [None] * len(master_grid)
    for i in range(len(master_grid)):
        contribs = frozenset(
            SAT_CODE[sat] for sat in SAT_CODE
            if pd.notna(sat_T[sat].iloc[i]))
        if 'T' in _DSCOVR_DEPRIORITIZE_VARS and mask_2sat_dscovr.iloc[i]:
            contribs = contribs - {_DSCOVR_CODE}
        t_source[i] = contribs if contribs else None
    t_source = pd.Series(t_source, index=master_grid)

    combined = pd.Series(out, index=master_grid)
    combined = pd.Series(median_filter_3(combined.values), index=master_grid)

    if not return_flag:
        return combined, t_source

    # Merged T provenance level over the *contributing* satellites' interp
    # flags (same contributing set as t_source — a satellite contributes iff
    # its post-spiky sat_T value is finite at that minute):
    #   0 = none interpolated, 1 = mixed, 2 = all interpolated.
    code_to_sat = {v: k for k, v in SAT_CODE.items()}
    t_interp_vals = np.zeros(len(master_grid), dtype=np.int64)
    interp_arr = {sat: sat_T_interp[sat].values for sat in SAT_CODE}
    src_vals = t_source.values
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
            if interp_arr[sat][i]:
                n_flagged += 1
        if n_contrib == 0 or n_flagged == 0:
            continue                  # 0: all direct
        elif n_flagged == n_contrib:
            t_interp_vals[i] = 2      # all contributing values interpolated
        else:
            t_interp_vals[i] = 1      # mixed
    t_interp = pd.Series(t_interp_vals, index=master_grid)
    return combined, t_source, t_interp
