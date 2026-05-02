"""
l1_quality.py
-------------
Plasma quality assessment for all L1 satellites.

Produces per-satellite, per-variable, per-timestep boolean bad-masks.
Three checks are applied per satellite:

  1. Flat-plateau detection   — stuck/near-constant instrument readings
  2. Outlier detection        — flags odd-one-out when others agree
  3. Near-zero clearing       — Uy/Uz pinned near zero (DSCOVR only)

Main entry point: ``score_all_plasma()``.
"""

from itertools import combinations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Flat-plateau detection (generalised from the original Uy/Uz-only check)
# ---------------------------------------------------------------------------

# Per-variable rolling-window parameters.  Larger windows and looser
# thresholds for Ux (advisor: "Ux is often OK").
_PLATEAU_PARAMS = {
    'Ux':  {'window': 15, 'std_thresh': 1.0,    'max_unique': 3},
    'Uy':  {'window': 11, 'std_thresh': 0.08,   'max_unique': 3},
    'Uz':  {'window': 11, 'std_thresh': 0.08,   'max_unique': 3},
    'rho': {'window': 15, 'std_thresh': 0.05,   'max_unique': 3},
}


def _detect_flat_plateau(series, window, std_thresh, max_unique):
    """Return boolean Series — True where the signal looks stuck/flat."""
    s = series.copy()
    if s.empty or s.isna().all():
        return pd.Series(False, index=s.index)

    rolling_std = s.rolling(window=window, center=True, min_periods=5).std()
    rolling_unique = s.rolling(window=window, center=True, min_periods=5).apply(
        lambda x: pd.Series(x).dropna().nunique(), raw=False,
    )

    mask = (rolling_std <= std_thresh) & (rolling_unique <= max_unique)
    return mask.fillna(False)


def check_flat_plateau(df_dsc, variables=None):
    """Run flat-plateau detection on each requested DSCOVR plasma variable.

    Returns dict[str, pd.Series] mapping variable name -> bad mask (True=bad).
    """
    if variables is None:
        variables = list(_PLATEAU_PARAMS.keys())

    masks = {}
    for var in variables:
        if var not in _PLATEAU_PARAMS:
            continue
        if var not in df_dsc.columns or df_dsc[var].isna().all():
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue
        p = _PLATEAU_PARAMS[var]
        masks[var] = _detect_flat_plateau(
            df_dsc[var], p['window'], p['std_thresh'], p['max_unique'],
        )
    return masks


# ---------------------------------------------------------------------------
# 2. Outlier detection
# ---------------------------------------------------------------------------

_OUTLIER_PARAMS = {
    'Ux':  {'mode': 'abs', 'threshold': 50.0,  'window': 31},
    'Uy':  {'mode': 'abs', 'threshold': 30.0,  'window': 31},
    'Uz':  {'mode': 'abs', 'threshold': 30.0,  'window': 31},
    'rho': {'mode': 'ratio', 'threshold': 3.0,  'window': 61},
}


def check_outlier_satellite(sat_dfs, variables=None):
    """Flag outlier satellites when other pairs agree and this one disagrees.

    A satellite is flagged only when at least one pair of the remaining
    satellites agrees (pairwise deviation within threshold) **and** it
    disagrees with both members of that pair.  Requires >= 3 satellites
    with non-NaN data for the variable; returns empty masks otherwise.

    Parameters
    ----------
    sat_dfs : dict[str, pd.DataFrame]
        Per-satellite DataFrames keyed by satellite name.

    Returns
    -------
    result : dict[str, dict[str, pd.Series]]
        ``result[sat_name][var]`` is True where that satellite is an outlier.
    """
    if variables is None:
        variables = list(_OUTLIER_PARAMS.keys())

    names = sorted(sat_dfs.keys())
    idx = next(iter(sat_dfs.values())).index
    result = {name: {} for name in names}

    for var in variables:
        p = _OUTLIER_PARAMS[var]
        w = p['window']

        s = {}
        for name in names:
            df = sat_dfs[name]
            s[name] = df[var].reindex(
                idx) if var in df.columns else pd.Series(np.nan, index=idx)

        has_data = [name for name in names if not s[name].isna().all()]

        if len(has_data) < 3:
            for name in names:
                result[name][var] = pd.Series(False, index=idx)
            continue

        pair_ok = {}
        for a, b in combinations(has_data, 2):
            key = (a, b)
            if p['mode'] == 'abs':
                dev = (s[a] - s[b]).abs()
                roll = dev.rolling(window=w, center=True,
                                   min_periods=5).median()
                pair_ok[key] = (roll <= p['threshold']).fillna(True)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = s[a] / s[b]
                log_r = np.log(ratio.clip(lower=1e-10)).abs()
                log_thresh = np.log(p['threshold'])
                roll = log_r.rolling(window=w, center=True,
                                     min_periods=5).median()
                pair_ok[key] = (roll <= log_thresh).fillna(True)

        for name in names:
            if name not in has_data:
                result[name][var] = pd.Series(False, index=idx)
                continue

            others = [n for n in has_data if n != name]
            flagged = pd.Series(False, index=idx)
            for a, b in combinations(others, 2):
                others_agree = pair_ok[(a, b)]
                key_a = (min(name, a), max(name, a))
                key_b = (min(name, b), max(name, b))
                disagrees_a = ~pair_ok[key_a]
                disagrees_b = ~pair_ok[key_b]
                flagged = flagged | (others_agree & disagrees_a & disagrees_b)

            result[name][var] = flagged.fillna(False)

    return result


# ---------------------------------------------------------------------------
# 3. NaN-fraction (data-gap) metric
# ---------------------------------------------------------------------------

def check_nan_fraction(df_dsc, variables=None, window=60, threshold=0.5):
    """Flag windows where DSCOVR has > threshold fraction of NaN.

    A high NaN rate suggests the instrument is struggling; surrounding
    non-NaN points in the same window are also suspect.

    Returns dict[str, pd.Series] mapping variable name -> bad mask (True=bad).
    """
    if variables is None:
        variables = ['Ux', 'Uy', 'Uz', 'rho']

    masks = {}
    for var in variables:
        if var not in df_dsc.columns:
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue
        nan_flag = df_dsc[var].isna().astype(float)
        frac = nan_flag.rolling(
            window=window, center=True, min_periods=10).mean()
        masks[var] = (frac > threshold).fillna(False)
    return masks


# ---------------------------------------------------------------------------
# 4. Near-zero Uy/Uz (legacy check, moved here from l1_combine)
# ---------------------------------------------------------------------------

def check_near_zero(df_dsc, variables=None, atol=0.5):
    """Flag Uy/Uz near-zero values as suspect.

    Returns dict[str, pd.Series] mapping variable name -> bad mask (True=bad).
    """
    if variables is None:
        variables = ['Uy', 'Uz']

    masks = {}
    for var in variables:
        if var not in df_dsc.columns:
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue
        masks[var] = df_dsc[var].abs().le(atol) & df_dsc[var].notna()
    return masks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

PLASMA_VARS = ['Ux', 'Uy', 'Uz', 'rho']


def score_all_plasma(sat_dfs):
    """Evaluate plasma quality for every satellite, variable, and minute.

    Parameters
    ----------
    sat_dfs : dict[str, pd.DataFrame]
        Per-satellite DataFrames keyed by satellite name.

    Returns
    -------
    all_bad : dict[str, dict[str, pd.Series]]
        ``all_bad[sat_name][var]`` is True where that satellite's value
        should not be used.
    """
    idx = None
    for df in sat_dfs.values():
        if not df.empty:
            idx = df.index
            break
    if idx is None:
        return {name: {} for name in sat_dfs}

    aligned = {}
    for name, df in sat_dfs.items():
        aligned[name] = df.reindex(idx) if not df.empty else pd.DataFrame(
            index=idx)

    outlier_masks = check_outlier_satellite(aligned, PLASMA_VARS)

    all_bad = {}
    for name, df_target in aligned.items():
        plateau_m = check_flat_plateau(df_target, PLASMA_VARS)
        zero_m = (check_near_zero(df_target, ['Uy', 'Uz'])
                  if name == 'dscovr' else {})

        bad = {}
        for var in PLASMA_VARS:
            composite = pd.Series(False, index=idx)
            if var in outlier_masks.get(name, {}):
                composite = composite | outlier_masks[name][var]
            for masks in [plateau_m, zero_m]:
                if var in masks:
                    composite = composite | masks[var]
            bad[var] = composite

        n_flagged = {v: int(m.sum()) for v, m in bad.items()}
        print(f'  {name.upper()} quality: flagged bad: {n_flagged}')
        all_bad[name] = bad

    return all_bad
