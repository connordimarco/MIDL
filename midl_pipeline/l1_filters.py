"""
l1_filters.py
-------------
Signal-quality filters applied to the combined L1 time series.

Public entry points:
  - despike()              — centered 3-point median on B and plasma.
  - smooth_transitions()   — boxcar smoothing at source-change steps exceeding a magnitude threshold (plasma only).

Shared utilities:
  - INTERP_LIMITS        — per-variable max-gap limits for interpolation.
  - interpolate_with_limits() — apply per-variable gap-limited interpolation.
"""
import numpy as np
import pandas as pd


# Max gap (in minutes) allowed when filling NaN runs per variable.
# Kept conservative so long outages remain NaN rather than being
# silently bridged with a ramp.
INTERP_LIMITS = {
    'Bx': 5,
    'By': 5,
    'Bz': 5,
    'Ux': 60,
    'Uy': 60,
    'Uz': 60,
    'rho': 60,
    'T': 60,
}


def interpolate_with_limits(df, limits=None, method='time', return_mask=False):
    """Interpolate each column with a per-variable max gap limit.

    Parameters
    ----------
    df : pd.DataFrame
    limits : dict[str, int] or None
        Maps column name -> max consecutive NaN steps to fill.
        Defaults to INTERP_LIMITS.
    method : str
        Interpolation method passed to pandas (default 'time').
    return_mask : bool
        When True, also return a DataFrame of boolean masks (same index,
        one column per interpolated variable) that are True at cells which
        changed from NaN to non-NaN — i.e. the gap-filled cells. The masks
        are purely diagnostic; the numeric output is byte-for-byte identical
        whether or not this flag is set.

    Returns
    -------
    out : pd.DataFrame
        Interpolated copy of *df* (always returned).
    fill_mask : pd.DataFrame
        Only when ``return_mask`` is True. Boolean, True where a NaN was
        filled by interpolation.
    """
    if limits is None:
        limits = INTERP_LIMITS
    out = df.copy()
    fill_mask = {}
    for col, limit in limits.items():
        if col in out.columns:
            before_nan = out[col].isna()
            out[col] = out[col].infer_objects(copy=False).interpolate(
                method=method,
                limit=limit,
                limit_area='inside',
            )
            if return_mask:
                # Cells that were NaN before and are now finite were filled.
                fill_mask[col] = before_nan & out[col].notna()
    if return_mask:
        return out, pd.DataFrame(fill_mask, index=out.index)
    return out


def median_filter_3(a, min_periods=2):
    """Apply a centered 3-point rolling median to a 1-D signal."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError('Input `a` must be one-dimensional.')
    if len(a) < 3:
        return a.copy()
    return pd.Series(a).rolling(
        window=3, center=True, min_periods=min_periods).median().to_numpy()


# ---------------------------------------------------------------------------
# Transition smoothing
# ---------------------------------------------------------------------------

# Plasma columns and how their jump magnitude C is computed.
#   'pct' : C = 100 * (max(|M1|, |M2|) / min(|M1|, |M2|) - 1)  [%]
#   'abs' : C = |M2 - M1|                                        [km/s]
_TRANSITION_COLS = {
    'Ux': 'pct',  # treat as positive magnitude
    'Uy': 'abs',
    'Uz': 'abs',
    'rho': 'pct',
    'T': 'pct',
}
_CMAX_DEFAULT = 20.0   # threshold below which no smoothing is applied
_WMAX_DEFAULT = 60     # maximum smoothing window [min]
_RATE_DEFAULT = 5.0    # W = C / rate, clipped to WMAX


def _jump_magnitude(m1, m2, col_type):
    """Return jump magnitude C given consecutive values and column type."""
    if col_type == 'pct':
        a, b = abs(m1), abs(m2)
        if min(a, b) == 0:
            return 0.0
        return 100.0 * (max(a, b) / min(a, b) - 1.0)
    return abs(m2 - m1)


def _apply_boxcar(smoothed, original, i, w):
    """Write a centered rolling mean of width *w* into *smoothed* around index *i*."""
    n = len(original)
    half = w // 2
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    ext_lo = max(0, lo - half)
    ext_hi = min(n, hi + half)
    rolled = (pd.Series(original[ext_lo:ext_hi])
              .rolling(w, center=True, min_periods=1)
              .mean().values)
    smoothed[lo:hi] = rolled[(lo - ext_lo):(lo - ext_lo) + (hi - lo)]


def smooth_transitions(df, source_changed=None, cmax=_CMAX_DEFAULT,
                       wmax=_WMAX_DEFAULT, rate=_RATE_DEFAULT):
    """Smooth large step changes in plasma output at source transitions.

    For each plasma column, only considers minutes where the contributing
    satellite set changed (``source_changed[col][i]`` is True).  At those
    minutes, if the jump magnitude C also exceeds *cmax*, replaces the values
    in a window of width W with a rolling mean of the original values.

        W = nint(min(wmax, C / rate))   [minutes]

    Parameters
    ----------
    df : pd.DataFrame
        Combined L1 data (must be on a 1-minute grid, no NaN gaps).
    source_changed : dict[str, pd.Series of bool] or None
        Per-column boolean mask that is True at minutes where the source
        changed.  If None, all minutes are candidates (legacy behaviour).
    cmax : float
        Minimum jump magnitude to trigger smoothing.
        Units: % for rho, T, |Ux|; km/s for Uy, Uz.
    wmax : int
        Maximum smoothing window in minutes.
    rate : float
        Rate parameter: W = C / rate.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with source-transition steps smoothed.
    """
    out = df.copy()

    for col, col_type in _TRANSITION_COLS.items():
        if col not in out.columns:
            continue

        sc = source_changed.get(col) if source_changed is not None else None
        original = out[col].values.astype(float)
        smoothed = original.copy()

        for i in range(1, len(original)):
            if sc is not None and not bool(sc.iloc[i]):
                continue  # same source as previous minute — preserve original
            m1, m2 = original[i - 1], original[i]
            if not (np.isfinite(m1) and np.isfinite(m2)):
                continue
            c = _jump_magnitude(m1, m2, col_type)
            if c <= cmax:
                continue
            w = max(2, int(np.round(min(wmax, c / rate))))
            _apply_boxcar(smoothed, original, i, w)

        out[col] = smoothed

    return out


def despike(df, return_mask=False):
    """Apply a centered 3-point median filter to B and plasma columns.

    Parameters
    ----------
    df : pd.DataFrame
        Combined L1 data with columns Bx/By/Bz/Ux/Uy/Uz/rho.
    return_mask : bool
        If True, also return a boolean DataFrame marking cells the median
        filter reconstructed from neighbors (NaN before, finite after).  The
        filter bridges isolated single-minute dropouts, so those cells are
        provenance-interpolated, not direct.

    Returns
    -------
    df_clean : pd.DataFrame
        Copy of *df* with the median filter applied.
    bridge_mask : pd.DataFrame
        Only returned when *return_mask* is True (see above).
    """
    print('  -> Running Despike (3-point median)...')
    df_clean = df.copy()
    bridge_mask = {}
    for col in ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho']:
        if col in df_clean.columns:
            before_nan = df_clean[col].isna()
            df_clean[col] = median_filter_3(df_clean[col].values)
            if return_mask:
                bridge_mask[col] = before_nan & df_clean[col].notna()
    if return_mask:
        return df_clean, pd.DataFrame(bridge_mask, index=df_clean.index)
    return df_clean
