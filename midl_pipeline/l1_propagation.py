"""
l1_propagation.py
-----------------
Ballistic solar-wind propagation from L1 to an inner boundary.

Assumes the solar wind travels radially at its measured speed (frozen-in
approximation).  Enforces causality by discarding parcels that are
overtaken by faster parcels emitted later.

For timing purposes, Ux is filled with unrestricted interpolation so that
variables with valid data (e.g. B) are not lost during gaps in plasma.
The actual Ux values in the output still reflect the original gap structure.

Public entry point: ballistic_propagation()
"""
import numpy as np
import pandas as pd


VX_COL = 'Vx Velocity, km/s, GSE'


def ballistic_propagation(orbit, raw_data, target_x_km=90000,
                          extend_minutes=0):
    """Propagate L1 observations to a target X position along the Sun-Earth line.

    Parameters
    ----------
    orbit : pd.Series
        Must contain 'X_GSE' (spacecraft X in km).
    raw_data : pd.DataFrame
        L1 time series with column 'Vx Velocity, km/s, GSE' and a
        DatetimeIndex.
    target_x_km : float
        Target boundary X position in km (positive Sunward, e.g.
        14*6371 ≈ 89 194 km for 14 Re).
    extend_minutes : int
        Extend the output grid this many minutes past the last input
        time.  The ballistic shift moves the newest measurements 30-90
        minutes into the future (the L1 transit lead); with the default
        0 those arrivals are clipped at the input's end — correct for
        historical reprocessing, but the realtime IMF views pass >0 to
        keep the lead.  Grid minutes beyond the last arrival stay NaN.

    Returns
    -------
    pd.DataFrame
        Propagated data on a complete 1-minute grid over the input time
        range (plus extend_minutes).  Arrival times are computed exactly
        (no rounding), then interpolated onto the regular grid so no
        minutes are dropped.  Variables with real data gaps remain NaN;
        only the timing calculation uses gap-filled Ux.
    """
    input_df = raw_data.copy()
    x_gse = np.float64(orbit['X_GSE'].item())
    target_x_km = np.float64(target_x_km)

    # Build a gap-free Ux for travel-time computation only.
    # This ensures B (and other vars) are not lost when Ux has gaps.
    vx_for_timing = input_df[VX_COL].interpolate(method='time')
    vx = np.asarray(vx_for_timing, dtype=np.float64)

    # Remember which input minutes had NaN Ux (on the original 1-min grid)
    # so we can restore those gaps in the output after propagation.
    vx_orig_nan = input_df[VX_COL].isna()

    # Ballistic travel time from spacecraft X to target X.
    travel_time_seconds = np.round((x_gse - target_x_km) / vx * (-1))
    travel_time = travel_time_seconds.astype('timedelta64[s]')

    # Compute arrival timestamps and enforce shock ordering.
    arrivals = input_df.index + travel_time
    valid_mask = pd.Series(True, index=input_df.index)

    for i in range(1, len(arrivals)):
        previous_indices = input_df.index[:i][arrivals[:i] > arrivals[i]]
        valid_mask.loc[previous_indices] = False

    # Drop older parcels overtaken by faster later parcels.
    input_df = input_df.loc[valid_mask]
    input_df.index = arrivals[valid_mask]

    # Resample onto a regular 1-minute grid.  Arrival times are irregular
    # (not aligned to whole minutes), so we merge them with the target grid
    # and interpolate to snap to grid points.  Limit=2 bridges only the
    # sub-minute jitter from time-shifting; real data gaps pass through as NaN.
    numeric_cols = input_df.select_dtypes(include='number').columns
    input_df = input_df[numeric_cols]
    input_df = input_df[input_df.index.notna()]
    input_df = input_df[~input_df.index.duplicated(keep='first')]
    input_df = input_df.sort_index()

    grid = pd.date_range(
        raw_data.index.min(),
        raw_data.index.max() + pd.Timedelta(minutes=extend_minutes),
        freq='min')
    combined = input_df.index.union(grid)
    result = input_df.reindex(combined).interpolate(
        method='index', limit=2).reindex(grid)

    # Restore NaN in Ux where it was originally missing.
    # The gap-filled Ux was only for timing, not for output.
    if VX_COL in result.columns:
        # Map original NaN mask onto the output grid (same freq, close alignment).
        vx_nan_on_grid = vx_orig_nan.reindex(grid, method='nearest', tolerance='30s')
        vx_nan_mask = vx_nan_on_grid.fillna(False).astype(bool)
        result.loc[vx_nan_mask, VX_COL] = np.nan

    return result
