"""Shared test helpers — synthetic DataFrame builders and constants."""
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _synthetic_sat_df(
    n=1440,
    bx=5.0, by=0.0, bz=0.0,
    ux=-400.0, uy=0.0, uz=0.0,
    rho=5.0, T=1e5,
    start="2024-05-01",
    noise=0.0,
    seed=42,
):
    """Build a synthetic per-satellite DataFrame on 1-min grid."""
    idx = pd.date_range(start, periods=n, freq="min")
    rng = np.random.default_rng(seed)
    data = {
        "Bx": np.full(n, bx) + noise * rng.standard_normal(n),
        "By": np.full(n, by) + noise * rng.standard_normal(n),
        "Bz": np.full(n, bz) + noise * rng.standard_normal(n),
        "Ux": np.full(n, ux) + noise * rng.standard_normal(n),
        "Uy": np.full(n, uy) + noise * rng.standard_normal(n),
        "Uz": np.full(n, uz) + noise * rng.standard_normal(n),
        "rho": np.full(n, rho) + noise * rng.standard_normal(n),
        "T": np.full(n, T) + noise * rng.standard_normal(n),
    }
    return pd.DataFrame(data, index=idx)


def _synthetic_sat_df_with_gap(n=60, gap_start=20, gap_end=30, gap_cols=None, **kwargs):
    """Synthetic df with NaN gap in specified columns."""
    df = _synthetic_sat_df(n=n, **kwargs)
    if gap_cols is None:
        gap_cols = df.columns.tolist()
    df.loc[df.index[gap_start:gap_end], gap_cols] = np.nan
    return df


def _synthetic_sat_df_with_spike(n=60, spike_index=30, spike_col="Bx", spike_value=999.0, **kwargs):
    """Synthetic df with a single spike injected."""
    df = _synthetic_sat_df(n=n, **kwargs)
    df.iloc[spike_index, df.columns.get_loc(spike_col)] = spike_value
    return df


def _synthetic_data_map(n_sats=3, n=1440, **kwargs):
    """Build dict[str, DataFrame] for combine_data_priority()."""
    sat_names = ["ace", "dscovr", "wind", "solar1"][:n_sats]
    return {name: _synthetic_sat_df(n=n, seed=42 + i, **kwargs) for i, name in enumerate(sat_names)}


def _synthetic_positions(x_ace_re=220.0, x_dscovr_re=230.0, x_wind_re=256.0, x_solar1_re=np.nan):
    """Build a positions dict in km (Re * 6371)."""
    def _to_km(re_val):
        return re_val * 6371.0 if np.isfinite(re_val) else np.nan
    return {
        "ace": _to_km(x_ace_re),
        "dscovr": _to_km(x_dscovr_re),
        "wind": _to_km(x_wind_re),
        "solar1": _to_km(x_solar1_re),
    }
