"""
midl_pipeline
------
Multi-satellite solar wind merging pipeline for SWMF/BATS-R-US boundary conditions.

Merges 1-minute data from ACE, DSCOVR, and WIND into quality-screened upstream
boundary condition files.

Public API
----------
midl(start, end, raw_dir)
    Process a date range: read L1_raw, despike, quality-score, merge, propagate.
    Returns a MIDLResult with unpropagated and propagated DataFrames.
download_day(day, cda, raw_dir)
    Download raw satellite data to raw_dir/.
write_monthly_outputs(result, output_dir)
    Write MIDLResult to monthly CSV and DAT files.
plot_day(result, day_str, output_dir)
    Plot all variables for one day.
plot_variable(result, var, day_str, output_dir)
    Plot a single variable for one day.
"""

# Public API is resolved lazily (PEP 562): the submodules behind it pull in
# heavy third-party stacks (l1_plot -> matplotlib, l1_pipeline -> spacepy +
# pyspedas, l1_mhd -> xarray) that lightweight importers — notably the
# per-minute midl_realtime tick — must not pay for.
_LAZY_API = {
    'midl': 'l1_midl',
    'MIDLResult': 'l1_midl',
    'write_monthly_outputs': 'l1_writers',
    'plot_day': 'l1_plot',
    'plot_variable': 'l1_plot',
    'plot_day_from_csv': 'l1_plot',
    'download_day': 'l1_pipeline',
    'mhd_propagation': 'l1_mhd',
}

__all__ = list(_LAZY_API)


def __getattr__(name):
    try:
        submodule = _LAZY_API[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(f'.{submodule}', __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_API))
