"""
l1_mhd.py
---------
Run BATSRUS in 1D configuration on a continuous MIDL combined time series
and return the full native-resolution solar-wind profile as an xarray
Dataset.

Public entry point: mhd_propagation()

Notes
-----
BATSRUS cannot ingest NaN in the inflow file, so gaps are unbounded-
interpolated before writing L1.dat.  The MHD solution is kept everywhere
— even at gap-filled input minutes, BATSRUS produces a physics-based
solution informed by the surrounding valid data.

Each year's run uses a fresh initial condition (uniform flow), which
produces transient spin-up artifacts in the first ~tens of minutes.
A 1-hour spin-up pad is prepended to L1.dat and sliced off after
parsing.

BATSRUS 1D plot files are Fortran-sequential binary records — one record
per cell, 15 doubles each:
    [dx, x, y, z, Rho, Ux, Uy, Uz, Bx, By, Bz, P, jx, jy, jz]
Temperature is derived from P and Rho via the ideal-gas law.
"""
import glob
import os
import re
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUMERIC_COLS = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']
_KB = 1.380649e-23            # J/K
_AMU_CC_TO_M3 = 1.0e6         # amu/cc * (1e6 cc/m^3) -> amu/m^3  (==n/m^3)
_NPA_TO_PA = 1.0e-9
_RE_KM = 6371.0
_OPENMPI_PATH = os.environ.get('MIDL_OPENMPI_PATH', '/usr/lib64/openmpi/bin')
_MPIRUN = os.environ.get('MIDL_MPIRUN', 'mpirun -np 1')
_SPINUP = pd.Timedelta(hours=1)

# Temperature floor [K] for the BATSRUS inflow boundary condition only. BATSRUS
# becomes numerically unstable and writes degenerate plot files when the inflow
# proton T drops into the ~8,000-10,000 K range (the combined inflow bottomed at
# ~8,100 K and crashed; moment data never went below ~11,000 K and ran fine).
# This is HIGHER than the science floor (_T_FLOOR=5,000 K in l1_combine.py) on
# purpose: the L1/14Re/32Re science products keep their lower floor for fidelity,
# while the MHD solver gets the headroom it needs for stability.
_MHD_T_FLOOR = 1.0e4

# IDL record layout: offsets into the 15-double cell record
_IDL_COL_X  = 1
_IDL_COL_RHO = 4
_IDL_COL_UX  = 5
_IDL_COL_UY  = 6
_IDL_COL_UZ  = 7
_IDL_COL_BX  = 8
_IDL_COL_BY  = 9
_IDL_COL_BZ  = 10
_IDL_COL_P   = 11
_IDL_NDOUBLE = 15


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mhd_propagation(df_combined, ref_x_daily, work_dir=None, batsrus_dir=None,
                    *, allow_partial=False):
    """Run BATSRUS 1D on the continuous combined L1 time series.

    Parameters
    ----------
    df_combined : pd.DataFrame
        Output of Stage 5 of midl() — columns Bx, By, Bz, Ux, Uy, Uz,
        rho, T at 1-min cadence, DatetimeIndex.
    ref_x_daily : dict[datetime.date, float]
        Per-day reference satellite X position in km.  Used as the L1
        inflow face for the 1D domain.
    work_dir : str or None
        Scratch directory for this invocation's BATSRUS run.  A fresh
        tempdir is created if None.  The caller is responsible for
        cleanup of the parent (we only clean the scratch .idl files).
    batsrus_dir : str or None
        Path to the built BATSRUS install (the directory containing
        run_mhd/BATSRUS.exe).  Defaults to `MIDL-Pipeline/BATSRUS`
        relative to the midl_pipeline package.  The PARAM.in.MIDL
        template is always loaded from the midl_pipeline package.
    allow_partial : bool
        If True, a BATSRUS crash mid-run is caught and whatever plot
        files were written before the crash are still parsed and
        returned.  The returned Dataset carries
        ``ds.attrs['batsrus_crashed'] = 1`` and
        ``ds.attrs['batsrus_crash_info']`` with the tail of the BATSRUS
        stdout/stderr.  Callers that want to resume past the crash point
        can re-invoke this function on a sub-slice of df_combined.  If
        BATSRUS produces no plot files at all, the crash is re-raised
        regardless of this flag.

    Returns
    -------
    xr.Dataset
        Dimensions: (time, x)
        Coords:     time (1-min cadence), x (Re)
        Data vars:  Bx, By, Bz, Ux, Uy, Uz, rho, T  (shape [time, x])
    """
    if df_combined.empty:
        raise ValueError('df_combined is empty — nothing to propagate.')

    # Resolve paths.
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if batsrus_dir is None:
        batsrus_dir = os.path.abspath(os.path.join(pkg_dir, '..', 'BATSRUS'))

    run_template_dir = os.path.join(batsrus_dir, 'run_mhd')
    batsrus_exe = os.path.join(run_template_dir, 'BATSRUS.exe')
    param_template = os.path.join(pkg_dir, 'PARAM.in.MIDL')

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='midl_mhd_')
    os.makedirs(work_dir, exist_ok=True)

    # Stage the BATSRUS run directory into work_dir.
    run_dir = _stage_run_dir(run_template_dir, work_dir)

    # Build the unbounded-filled inflow frame + spin-up pad.
    df_filled = _fill_for_mhd(df_combined)
    df_padded, real_start = _prepend_spinup_pad(df_filled, _SPINUP)

    # The L1.dat header TIMEDELAY is 0, so the #STARTTIME in PARAM.in
    # must equal the first timestamp in L1.dat (i.e. the start of the
    # spin-up pad).  tSimulationMax in PARAM.in is irrelevant because
    # #ENDTIME overrides it.
    sim_start = df_padded.index[0].to_pydatetime()
    sim_end   = df_padded.index[-1].to_pydatetime()

    # Write L1.dat using per-day reference X (constant per day).
    l1_dat = os.path.join(run_dir, 'L1.dat')
    _write_l1_dat(df_padded, ref_x_daily, l1_dat)

    # Render PARAM.in.
    param_in = os.path.join(run_dir, 'PARAM.in')
    _render_param_in(param_template, param_in, sim_start, sim_end)

    # Run BATSRUS.  In allow_partial mode, a mid-run crash is captured
    # and we fall through to plot-file parsing so whatever snapshots
    # were written before the abort are still recovered.
    crash_info = None
    try:
        _run_batsrus(batsrus_exe, run_dir)
    except RuntimeError as e:
        if not allow_partial:
            raise
        crash_info = str(e)

    # Parse plot files -> xr.Dataset over the padded time range.
    try:
        ds = _parse_plot_files(run_dir, sim_start)
    except RuntimeError:
        if crash_info is not None:
            raise RuntimeError(
                'BATSRUS crashed and produced no recoverable plot files.\n'
                f'{crash_info}')
        raise

    # Cleanup scratch .idl files (2-3 GB/year, not kept).
    _cleanup_plot_files(run_dir)

    # Slice off the spin-up pad so the returned dataset starts at the
    # first real minute.
    ds = ds.sel(time=slice(real_start, None))

    if crash_info is not None:
        ds.attrs['batsrus_crashed'] = 1
        ds.attrs['batsrus_crash_info'] = crash_info

    return ds


# ---------------------------------------------------------------------------
# Gap fill / spin-up pad
# ---------------------------------------------------------------------------

def _fill_for_mhd(df):
    """Unbounded interpolation + ffill/bfill so BATSRUS gets a gap-free inflow.

    BATSRUS will not tolerate a single NaN in L1.dat, so we must fill
    every gap regardless of length.
    """
    df = df.copy()

    # Ensure numeric columns exist.
    missing = [c for c in _NUMERIC_COLS if c not in df.columns]
    if missing:
        raise KeyError(f'df_combined missing columns: {missing}')

    # Unbounded time-based interpolation, then edge fill.
    filled = df[_NUMERIC_COLS].interpolate(
        method='time', limit_direction='both')
    filled = filled.ffill().bfill()

    # If a column is still all-NaN (e.g. whole range missing), substitute
    # a quiet-wind default so BATSRUS runs at all.
    _DEFAULTS = {
        'Bx': 0.0, 'By': 0.0, 'Bz': 0.0,
        'Ux': -400.0, 'Uy': 0.0, 'Uz': 0.0,
        'rho': 5.0, 'T': 1.0e5,
    }
    for col, default in _DEFAULTS.items():
        if filled[col].isna().any():
            filled[col] = filled[col].fillna(default)

    df[_NUMERIC_COLS] = filled

    # Floor the proton T for the MHD inflow only (see _MHD_T_FLOOR). The science
    # products are written separately from their own (lower) floor; this clip
    # keeps BATSRUS out of its low-pressure instability regime.
    df['T'] = df['T'].clip(lower=_MHD_T_FLOOR)
    return df


def _prepend_spinup_pad(df, pad):
    """Prepend `pad` duration of constant flow equal to the first real minute.

    Returns (padded_df, real_start_timestamp).
    """
    real_start = df.index[0]
    pad_end = real_start - pd.Timedelta(minutes=1)
    pad_start = real_start - pad
    pad_index = pd.date_range(pad_start, pad_end, freq='1min')

    first_row = df.iloc[0]
    pad_df = pd.DataFrame(
        np.tile(first_row[_NUMERIC_COLS].values, (len(pad_index), 1)),
        index=pad_index, columns=_NUMERIC_COLS)
    padded = pd.concat([pad_df, df[_NUMERIC_COLS]]).sort_index()
    return padded, real_start


# ---------------------------------------------------------------------------
# L1.dat writer
# ---------------------------------------------------------------------------

def _write_l1_dat(df, ref_x_daily, path):
    """Write a BATSRUS #SOLARWINDFILE-compatible L1.dat.

    Format (per Param/EARTH/imf19980504.dat):
        Free-form header lines
        #COORD
        GSM
        #TIMEDELAY
              0.00000
         year mn dy hr mn sc msc bx by bz ux uy uz rho T
        #START
         <data rows>

    BATSRUS's solar-wind-file reader does not consume per-row X position
    — the inflow face is defined by the #GRID xMax in PARAM.in.  We
    encode `ref_x_daily` only in a comment header for traceability.
    """
    # Build spin-up-aware default: if a timestamp predates the first
    # ref_x_daily entry, use that first entry.
    dates_sorted = sorted(ref_x_daily) if ref_x_daily else []
    fallback_x_km = (ref_x_daily[dates_sorted[0]]
                     if dates_sorted else 235.0 * _RE_KM)

    def lookup_x_km(ts):
        d = ts.date()
        if d in ref_x_daily and np.isfinite(ref_x_daily[d]):
            return ref_x_daily[d]
        return fallback_x_km

    lines = []
    lines.append('MIDL L1.dat — written by l1_mhd.mhd_propagation()')
    lines.append('Units: nT, km/s, amu/cc, K')
    lines.append('')
    lines.append('#COORD')
    lines.append('GSM')
    lines.append('')
    lines.append('#TIMEDELAY')
    lines.append('      0.00000')
    lines.append('')
    lines.append(' year mn dy hr mn sc msc bx by bz ux uy uz rho T')
    lines.append('#START')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        it = df.itertuples()
        for row in it:
            ts = row.Index
            f.write(
                f' {ts.year:4d} {ts.month:2d} {ts.day:2d} '
                f'{ts.hour:2d} {ts.minute:2d} {ts.second:2d} {0:3d} '
                f'{row.Bx:8.3f} {row.By:8.3f} {row.Bz:8.3f} '
                f'{row.Ux:9.2f} {row.Uy:9.2f} {row.Uz:9.2f} '
                f'{row.rho:8.3f} {row.T:12.1f}\n')

    # Unused but retained for debugging: record per-day ref X.
    sidecar = path + '.refx'
    with open(sidecar, 'w') as f:
        f.write('# date  X_ref_km\n')
        for d in dates_sorted:
            f.write(f'{d.isoformat()}  {ref_x_daily[d]:.3f}\n')


# ---------------------------------------------------------------------------
# PARAM.in rendering
# ---------------------------------------------------------------------------

def _render_param_in(template_path, out_path, sim_start, sim_end):
    """String-substitute {START_*}/{END_*} placeholders into PARAM.in."""
    with open(template_path, 'r') as f:
        template = f.read()

    rendered = template.format(
        START_YEAR=f'{sim_start.year:04d}',
        START_MONTH=f'{sim_start.month:02d}',
        START_DAY=f'{sim_start.day:02d}',
        START_HOUR=f'{sim_start.hour:02d}',
        START_MINUTE=f'{sim_start.minute:02d}',
        START_SECOND=f'{sim_start.second:02d}',
        END_YEAR=f'{sim_end.year:04d}',
        END_MONTH=f'{sim_end.month:02d}',
        END_DAY=f'{sim_end.day:02d}',
        END_HOUR=f'{sim_end.hour:02d}',
        END_MINUTE=f'{sim_end.minute:02d}',
        END_SECOND=f'{sim_end.second:02d}',
    )
    with open(out_path, 'w') as f:
        f.write(rendered)


# ---------------------------------------------------------------------------
# Run dir staging + subprocess invocation
# ---------------------------------------------------------------------------

def _stage_run_dir(template_dir, work_dir):
    """Stage a per-invocation run dir from the run_mhd template.

    We cannot clobber the shared template (parallel workers would
    collide), so mkdir a fresh 'run' inside work_dir and symlink in the
    heavy files (BATSRUS.exe, restartIN, etc.) while copying PARAM.in
    and L1.dat targets.  To keep this simple and robust, just copy the
    whole template (it's small — no restart dumps in the fresh template).
    """
    run_dir = os.path.join(work_dir, 'run')
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    shutil.copytree(template_dir, run_dir, symlinks=True)
    # Ensure the per-run output tree exists.
    io2 = os.path.join(run_dir, 'GM', 'IO2')
    os.makedirs(io2, exist_ok=True)
    # Clear any pre-existing plot files from the template.
    for f in glob.glob(os.path.join(io2, '1d__mhd_*.idl')):
        os.remove(f)
    for f in glob.glob(os.path.join(io2, '1d__mhd_*.h')):
        os.remove(f)
    for f in glob.glob(os.path.join(io2, '1d__mhd_*.tree')):
        os.remove(f)
    return run_dir


def _run_batsrus(batsrus_exe, run_dir, timeout=None):
    """Launch BATSRUS.exe in run_dir with unlimited stack + OpenMPI on PATH."""
    env = os.environ.copy()
    env['PATH'] = _OPENMPI_PATH + os.pathsep + env.get('PATH', '')

    # Use a shell wrapper so `ulimit -s unlimited` applies to the child.
    # Launch via mpirun because OpenMPI (4.x+) refuses to bootstrap in a
    # SLURM allocation without a PMI-aware launcher (MPI_Init_thread fails
    # on "NULL communicator"). `-np 1` is a single-rank BATSRUS 1D run.
    cmd = f'ulimit -s unlimited && {_MPIRUN} ./BATSRUS.exe'

    def _dump(rc, out_b, err_b):
        """Persist the full BATSRUS output next to the run for diagnosis."""
        try:
            with open(os.path.join(run_dir, 'batsrus_crash.log'), 'w') as f:
                f.write(f'returncode={rc}\n=== STDOUT ===\n'
                        f'{(out_b or b"").decode(errors="replace")}\n'
                        f'=== STDERR ===\n'
                        f'{(err_b or b"").decode(errors="replace")}\n')
        except Exception:
            pass

    try:
        subprocess.run(
            cmd, cwd=run_dir, env=env, shell=True, check=True,
            capture_output=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        _dump(e.returncode, e.stdout, e.stderr)
        tail_out = (e.stdout or b'').decode(errors='replace')[-2000:]
        tail_err = (e.stderr or b'').decode(errors='replace')[-2000:]
        raise RuntimeError(
            f'BATSRUS.exe failed in {run_dir} (rc={e.returncode}):\n'
            f'STDOUT tail:\n{tail_out}\nSTDERR tail:\n{tail_err}')
    except subprocess.TimeoutExpired as e:
        _dump('TIMEOUT', e.stdout, e.stderr)
        raise RuntimeError(f'BATSRUS.exe timed out in {run_dir}')


# ---------------------------------------------------------------------------
# Plot-file parsing
# ---------------------------------------------------------------------------

_PLOT_NAME_RE = re.compile(
    r'1d__mhd_\d+_t(\d{8})_n(\d{8})_pe0000\.idl$')


def _parse_plot_files(run_dir, sim_start):
    """Parse all 1d__mhd_*.idl snapshots into an xr.Dataset.

    Each plot file is Fortran-sequential binary, one record per cell,
    15 doubles per record:
        dx, x, y, z, Rho, Ux, Uy, Uz, Bx, By, Bz, P, jx, jy, jz

    We sort by runtime read from the companion .h file (TIMESIMULATION),
    and map each to an integer minute offset from sim_start.  BATSRUS
    writes one snapshot every 60s in the MIDL PARAM, so the mapping is
    injective.
    """
    io2 = os.path.join(run_dir, 'GM', 'IO2')
    idl_files = sorted(glob.glob(os.path.join(io2, '1d__mhd_*_pe0000.idl')))
    if not idl_files:
        raise RuntimeError(
            f'No BATSRUS plot files found in {io2}.  Run likely failed.')

    # Read every snapshot, skipping any that fail to parse.  When BATSRUS is
    # killed mid-write (e.g. memory pressure under concurrent runs) the final
    # file is truncated; the caller's restart loop then resumes from the last
    # good snapshot, so a bad file must be skipped, not fatal.
    snaps, runtimes_l = [], []
    n_skipped = 0
    for path in idl_files:
        try:
            snap = _read_idl_record_file(path)
            rt = _read_runtime_from_header(path)
        except (ValueError, struct.error, OSError, IndexError) as e:
            n_skipped += 1
            print(f'  Skipping unreadable IDL snapshot {os.path.basename(path)}: {e}')
            continue
        snaps.append(snap)
        runtimes_l.append(rt)

    if not snaps:
        raise RuntimeError(
            f'No parseable BATSRUS plot files in {io2} '
            f'({len(idl_files)} present, all unreadable).')
    if n_skipped:
        print(f'  Parsed {len(snaps)} snapshots, skipped {n_skipped} truncated/unreadable.')

    x_re = snaps[0]['x']
    n_x = len(x_re)
    n_t = len(snaps)
    data = {k: np.empty((n_t, n_x), dtype=np.float32)
            for k in ('Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T')}
    runtimes = np.asarray(runtimes_l, dtype=np.float64)

    for i, snap in enumerate(snaps):
        rho = snap['Rho']
        p   = snap['P']
        # T[K] = P[Pa] / (n[m^-3] * kB).  rho is amu/cc ≈ proton/cc for
        # a pure-proton solar wind approximation; convert to m^-3.
        n_m3 = rho * _AMU_CC_TO_M3
        p_pa = p * _NPA_TO_PA
        with np.errstate(divide='ignore', invalid='ignore'):
            T = np.where(n_m3 > 0, p_pa / (n_m3 * _KB), np.nan)

        data['Bx'][i]  = snap['Bx'].astype(np.float32)
        data['By'][i]  = snap['By'].astype(np.float32)
        data['Bz'][i]  = snap['Bz'].astype(np.float32)
        data['Ux'][i]  = snap['Ux'].astype(np.float32)
        data['Uy'][i]  = snap['Uy'].astype(np.float32)
        data['Uz'][i]  = snap['Uz'].astype(np.float32)
        data['rho'][i] = rho.astype(np.float32)
        data['T'][i]   = T.astype(np.float32)

    # Map runtimes to wall-clock minutes.  BATSRUS runtime is seconds
    # since sim_start.  Plot cadence is 60s — round to nearest minute.
    offsets_min = np.round(runtimes / 60.0).astype(np.int64)
    times = pd.DatetimeIndex(
        [pd.Timestamp(sim_start) + pd.Timedelta(minutes=int(m))
         for m in offsets_min])

    # Drop the initial dump (t=0) if present so every minute has one
    # snapshot and the index is unique; keep later dumps untouched.
    _, unique_idx = np.unique(times, return_index=True)
    unique_idx = np.sort(unique_idx)
    times = times[unique_idx]
    for k in data:
        data[k] = data[k][unique_idx]

    ds = xr.Dataset(
        data_vars={k: (('time', 'x'), v) for k, v in data.items()},
        coords={'time': times, 'x': x_re.astype(np.float32)},
    )
    ds['x'].attrs['units'] = 'Re'
    ds['Bx'].attrs['units'] = 'nT'
    ds['By'].attrs['units'] = 'nT'
    ds['Bz'].attrs['units'] = 'nT'
    ds['Ux'].attrs['units'] = 'km/s'
    ds['Uy'].attrs['units'] = 'km/s'
    ds['Uz'].attrs['units'] = 'km/s'
    ds['rho'].attrs['units'] = 'amu/cc'
    ds['T'].attrs['units'] = 'K'
    return ds


def _read_idl_record_file(path):
    """Read a BATSRUS 1D binary IDL plot file.

    Returns a dict keyed by variable name (x, Rho, Ux, ..., P, jx, jy,
    jz), each value a numpy float64 array of length n_cells.

    Record format: Fortran sequential, one record per cell, each
    record = 15 little-endian float64 values.
    """
    with open(path, 'rb') as f:
        raw = f.read()

    n_bytes = len(raw)
    rows = []
    pos = 0
    expect_reclen = 8 * _IDL_NDOUBLE
    while pos < n_bytes:
        reclen = struct.unpack('<i', raw[pos:pos + 4])[0]
        if reclen != expect_reclen:
            raise ValueError(
                f'Unexpected record length {reclen} at offset {pos} '
                f'in {path} (expected {expect_reclen}).  The IDL file '
                f'may have been written in a format this parser does '
                f'not handle.')
        vals = struct.unpack(
            f'<{_IDL_NDOUBLE}d', raw[pos + 4:pos + 4 + reclen])
        rows.append(vals)
        pos += 4 + reclen + 4

    arr = np.asarray(rows, dtype=np.float64)  # shape (n_cells, 15)
    # A file truncated mid-write (e.g. BATSRUS killed under memory pressure)
    # yields zero/partial records, so arr is not the expected (n_cells, 15).
    # Raise a clean error the parser can catch and skip.
    if arr.ndim != 2 or arr.shape[1] != _IDL_NDOUBLE or arr.shape[0] == 0:
        raise ValueError(
            f'Truncated or malformed IDL file {path}: parsed array shape '
            f'{arr.shape}, expected (n_cells, {_IDL_NDOUBLE}).')
    return {
        'x':   arr[:, _IDL_COL_X],
        'Rho': arr[:, _IDL_COL_RHO],
        'Ux':  arr[:, _IDL_COL_UX],
        'Uy':  arr[:, _IDL_COL_UY],
        'Uz':  arr[:, _IDL_COL_UZ],
        'Bx':  arr[:, _IDL_COL_BX],
        'By':  arr[:, _IDL_COL_BY],
        'Bz':  arr[:, _IDL_COL_BZ],
        'P':   arr[:, _IDL_COL_P],
    }


def _read_runtime_from_header(idl_path):
    """Read TIMESIMULATION from the companion .h header file."""
    h_path = idl_path.replace('_pe0000.idl', '.h')
    if not os.path.exists(h_path):
        # Fall back: parse from filename (tHHMMSS after the 't').
        m = _PLOT_NAME_RE.search(os.path.basename(idl_path))
        if m is None:
            return 0.0
        t_digits = m.group(1)  # 8 digits
        # Conventional BATSRUS encoding: HHMMSSxx?  The t00000100
        # sample corresponds to 60.97s, i.e. "00000100" -> hh=0 mm=1
        # ss=00.  Treat as HH*3600 + MM*60 + SS ignoring trailing 2.
        h = int(t_digits[0:3])
        mnt = int(t_digits[3:5])
        s = int(t_digits[5:7])
        return h * 3600 + mnt * 60 + s
    with open(h_path, 'r') as f:
        in_block = False
        for line in f:
            if line.startswith('#TIMESIMULATION'):
                in_block = True
                continue
            if in_block:
                return float(line.split()[0])
    return 0.0


def _cleanup_plot_files(run_dir):
    """Remove .idl/.h/.tree plot snapshots to reclaim ~2-3 GB/year."""
    io2 = os.path.join(run_dir, 'GM', 'IO2')
    for pattern in ('1d__mhd_*.idl', '1d__mhd_*.h', '1d__mhd_*.tree'):
        for f in glob.glob(os.path.join(io2, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass
