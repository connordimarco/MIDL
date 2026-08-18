"""
l1_angles.py
------------
GSM<->GSE and GSM<->SM rotation angles for the published angle tables
(data/YYYY/MM/YYYYMM_angles.csv), consumed by the website and the
CSEM-MIDL client to rotate MIDL vector data out of its native GSM frame.

Sign conventions (pinned by tests/test_angles.py against full spacepy
conversions -- do not change one without the other):

    v_GSM = Rx(psi)  @ v_GSE        Rx(a) = [[1,   0,    0  ],
                                             [0,  cos a, sin a],
                                             [0, -sin a, cos a]]

    v_SM  = Ry(tilt) @ v_GSM        Ry(a) = [[cos a, 0, -sin a],
                                             [  0,   1,    0  ],
                                             [sin a, 0,  cos a]]

where psi is `gsm_gse_angle_deg` and tilt is `dipole_tilt_deg` (spacepy's
DipoleTilt: positive when the northern dipole end leans sunward). GSE and
GSM share the +X (sunward) axis, so Bx/Ux are GSE<->GSM invariant.
Component form of GSE->GSM:

    Bx_gsm = Bx_gse
    By_gsm =  cos(psi)*By_gse + sin(psi)*Bz_gse
    Bz_gsm = -sin(psi)*By_gse + cos(psi)*Bz_gse

Method: exact spacepy CTrans evaluation on a coarse knot grid (KNOT_MINUTES),
cubic-splined to the 1-minute product grid. At 15-minute knots the spline
error is ~7e-7 deg (~1e-7 nT on a 10 nT field), far below the 4-decimal
storage rounding, so stored values are rounding-limited.

Timestamps are naive UTC on a fixed 1-minute grid (matching every other
MIDL product). Leap seconds are invisible at this cadence; spacepy handles
UTC->TT/UT1 internally, so the angle *values* are leap-second-correct even
though the index is naive. Angle values depend on the IGRF dipole: this
module requires the locally-built spacepy with IGRF generation
IGRF_GENERATION and refuses to compute past IGRF_VALID_UNTIL, where
spacepy silently clamps coefficients instead of erroring.

Heavy imports (spacepy, scipy) are function-local: midl_pipeline/__init__
is deliberately lazy so the realtime tick never pays for them.
"""
import calendar
import datetime as dt

import numpy as np
import pandas as pd

# IGRF generation the local spacepy build carries; recorded in the manifest
# via the angles sidecar so published files are traceable to their dipole
# model. Bump alongside a spacepy/IGRF upgrade and extend IGRF_VALID_UNTIL.
IGRF_GENERATION = 14
IGRF_VALID_UNTIL = dt.datetime(2030, 1, 1)

KNOT_MINUTES = 15
ANGLE_DECIMALS = 4
COLUMNS = ('gsm_gse_angle_deg', 'dipole_tilt_deg')


def rx(deg):
    """Rotation matrix Rx(deg) such that v_GSM = Rx(psi) @ v_GSE."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def ry(deg):
    """Rotation matrix Ry(deg) such that v_SM = Ry(tilt) @ v_GSM."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def exact_angles(when):
    """(psi_deg, tilt_deg) at one naive-UTC datetime, via full spacepy CTrans.

    Slow (~10 ms/call) -- the exact reference used at spline knots and in
    validation tests. month_angles() is the production path.
    """
    from spacepy.ctrans import CTrans

    c = CTrans(when)
    c.calcCoreTransforms()
    c.calcMagTransforms()
    m = c['Transform']['ECIMOD_GSM'] @ c['Transform']['GSE_ECIMOD']
    psi = np.degrees(np.arctan2(m[1, 2], m[1, 1]))
    return float(psi), float(c['DipoleTilt'])


def month_angles(year, month, knot_minutes=KNOT_MINUTES):
    """One month of per-minute angles as a DataFrame indexed by timestamp.

    Exact spacepy values at `knot_minutes` spacing, cubic-splined to the
    1-minute grid. Knots extend 2*knot_minutes past both month edges so the
    spline's end conditions sit outside the month and adjacent months join
    seamlessly.
    """
    from scipy.interpolate import CubicSpline

    start = dt.datetime(year, month, 1)
    if start >= IGRF_VALID_UNTIL:
        raise ValueError(
            f'{year}-{month:02d} is beyond IGRF-{IGRF_GENERATION} validity '
            f'({IGRF_VALID_UNTIL:%Y-%m}); spacepy would silently clamp '
            'coefficients. Upgrade spacepy/IGRF and bump IGRF_GENERATION / '
            'IGRF_VALID_UNTIL in l1_angles.py.')

    n = calendar.monthrange(year, month)[1] * 1440
    knots = np.arange(-2 * knot_minutes, n + 2 * knot_minutes + 1,
                      knot_minutes, dtype=float)
    vals = np.array([exact_angles(start + dt.timedelta(minutes=float(m)))
                     for m in knots])
    mins = np.arange(n, dtype=float)
    # unwrap guards the atan2 branch cut; psi stays within ~+/-35 deg in
    # practice, so this is belt-and-braces for the spline only.
    psi = CubicSpline(knots, np.unwrap(vals[:, 0], period=360))(mins)
    tilt = CubicSpline(knots, vals[:, 1])(mins)

    df = pd.DataFrame(
        {COLUMNS[0]: np.round((psi + 180) % 360 - 180, ANGLE_DECIMALS),
         COLUMNS[1]: np.round(tilt, ANGLE_DECIMALS)},
        index=pd.date_range(start, periods=n, freq='min'))
    df.index.name = 'timestamp'
    return df
