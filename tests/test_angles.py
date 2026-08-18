"""Tests for l1_angles: sign conventions pinned against full spacepy
conversions, spline accuracy, month-boundary continuity, IGRF guard.

These are the ground-truth anchors for every downstream rotation
implementation (website propagate.js, CSEM-MIDL _coords.py) -- if a sign
here changes, those must change with it.
"""
import datetime as dt

import numpy as np
import pytest

from midl_pipeline.l1_angles import (
    IGRF_VALID_UNTIL, exact_angles, month_angles, rx, ry,
)

# Solstice/equinox epochs where psi and tilt take extreme values, plus
# off-hours to exercise the diurnal term.
EPOCHS = [
    dt.datetime(2024, 3, 20, 0), dt.datetime(2024, 6, 21, 12),
    dt.datetime(2024, 9, 22, 18), dt.datetime(2024, 12, 21, 6),
    dt.datetime(2026, 6, 10, 20, 35),
]


def _spacepy_convert(vec, src, dst, when):
    from spacepy.coordinates import Coords
    from spacepy.time import Ticktock
    c = Coords([list(vec)], src, 'car', use_irbem=False)
    c.ticks = Ticktock([when.strftime('%Y-%m-%dT%H:%M:%S')], 'ISO')
    return np.asarray(c.convert(dst, 'car').data[0])


class TestSignConventions:
    def test_x_axis_shared_gse_gsm(self):
        for when in EPOCHS:
            out = _spacepy_convert([1, 0, 0], 'GSE', 'GSM', when)
            assert np.abs(out - [1, 0, 0]).max() < 1e-12

    def test_rx_psi_matches_full_gse_to_gsm(self):
        rng = np.random.default_rng(42)
        for when in EPOCHS:
            psi, _ = exact_angles(when)
            for vec in rng.normal(size=(5, 3)):
                expect = _spacepy_convert(vec, 'GSE', 'GSM', when)
                got = rx(psi) @ vec
                assert np.abs(got - expect).max() < 1e-9

    def test_ry_tilt_matches_full_gsm_to_sm(self):
        rng = np.random.default_rng(43)
        for when in EPOCHS:
            _, tilt = exact_angles(when)
            for vec in rng.normal(size=(5, 3)):
                expect = _spacepy_convert(vec, 'GSM', 'SM', when)
                got = ry(tilt) @ vec
                assert np.abs(got - expect).max() < 1e-9

    def test_angle_ranges_plausible(self):
        # psi extreme near equinox, tilt extreme near solstice
        psi_eq, _ = exact_angles(dt.datetime(2024, 3, 20, 0))
        _, tilt_sol = exact_angles(dt.datetime(2024, 12, 21, 6))
        assert 25 < abs(psi_eq) < 36
        assert 25 < abs(tilt_sol) < 36


class TestMonthAngles:
    def test_spline_matches_exact(self):
        df = month_angles(2026, 6)
        rng = np.random.default_rng(7)
        for i in rng.integers(0, len(df), size=4):
            t = df.index[int(i)].to_pydatetime()
            psi, tilt = exact_angles(t)
            assert abs(df.iloc[int(i), 0] - psi) < 1e-3
            assert abs(df.iloc[int(i), 1] - tilt) < 1e-3

    def test_month_boundary_continuity(self):
        a = month_angles(2026, 5)
        b = month_angles(2026, 6)
        # one minute apart across the boundary; angles drift < 0.02 deg/min
        assert abs(a.iloc[-1, 0] - b.iloc[0, 0]) < 0.05
        assert abs(a.iloc[-1, 1] - b.iloc[0, 1]) < 0.05

    def test_row_count_and_grid(self):
        df = month_angles(2024, 2)          # leap February
        assert len(df) == 29 * 1440
        assert df.index[0] == dt.datetime(2024, 2, 1)
        assert (df.index[1] - df.index[0]).total_seconds() == 60

    def test_igrf_ceiling_refused(self):
        with pytest.raises(ValueError, match='IGRF'):
            month_angles(IGRF_VALID_UNTIL.year, 1)
