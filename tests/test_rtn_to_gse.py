"""Tests for the RTN -> GSE rotation used for IMAP SWAPI velocities."""
import numpy as np
import pandas as pd
import pytest

from midl_pipeline.l1_pipeline import rtn_to_gse


TIMES = pd.date_range('2026-01-01', '2026-12-31', freq='7D')


def _basis(times):
    n = len(times)
    eye = np.eye(3)
    return [rtn_to_gse(times, np.tile(eye[i], (n, 1))) for i in range(3)]


def test_radial_is_minus_x():
    r_hat, _, _ = _basis(TIMES)
    assert np.allclose(r_hat, [-1.0, 0.0, 0.0])


def test_basis_is_orthonormal_and_right_handed():
    r_hat, t_hat, n_hat = _basis(TIMES)
    for v in (r_hat, t_hat, n_hat):
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose((t_hat * n_hat).sum(1), 0.0)
    assert np.allclose(np.cross(r_hat, t_hat), n_hat)


def test_tangential_tracks_orbital_motion_within_axis_tilt():
    # T ~ -Y_GSE (direction of Earth's orbital motion), N ~ +Z_GSE, off by at
    # most the 7.25 deg tilt of the solar spin axis to the ecliptic normal.
    _, t_hat, n_hat = _basis(TIMES)
    tilt = np.degrees(np.arccos(np.clip(n_hat[:, 2], -1, 1)))
    assert np.all(t_hat[:, 1] < -0.99)
    assert tilt.max() == pytest.approx(7.25, abs=0.05)
    assert tilt.min() < 0.5          # the tilt cycles through zero yearly


def test_speed_preserved():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 50, (len(TIMES), 3)) + [400.0, 0, 0]
    out = rtn_to_gse(TIMES, v)
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(v, axis=1))
