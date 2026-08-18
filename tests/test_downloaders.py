"""Tests for the upstream-availability helpers in l1_downloaders."""
import pytest

from midl_pipeline import l1_downloaders as dl


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')


# ---------------------------------------------------------------------------
# _parse_s3_keys
# ---------------------------------------------------------------------------

class TestParseS3Keys:
    def test_keys_and_no_token(self):
        xml = (
            '<ListBucketResult>'
            '<Key>DSCOVR/DSCOVR/FC/f1m/2026/06/a.nc.gz</Key>'
            '<Key>DSCOVR/DSCOVR/FC/f1m/2026/06/b.nc.gz</Key>'
            '</ListBucketResult>'
        )
        keys, token = dl._parse_s3_keys(xml)
        assert len(keys) == 2
        assert keys[0].endswith('a.nc.gz')
        assert token is None

    def test_continuation_token(self):
        xml = (
            '<Key>k1</Key>'
            '<NextContinuationToken>abc==</NextContinuationToken>'
        )
        keys, token = dl._parse_s3_keys(xml)
        assert keys == ['k1']
        assert token == 'abc=='


# ---------------------------------------------------------------------------
# cdaweb_available_days
# ---------------------------------------------------------------------------

class TestCdawebAvailableDays:
    def test_parses_listing(self, monkeypatch):
        html = (
            '<a href="ac_h0_swe_20240708_v11.cdf">..</a>'
            '<a href="ac_h0_swe_20240709_v11.cdf">..</a>'
        )
        monkeypatch.setattr(dl.requests, 'get',
                            lambda url, timeout: _FakeResponse(html))
        days = dl.cdaweb_available_days('ace/swepam/level_2_cdaweb/swe_h0',
                                        [2024])
        assert days == {'2024-07-08', '2024-07-09'}

    def test_missing_year_dir_is_empty(self, monkeypatch):
        monkeypatch.setattr(dl.requests, 'get',
                            lambda url, timeout: _FakeResponse('', 404))
        assert dl.cdaweb_available_days('x/y', [2030]) == set()

    def test_fetch_error_returns_none(self, monkeypatch):
        def boom(url, timeout):
            raise OSError('network down')
        monkeypatch.setattr(dl.requests, 'get', boom)
        assert dl.cdaweb_available_days('x/y', [2024]) is None


# ---------------------------------------------------------------------------
# dscovr_available_days (via the module listing cache -- no network)
# ---------------------------------------------------------------------------

class TestDscovrAvailableDays:
    def test_matches_product_days(self, monkeypatch):
        prefix = f"{dl._DSCOVR_PRODUCT_PREFIX['f1m']}/2026/06/"
        keys = [
            f'{prefix}oe_f1m_dscovr_s20260601000000_e20260601235959'
            '_p20260602022046_pub.nc.gz',
            f'{prefix}oe_f1m_dscovr_s20260602000000_e20260602235959'
            '_p20260603021702_pub.nc.gz',
            # different product must not count for f1m
            f'{prefix}oe_m1m_dscovr_s20260603000000_e20260603235959'
            '_p20260604021954_pub.nc.gz',
        ]
        monkeypatch.setitem(dl._dscovr_listing_cache, prefix, keys)
        days = dl.dscovr_available_days('f1m', ['2026-06'])
        assert days == {'2026-06-01', '2026-06-02'}

    def test_listing_error_returns_none(self, monkeypatch):
        def boom(prefix):
            raise RuntimeError('listing failed')
        monkeypatch.setattr(dl, 'noaa_archive_list', boom)
        dl._dscovr_listing_cache.pop(
            f"{dl._DSCOVR_PRODUCT_PREFIX['m1m']}/2031/01/", None)
        assert dl.dscovr_available_days('m1m', ['2031-01']) is None


# ---------------------------------------------------------------------------
# hapi_coverage
# ---------------------------------------------------------------------------

class TestHapiCoverage:
    def test_parses_dates(self, monkeypatch):
        class _JsonResponse(_FakeResponse):
            def json(self):
                return {'startDate': '2026-04-01T00:00:00.000Z',
                        'stopDate': '2026-08-17T23:59:00.000Z'}
        monkeypatch.setattr(dl.requests, 'get',
                            lambda url, params, timeout: _JsonResponse(''))
        assert dl.hapi_coverage('mag-l3_solar1') == ('2026-04-01', '2026-08-17')

    def test_error_returns_none(self, monkeypatch):
        def boom(url, params, timeout):
            raise OSError('network down')
        monkeypatch.setattr(dl.requests, 'get', boom)
        assert dl.hapi_coverage('mag-l3_solar1') is None
