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
        monkeypatch.setitem(dl._archive_listing_cache, prefix, keys)
        days = dl.dscovr_available_days('f1m', ['2026-06'])
        assert days == {'2026-06-01', '2026-06-02'}

    def test_listing_error_returns_none(self, monkeypatch):
        def boom(prefix):
            raise RuntimeError('listing failed')
        monkeypatch.setattr(dl, 'noaa_archive_list', boom)
        dl._archive_listing_cache.pop(
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


# ---------------------------------------------------------------------------
# SOLAR-1 SWiPS L2 (NOAA archive bucket, via the listing cache)
# ---------------------------------------------------------------------------

def _swips_key(day, produced):
    return (f'{dl._SWIPS_L2_PREFIX}/{day[:4]}/{day[4:6]}/'
            f'oe_swips-l2_solar1_s{day}T000000Z_e{day}T235959Z'
            f'_p{produced}Z_pub.nc.gz')


class TestSwips:
    def test_available_days(self, monkeypatch):
        prefix = f'{dl._SWIPS_L2_PREFIX}/2026/09/'
        keys = [_swips_key('20260901', '20260915T032107'),
                _swips_key('20260902', '20260915T031335'),
                prefix + 'unrelated.txt']
        monkeypatch.setitem(dl._archive_listing_cache, prefix, keys)
        assert dl.swips_available_days(['2026-09']) == {'2026-09-01',
                                                         '2026-09-02'}

    def test_listing_error_returns_none(self, monkeypatch):
        def boom(prefix):
            raise RuntimeError('listing failed')
        monkeypatch.setattr(dl, 'noaa_archive_list', boom)
        dl._archive_listing_cache.pop(f'{dl._SWIPS_L2_PREFIX}/2031/01/', None)
        assert dl.swips_available_days(['2031-01']) is None

    def test_download_picks_newest_reprocessing(self, monkeypatch, tmp_path):
        prefix = f'{dl._SWIPS_L2_PREFIX}/2026/09/'
        old = _swips_key('20260901', '20260915T032107')
        new = _swips_key('20260901', '20261020T010000')
        monkeypatch.setitem(dl._archive_listing_cache, prefix, [new, old])
        fetched = []
        monkeypatch.setattr(dl, '_download_url',
                            lambda url, path, timeout=300: fetched.append(url))
        path = dl.download_swips_l2('2026-09-01', str(tmp_path))
        assert fetched == [f'{dl._NOAA_ARCHIVE_BASE}/{new}']
        assert path.endswith('solar1_swips_20260901.nc.gz')

    def test_download_missing_day_returns_none(self, monkeypatch, tmp_path):
        prefix = f'{dl._SWIPS_L2_PREFIX}/2026/09/'
        monkeypatch.setitem(dl._archive_listing_cache, prefix,
                            [_swips_key('20260901', '20260915T032107')])
        assert dl.download_swips_l2('2026-09-02', str(tmp_path)) is None


# ---------------------------------------------------------------------------
# IMAP SDC helpers
# ---------------------------------------------------------------------------

class TestImapSdc:
    _RECORDS = [
        {'start_date': '20260715', 'version': 'v001.0001',
         'file_path': 'imap/mag/l2/2026/07/a_20260715_v001.0001.cdf'},
        {'start_date': '20260715', 'version': 'v001.0002',
         'file_path': 'imap/mag/l2/2026/07/a_20260715_v001.0002.cdf'},
        {'start_date': '20260716', 'version': 'v001.0001',
         'file_path': 'imap/mag/l2/2026/07/a_20260716_v001.0001.cdf'},
    ]

    def test_latest_version_per_day(self):
        files = dl._latest_imap_files(self._RECORDS)
        assert files == {
            '20260715': 'imap/mag/l2/2026/07/a_20260715_v001.0002.cdf',
            '20260716': 'imap/mag/l2/2026/07/a_20260716_v001.0001.cdf',
        }

    def test_available_days(self, monkeypatch):
        monkeypatch.setattr(dl, '_imap_query', lambda *a, **k: self._RECORDS)
        assert dl.imap_available_days('mag', '2026-07-01', '2026-07-31') == {
            '2026-07-15', '2026-07-16'}

    def test_query_error_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise OSError('network down')
        monkeypatch.setattr(dl, '_imap_query', boom)
        assert dl.imap_available_days('swapi', '2026-07-01', '2026-07-31') is None


# ---------------------------------------------------------------------------
# sscweb_position_gsm
# ---------------------------------------------------------------------------

class TestSscwebPosition:
    def test_mean_position(self, monkeypatch):
        from datetime import datetime
        payload = ['Response', {'Result': ['DataResult', {'Data': ['list', [
            ['SatelliteData', {'Coordinates': ['list', [['CoordinateData', {
                'X': ['list', [1.4e6, 1.6e6]],
                'Y': ['list', [1.0e5, 3.0e5]],
                'Z': ['list', [-2.0e4, 0.0]],
            }]]]}]]]}]}]

        class _JsonResponse(_FakeResponse):
            def json(self):
                return payload
        monkeypatch.setattr(dl.requests, 'get',
                            lambda url, headers, timeout: _JsonResponse(''))
        xyz = dl.sscweb_position_gsm('imap', datetime(2026, 7, 15, 11),
                                     datetime(2026, 7, 15, 13))
        assert xyz == pytest.approx((1.5e6, 2.0e5, -1.0e4))

    def test_error_returns_none(self, monkeypatch):
        from datetime import datetime
        def boom(url, headers, timeout):
            raise OSError('network down')
        monkeypatch.setattr(dl.requests, 'get', boom)
        assert dl.sscweb_position_gsm('imap', datetime(2026, 7, 15, 11),
                                      datetime(2026, 7, 15, 13)) is None
