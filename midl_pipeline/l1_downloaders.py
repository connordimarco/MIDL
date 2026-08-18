"""
l1_downloaders.py
-----------------
Download helpers for raw L1 data files.

Three sources are supported:
  - CDAWeb  (ACE, WIND, DSCOVR orbit) via the pyspedas CDAWeb client.
  - NOAA archive S3 bucket (DSCOVR 1-min plasma + mag) via direct HTTP
    requests. (Until ~2026-02 this lived at www.ngdc.noaa.gov/dscovr/data/,
    which now redirects to a JS bucket explorer.)
  - NOAA NCEI HAPI (SOLAR-1 mag + orbit) via REST CSV downloads.

Also provides the upstream-availability helpers (cdaweb_available_days,
dscovr_available_days, hapi_coverage) used by the gapfill script to skip
day-fetches for data that does not exist upstream.

Files are written to local scratch directories inside cdf_temp/ and are
expected to be cleaned up by the calling pipeline after processing.
"""
import os
import re
from datetime import datetime, timedelta
import time

import requests


_HAPI_BASE = (
    'https://www.ncei.noaa.gov/cloud-access/space-weather-portal'
    '/api/v1/hapi/data'
)
_HAPI_INFO_BASE = (
    'https://www.ncei.noaa.gov/cloud-access/space-weather-portal'
    '/api/v1/hapi/info'
)

# NOAA archive bucket holding the DSCOVR 1-min products (formerly NGDC).
_NOAA_ARCHIVE_BASE = 'https://archive.data.noaa.gov/satellite-spaceweather'
_DSCOVR_PRODUCT_PREFIX = {
    'f1m': 'DSCOVR/DSCOVR/FC/f1m',
    'm1m': 'DSCOVR/DSCOVR/MAG/m1m',
}
# Month-listing cache: {S3 prefix -> list of keys}. Per-process, so repeated
# same-month day downloads cost one listing request instead of one per day.
_dscovr_listing_cache = {}


def _parse_s3_keys(xml_text):
    """(keys, continuation_token) from an S3 ListObjectsV2 XML response."""
    keys = re.findall(r'<Key>([^<]+)</Key>', xml_text)
    m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>',
                  xml_text)
    return keys, (m.group(1) if m else None)


def noaa_archive_list(prefix, timeout=60):
    """List all object keys under `prefix` in the NOAA archive bucket.

    Raises RuntimeError if the listing cannot be fetched.
    """
    keys = []
    token = None
    while True:
        params = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if token:
            params['continuation-token'] = token
        try:
            resp = requests.get(_NOAA_ARCHIVE_BASE, params=params,
                                timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f'Failed to list NOAA archive prefix {prefix}: {e}') from e
        page_keys, token = _parse_s3_keys(resp.text)
        keys.extend(page_keys)
        if not token:
            return keys


# ---------------------------------------------------------------------------
# Upstream availability (used by the gapfill script to prune futile fetches)
# ---------------------------------------------------------------------------

_CDAWEB_DATA_BASE = 'https://cdaweb.gsfc.nasa.gov/sp_phys/data'


def cdaweb_available_days(dataset_dir, years, timeout=60):
    """Days ('YYYY-MM-DD') with a file in a CDAWeb dataset directory.

    One listing request per year. `dataset_dir` is the path below sp_phys/data,
    e.g. 'ace/mag/level_2_cdaweb/mfi_h0'. Returns None on any fetch error so
    the caller can fail open (attempt the download rather than skip it).
    """
    days = set()
    for year in years:
        url = f'{_CDAWEB_DATA_BASE}/{dataset_dir}/{year}/'
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                continue          # year directory not created yet -> no days
            resp.raise_for_status()
        except Exception as e:
            print(f'  WARNING: could not list {url}: {e}')
            return None
        for d in re.findall(r'_(\d{8})_v\d+\.cdf', resp.text):
            days.add(f'{d[:4]}-{d[4:6]}-{d[6:]}')
    return days


def dscovr_available_days(product, months):
    """Days ('YYYY-MM-DD') with a DSCOVR file for `product` ('f1m'/'m1m').

    One bucket listing per month in `months` (iterable of 'YYYY-MM').
    Listings are cached, so the subsequent per-day downloads reuse them.
    Returns None on any fetch error (caller fails open).
    """
    days = set()
    pattern = re.compile(
        rf'oe_{re.escape(product)}_dscovr_s(\d{{8}})\d+_e\d+_p\d+_pub\.nc\.gz$')
    for month in months:
        y, m = month.split('-')
        prefix = f'{_DSCOVR_PRODUCT_PREFIX[product]}/{y}/{m}/'
        try:
            if prefix not in _dscovr_listing_cache:
                _dscovr_listing_cache[prefix] = noaa_archive_list(prefix)
        except RuntimeError as e:
            print(f'  WARNING: {e}')
            return None
        for key in _dscovr_listing_cache[prefix]:
            match = pattern.search(key)
            if match:
                d = match.group(1)
                days.add(f'{d[:4]}-{d[4:6]}-{d[6:]}')
    return days


def hapi_coverage(dataset, timeout=60):
    """(startDate, stopDate) as 'YYYY-MM-DD' strings from a HAPI info request.

    Returns None on any error (caller fails open).
    """
    try:
        resp = requests.get(_HAPI_INFO_BASE, params={'dataset': dataset},
                            timeout=timeout)
        resp.raise_for_status()
        info = resp.json()
        return info['startDate'][:10], info['stopDate'][:10]
    except Exception as e:
        print(f'  WARNING: HAPI info for {dataset} failed: {e}')
        return None


def download_cdaweb_files(cda, datasets, trange_start, trange_end, data_dir,
                          label='CDAWeb', max_attempts=3, retry_delay=30):
    """Download a set of CDAWeb datasets for a given time range.

    Parameters
    ----------
    cda : pyspedas.CDAWeb
        Authenticated CDAWeb client.
    datasets : list[str]
        Dataset IDs to request (e.g. ['AC_H0_MFI', 'WI_H0_MFI']).
    trange_start, trange_end : str
        ISO date strings 'YYYY-MM-DD' (or 'YYYY-MM-DD/HH:MM:SS').
    data_dir : str
        Local directory to write downloaded files into.
    label : str
        Human-readable label for log messages.
    max_attempts : int
        Maximum number of query/download attempts before giving up.
    retry_delay : int | float
        Seconds to wait between failed attempts.

    Returns
    -------
    urllist : list[str]
        List of URLs that were downloaded (empty if nothing found).
    """
    # Make sure the target folder exists before downloading.
    os.makedirs(data_dir, exist_ok=True)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            # Ask CDAWeb for files in the requested time window.
            print(f'Querying {label} (attempt {attempt}/{max_attempts})...')
            urllist = cda.get_filenames(datasets, trange_start, trange_end)

            if not urllist:
                # Nothing found is okay; caller can decide what to do.
                print('No files found.')
                return []

            # Pull files to local scratch storage.
            print(f'Downloading {len(urllist)} files...')
            cda.cda_download(urllist, local_dir=data_dir, download_only=True)
            return urllist
        except Exception as exc:
            last_error = exc
            print(f'WARNING: {label} download attempt {attempt} failed: {exc}')
            if attempt < max_attempts:
                print(f'Waiting {retry_delay} seconds before retrying...')
                time.sleep(retry_delay)

    raise RuntimeError(
        f'Failed to download {label} after {max_attempts} attempts: {last_error}'
    ) from last_error


def download_position_cdaweb_files(cda, day, data_dir,
                                   max_attempts=3, retry_delay=30):
    """Download a narrow (~2 h) noon window of position data for one day.

    Fetches ACE MFI, WIND MFI, and DSCOVR orbit CDF files.
    Only 11:00–13:00 UT is requested because a stable mean position is all
    that downstream propagation needs.

    Parameters
    ----------
    cda : pyspedas.CDAWeb
    day : str  ('YYYY-MM-DD')
    data_dir : str
    max_attempts : int
    retry_delay : int | float

    Returns
    -------
    urllist : list[str]
    """
    # Position file generation only needs a 2-hour noon window.
    dt_day = datetime.strptime(day, '%Y-%m-%d')
    trange_start = (dt_day.replace(hour=11, minute=0, second=0)
                    ).strftime('%Y-%m-%d/%H:%M:%S')
    trange_end = (dt_day.replace(hour=13, minute=0, second=0)
                  ).strftime('%Y-%m-%d/%H:%M:%S')
    datasets = ['AC_H0_MFI', 'WI_H0_MFI', 'DSCOVR_ORBIT_PRE']

    return download_cdaweb_files(
        cda,
        datasets,
        trange_start,
        trange_end,
        data_dir,
        label='Position Data (11:00-13:00 UT)',
        max_attempts=max_attempts,
        retry_delay=retry_delay,
    )


def download_dscovr_ngdc(day, data_dir, products=('f1m', 'm1m')):
    """Download DSCOVR 1-minute products from the NOAA archive bucket.

    The NGDC portal (www.ngdc.noaa.gov/dscovr/data/YYYY/MM/) went away in
    early 2026; the files now live in the NOAA archive S3 bucket under
    DSCOVR/DSCOVR/{FC/f1m,MAG/m1m}/YYYY/MM/ with unchanged filenames.
    Month listings are cached per process, so re-downloading a run of days
    in the same month costs one listing request total.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    data_dir : str
    products : tuple[str]
        Product codes to fetch.  Defaults: 'f1m' (plasma), 'm1m' (mag).

    Returns
    -------
    paths : dict[str, str]
        Maps product code -> local file path for each successfully downloaded file.

    Raises
    ------
    RuntimeError
        If a bucket listing cannot be fetched.
    """
    dt = datetime.strptime(day, '%Y-%m-%d')
    date_str = dt.strftime('%Y%m%d')

    os.makedirs(data_dir, exist_ok=True)

    paths = {}
    for product in products:
        prefix = f'{_DSCOVR_PRODUCT_PREFIX[product]}/{dt.year}/{dt.month:02d}/'
        if prefix not in _dscovr_listing_cache:
            _dscovr_listing_cache[prefix] = noaa_archive_list(prefix)

        # Filenames include run-specific tags, so we match by regex.
        pattern = re.compile(
            rf'oe_{re.escape(product)}_dscovr_s{date_str}\d+'
            rf'_e{date_str}\d+_p\d+_pub\.nc\.gz$')
        key = next((k for k in _dscovr_listing_cache[prefix]
                    if pattern.search(k)), None)
        if key is None:
            print(f"  WARNING: No NGDC {product} file found for {day}.")
            continue

        file_url = f'{_NOAA_ARCHIVE_BASE}/{key}'
        local_path = os.path.join(
            data_dir, f"dscovr_{product}_{date_str}.nc.gz")

        try:
            # Stream download to avoid loading the whole file into memory.
            r = requests.get(file_url, timeout=120, stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            paths[product] = local_path
            print(f"  Downloaded {os.path.basename(key)} -> {local_path}")
        except Exception as e:
            print(f"  WARNING: Failed to download {file_url}: {e}")

    return paths


def _download_hapi_csv(dataset, parameters, start, stop, data_dir, local_name,
                       max_attempts=3, retry_delay=30):
    """Download a HAPI dataset as CSV and write to a local file.

    Returns the local file path, or None if the dataset had no rows.
    """
    url = (
        f'{_HAPI_BASE}?dataset={dataset}'
        f'&start={start}&stop={stop}'
        f'&parameters={parameters}&format=csv'
    )
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, local_name)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            lines = r.text.strip().split('\n')
            if len(lines) <= 1:
                print(f'  SOLAR-1 HAPI: no data rows for {dataset} '
                      f'({start} – {stop}).')
                return None
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
            print(f'  Downloaded HAPI {dataset} -> {local_path}')
            return local_path
        except Exception as exc:
            last_error = exc
            print(f'  WARNING: HAPI {dataset} attempt {attempt} failed: {exc}')
            if attempt < max_attempts:
                time.sleep(retry_delay)

    print(f'  WARNING: HAPI {dataset} failed after {max_attempts} attempts: '
          f'{last_error}')
    return None


def download_solar1_hapi(day, data_dir, max_attempts=3, retry_delay=30):
    """Download SOLAR-1 1-minute magnetometer data from NCEI HAPI.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    data_dir : str
    max_attempts : int
    retry_delay : int | float

    Returns
    -------
    str or None
        Local CSV path, or None if no data was available.
    """
    dt = datetime.strptime(day, '%Y-%m-%d')
    next_day = (dt + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
    start = f'{day}T00:00:00Z'
    date_str = dt.strftime('%Y%m%d')

    return _download_hapi_csv(
        dataset='mag-l3_solar1',
        parameters='b_gsm_min_x,b_gsm_min_y,b_gsm_min_z',
        start=start,
        stop=next_day,
        data_dir=data_dir,
        local_name=f'solar1_mag_{date_str}.csv',
        max_attempts=max_attempts,
        retry_delay=retry_delay,
    )


def download_solar1_position_hapi(day, data_dir,
                                  max_attempts=3, retry_delay=30):
    """Download SOLAR-1 predicted orbit for a noon window from NCEI HAPI.

    Fetches a narrow 11:00–13:00 UT window to match the existing position
    download pattern used for ACE/WIND/DSCOVR.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    data_dir : str
    max_attempts : int
    retry_delay : int | float

    Returns
    -------
    str or None
        Local CSV path, or None if no data was available.
    """
    dt = datetime.strptime(day, '%Y-%m-%d')
    start = f'{day}T11:00:00Z'
    stop = f'{day}T13:00:00Z'
    date_str = dt.strftime('%Y%m%d')

    return _download_hapi_csv(
        dataset='orb-pr_solar1',
        parameters='sat_x_gsm,sat_y_gsm,sat_z_gsm',
        start=start,
        stop=stop,
        data_dir=data_dir,
        local_name=f'solar1_pos_{date_str}.csv',
        max_attempts=max_attempts,
        retry_delay=retry_delay,
    )
