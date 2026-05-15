import sys
from pathlib import Path

import pytest

# midl_pipeline is not pip-installed; make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Make tests/ importable so test modules can `from helpers import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import FIXTURES_DIR  # noqa: E402


@pytest.fixture
def fixture_dir():
    return FIXTURES_DIR


@pytest.fixture
def ace_only_day():
    return FIXTURES_DIR / "raw" / "2003_01_01"


@pytest.fixture
def two_sat_day():
    return FIXTURES_DIR / "raw" / "2005_01_01"


@pytest.fixture
def three_sat_day():
    return FIXTURES_DIR / "raw" / "2020_06_15"


@pytest.fixture
def plasma_gap_day():
    return FIXTURES_DIR / "raw" / "2024_06_15"


@pytest.fixture
def tmp_output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return out
