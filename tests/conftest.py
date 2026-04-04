import pytest
from dotenv import load_dotenv

load_dotenv()

# Known stable item in the library used across integration tests
KNOWN_PAPER_KEY = "4A984BQC"  # "The Limits of Optimization in Strategic Multi-Asset Allocation"
KNOWN_PAPER_TITLE = "The Limits of Optimization"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests that call the live Zotero API")
