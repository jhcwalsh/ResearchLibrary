import pytest
from dotenv import load_dotenv

load_dotenv()

# Known stable items in the library used across integration tests.
# All three have an attached PDF that Zotero has indexed; don't swap them for
# keyword-search results, whose top hits are often older papers with no PDF.
KNOWN_PAPER_KEY = "4A984BQC"  # "The Limits of Optimization in Strategic Multi-Asset Allocation"
KNOWN_PAPER_TITLE = "The Limits of Optimization"
KNOWN_FULLTEXT_KEYS = [
    "4A984BQC",  # The Limits of Optimization in Strategic Multi-Asset Allocation
    "E27ZTLWH",  # Robust Optimization of Strategic and Tactical Asset Allocation
    "27S8STN4",  # Deep Reinforcement Learning for Optimal Portfolio Allocation
]


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests that call the live Zotero API")
