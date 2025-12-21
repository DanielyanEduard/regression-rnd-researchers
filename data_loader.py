from typing import Dict, Tuple
import ssl
import certifi

import pandas as pd
import requests

from config import (
    EDUCATION_CSV_URL,
    EDUCATION_METADATA_URL,
    OWID_USER_AGENT,
    RESEARCHERS_CSV_URL,
    RESEARCHERS_METADATA_URL,
    SPENDING_CSV_URL,
    SPENDING_METADATA_URL,
    validate_config,
)


HEADERS = {"User-Agent": OWID_USER_AGENT}


def _read_csv(url: str) -> pd.DataFrame:
    """Read CSV from URL with proper SSL context."""
    # Use requests with SSL verification to download, then load with pandas
    response = requests.get(url, headers=HEADERS, timeout=30, verify=certifi.where())
    response.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(response.text))


def _read_metadata(url: str) -> Dict:
    response = requests.get(url, headers=HEADERS, timeout=30, verify=certifi.where())
    response.raise_for_status()
    return response.json()


def load_researcher_data() -> Tuple[pd.DataFrame, Dict]:
    """Load researcher dataset and metadata."""

    df = _read_csv(RESEARCHERS_CSV_URL)
    metadata = _read_metadata(RESEARCHERS_METADATA_URL)
    return df, metadata


def load_spending_data() -> Tuple[pd.DataFrame, Dict]:
    """Load R&D spending dataset and metadata."""

    df = _read_csv(SPENDING_CSV_URL)
    metadata = _read_metadata(SPENDING_METADATA_URL)
    return df, metadata


def load_education_data() -> Tuple[pd.DataFrame, Dict]:
    """Load education spending dataset and metadata."""

    df = _read_csv(EDUCATION_CSV_URL)
    metadata = _read_metadata(EDUCATION_METADATA_URL)
    return df, metadata


def load_all_datasets():
    """
    Convenience function to validate configuration and load all three datasets.
    Returns:
        researcher_df, spending_df, education_df
    """

    validate_config()

    researcher_df, _ = load_researcher_data()
    spending_df, _ = load_spending_data()
    education_df, _ = load_education_data()

    return researcher_df, spending_df, education_df


