import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)


RESEARCHERS_CSV_URL = os.getenv("RESEARCHERS_CSV_URL")
RESEARCHERS_METADATA_URL = os.getenv("RESEARCHERS_METADATA_URL")

SPENDING_CSV_URL = os.getenv("SPENDING_CSV_URL")
SPENDING_METADATA_URL = os.getenv("SPENDING_METADATA_URL")

EDUCATION_CSV_URL = os.getenv("EDUCATION_CSV_URL")
EDUCATION_METADATA_URL = os.getenv("EDUCATION_METADATA_URL")

OWID_USER_AGENT = os.getenv("OWID_USER_AGENT", "Our World In Data data fetch/1.0")

# Where to save figures produced by EDA
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def validate_config() -> None:
    """
    Validate that the mandatory dataset URLs are available.
    Raises ValueError with a helpful message if anything critical is missing.
    """

    missing = []
    for name, value in [
        ("RESEARCHERS_CSV_URL", RESEARCHERS_CSV_URL),
        ("SPENDING_CSV_URL", SPENDING_CSV_URL),
        ("EDUCATION_CSV_URL", EDUCATION_CSV_URL),
    ]:
        if not value:
            missing.append(name)

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Create a .env file (e.g. by copying .env.example) and set the dataset URLs."
        )


