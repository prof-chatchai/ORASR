"""Repository-local constants for ORASR."""

from typing import Final, Tuple

PACKAGE_NAME: Final[str] = "orasr"
PACKAGE_VERSION: Final[str] = "1.3.0"
FRAMEWORK_NAME: Final[str] = "ORASR"
SUPPORTED_PYTHON_VERSION_MIN: Final[str] = "3.9"

ISO_DATE_FORMAT: Final[str] = "%Y-%m-%d"
ISO_DATETIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"
ISO_TIMEZONE_UTC: Final[str] = "UTC"

DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95
DEFAULT_EPSILON: Final[float] = 1e-9
DEFAULT_DPI: Final[int] = 300
DEFAULT_FIGURE_SIZE: Final[Tuple[float, float]] = (10.0, 6.0)
