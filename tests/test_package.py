"""Basic package tests."""

import pytest

from electricity_forecast import __version__


def test_version() -> None:
    """Test package version is defined."""
    assert __version__ == "0.1.0"
