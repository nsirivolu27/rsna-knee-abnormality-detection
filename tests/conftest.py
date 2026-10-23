"""Shared pytest fixtures."""

import pytest

from src.fixture import create_synthetic_dataset


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Return a fresh synthetic dataset rooted in pytest's temporary directory."""
    return create_synthetic_dataset(tmp_path / "synthetic")
