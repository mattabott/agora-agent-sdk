"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def fixed_tick() -> int:
    """A deterministic tick value used across tests."""
    return 12345
