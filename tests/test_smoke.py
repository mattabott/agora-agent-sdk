"""Smoke tests: package imports work and version constants are defined."""
import agora_core
import agora_agent_sdk


def test_agora_core_imports():
    assert agora_core.__version__ == "0.1.0"
    assert agora_core.ACTION_SCHEMA_VERSION == 1


def test_agora_agent_sdk_imports():
    assert agora_agent_sdk.__version__ == "0.1.0"
    assert agora_agent_sdk.ACTION_SCHEMA_VERSION == 1
