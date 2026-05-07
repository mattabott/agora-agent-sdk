import sys
from unittest.mock import patch

import pytest

from agora_agent_sdk.cli import _build_parser, main


def test_parser_requires_name_seed_sex():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing required


def test_parser_minimal():
    parser = _build_parser()
    ns = parser.parse_args([
        "--server", "http://x", "--name", "Maya",
        "--seed", "curious", "--sex", "F",
    ])
    assert ns.name == "Maya"
    assert ns.sex == "F"
    assert ns.llm_decide_interval == 120
    assert ns.ring_buffer == 30


def test_parser_no_llm_flag():
    parser = _build_parser()
    ns = parser.parse_args([
        "--server", "http://x", "--name", "Maya",
        "--seed", "x", "--sex", "F", "--no-llm",
    ])
    assert ns.no_llm is True


def test_main_missing_server_returns_2(capsys, monkeypatch):
    monkeypatch.delenv("AGORA_SERVER", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "agora-agent", "--name", "Maya", "--seed", "x", "--sex", "F",
    ])
    code = main()
    assert code == 2
