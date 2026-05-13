import sys
from unittest.mock import patch

import pytest

from agora_agent_sdk.cli import _build_parser, _normalize_color, main


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


class TestNormalizeColor:
    def test_none_passes_through(self):
        assert _normalize_color(None) is None

    def test_hex_passes_through_lowercased(self):
        assert _normalize_color("#A87FD2") == "#a87fd2"
        assert _normalize_color("#000000") == "#000000"

    def test_named_blue(self):
        assert _normalize_color("blue") == "#0000ff"

    def test_named_case_insensitive(self):
        assert _normalize_color("Blue") == "#0000ff"
        assert _normalize_color("RED") == "#ff0000"

    def test_named_whitespace_tolerant(self):
        assert _normalize_color("  green  ") == "#008000"

    def test_grey_synonym(self):
        assert _normalize_color("grey") == _normalize_color("gray")

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError) as exc:
            _normalize_color("definitely-not-a-color")
        assert "definitely-not-a-color" in str(exc.value)
        assert "#RRGGBB" in str(exc.value)

    def test_short_hex_rejected(self):
        with pytest.raises(ValueError):
            _normalize_color("#fff")


def test_main_invalid_color_returns_2(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "agora-agent", "--server", "http://x", "--name", "Maya",
        "--seed", "x", "--sex", "F", "--color", "nonsense",
    ])
    code = main()
    assert code == 2
    captured = capsys.readouterr()
    assert "nonsense" in captured.err
