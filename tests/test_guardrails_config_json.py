"""Tests for :class:`GuardrailConfig` JSON serialisation (Fix A)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroguage.guardrails import GuardrailConfig  # noqa: E402


def test_guardrail_config_default_round_trip() -> None:
    cfg = GuardrailConfig()
    payload = cfg.to_dict()
    restored = GuardrailConfig.from_dict(payload)
    assert restored.to_dict() == payload


@pytest.mark.parametrize(
    "preset_name",
    ["guardrails_default", "guardrails_calibrated", "guardrails_personalized"],
)
def test_shipped_preset_loads_and_validates(preset_name: str) -> None:
    preset_path = ROOT / "configs" / f"{preset_name}.json"
    assert preset_path.exists(), f"Missing preset file: {preset_path}"
    cfg = GuardrailConfig.from_json(preset_path)
    assert cfg.profile_name
    assert cfg.profile_description
    assert 0.0 < cfg.yellow_threshold < cfg.red_threshold < cfg.black_threshold <= 1.0


def test_partial_dict_preserves_other_defaults() -> None:
    cfg = GuardrailConfig.from_dict({"yellow_threshold": 0.5})
    assert cfg.yellow_threshold == pytest.approx(0.5)
    assert cfg.red_threshold == GuardrailConfig().red_threshold


def test_unknown_keys_are_ignored() -> None:
    cfg = GuardrailConfig.from_dict({
        "yellow_threshold": 0.45,
        "this_key_does_not_exist": "ignore-me",
    })
    assert cfg.yellow_threshold == pytest.approx(0.45)


def test_invalid_threshold_ordering_rejected() -> None:
    with pytest.raises(ValueError):
        GuardrailConfig.from_dict({
            "yellow_threshold": 0.7,
            "red_threshold": 0.5,
        })


def test_from_json_uses_filename_as_default_profile_name(tmp_path: Path) -> None:
    target = tmp_path / "my_profile.json"
    target.write_text(json.dumps({"yellow_threshold": 0.50,
                                  "red_threshold": 0.62,
                                  "black_threshold": 0.72,
                                  "green_recovery_threshold": 0.45,
                                  "yellow_recovery_threshold": 0.55,
                                  "red_recovery_threshold": 0.65}),
                      encoding="utf-8")
    cfg = GuardrailConfig.from_json(target)
    assert cfg.profile_name == "my_profile"
