"""Tests for Day 6 — T17.4 Release Packaging module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuroguage.release import _sha256, build_release_bundle


def test_sha256_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"hello-neuroguage")
    h1 = _sha256(path)
    h2 = _sha256(path)
    assert h1 == h2
    assert len(h1) == 64


def test_build_release_bundle_produces_manifest_and_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundle must produce a manifest and evidence index even when
    all source artifacts are missing (graceful degradation)."""
    monkeypatch.chdir(tmp_path)

    manifest = build_release_bundle(
        release_dir=tmp_path / "release",
        version="test-0.0.1",
        commit="deadbeef",
    )
    manifest_path = tmp_path / "release" / "manifest.json"
    index_path = tmp_path / "release" / "evidence_index.md"
    assert manifest_path.exists()
    assert index_path.exists()

    data = json.loads(manifest_path.read_text())
    assert data["version"] == "test-0.0.1"
    assert data["commit"] == "deadbeef"
    # None of the sources exist in the tmp workspace, so included=False everywhere
    assert all(e["included"] is False for e in data["entries"])


def test_build_release_bundle_copies_existing_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sources that exist should be copied with SHA-256 recorded."""
    monkeypatch.chdir(tmp_path)

    # Stage a minimal source tree
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data_contract.json").write_text("{}")
    (tmp_path / "configs" / "demo_timing.json").write_text("{}")
    (tmp_path / "README.md").write_text("# README")
    (tmp_path / "environment.yml").write_text("name: neuroguage")
    (tmp_path / "requirements.txt").write_text("numpy")

    manifest = build_release_bundle(release_dir=tmp_path / "rel", version="v1")
    data = json.loads((tmp_path / "rel" / "manifest.json").read_text())

    included_labels = {e["label"] for e in data["entries"] if e["included"]}
    assert "Day 1 data contract" in included_labels
    assert "Day 5 demo timing envelope" in included_labels
    assert "Updated README" in included_labels

    # Copied files should carry a sha256
    for e in data["entries"]:
        if e["included"]:
            assert "sha256" in e
            assert len(e["sha256"]) == 64
