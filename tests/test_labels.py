"""Taxonomy loader — does it correctly read the bundled YAML?"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scvterrascope.labels import load_taxonomy


def _write_taxonomy(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "coda_taxonomy.yaml"
    target.write_text(body, encoding="utf-8")
    return target


def test_loads_explicit_path(tmp_path: Path) -> None:
    yaml_path = _write_taxonomy(
        tmp_path,
        textwrap.dedent(
            """\
            version: 1
            operational_classes:
              - {id: 1, name: pedestrian}
              - {id: 2, name: bicycle}
              - {id: 3, name: vehicle}
            """
        ),
    )
    tax = load_taxonomy(yaml_path)
    assert len(tax) == 3
    assert tax.names() == ("pedestrian", "bicycle", "vehicle")


def test_label_index_resolution(tmp_path: Path) -> None:
    yaml_path = _write_taxonomy(
        tmp_path,
        textwrap.dedent(
            """\
            version: 1
            operational_classes:
              - {id: 1, name: pedestrian}
              - {id: 2, name: bicycle}
            """
        ),
    )
    tax = load_taxonomy(yaml_path)
    assert tax.name_for(0) == "pedestrian"  # 0-indexed (HF DETR)
    assert tax.name_for(1) == "bicycle"
    with pytest.raises(IndexError):
        tax.name_for(2)


def test_rejects_non_contiguous_ids(tmp_path: Path) -> None:
    yaml_path = _write_taxonomy(
        tmp_path,
        textwrap.dedent(
            """\
            version: 1
            operational_classes:
              - {id: 1, name: pedestrian}
              - {id: 3, name: bicycle}
            """
        ),
    )
    with pytest.raises(ValueError):
        load_taxonomy(yaml_path)


def test_bundled_taxonomy_loads_with_no_args() -> None:
    """The package-bundled `data/coda_taxonomy.yaml` should load without any arguments."""
    tax = load_taxonomy()
    assert len(tax) == 16
    # First 5 are stable per Phase 1-1b approval.
    assert tax.names()[:5] == ("pedestrian", "bicycle", "motorcycle", "scooter", "vehicle")
    # Last one is fire_hydrant (id=16).
    assert tax.classes[-1].id == 16
    assert tax.classes[-1].name == "fire_hydrant"


def test_explicit_missing_path_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_taxonomy(bogus)
