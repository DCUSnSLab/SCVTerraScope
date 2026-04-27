"""CODa 16-class operational taxonomy loader.

The bundled YAML at `scvterrascope/data/coda_taxonomy.yaml` is the default
source of truth — vendored from SCVTerraVision (Phase 1-1b approved). For
overrides (e.g., a downstream user with a different class set), pass a
custom path explicitly.

Returning the same `id, name` order as the training-side YAML is critical:
HF DeformableDetr emits 0-indexed labels, and we map index → name as
`operational_classes[label_idx].name`. Drift here silently corrupts the
class column in the GUI results table.
"""

from __future__ import annotations

import importlib.resources as ir
from dataclasses import dataclass
from pathlib import Path

_BUNDLED_YAML = ("scvterrascope.data", "coda_taxonomy.yaml")


@dataclass(frozen=True)
class OperationalClass:
    id: int  # 1-indexed (matches COCO category_id)
    name: str


@dataclass(frozen=True)
class Taxonomy:
    classes: tuple[OperationalClass, ...]

    def __len__(self) -> int:
        return len(self.classes)

    def name_for(self, label_index_zero_based: int) -> str:
        """HF DeformableDetr returns 0-indexed labels; resolve to a class name."""
        if not 0 <= label_index_zero_based < len(self.classes):
            raise IndexError(
                f"label index {label_index_zero_based} out of range for "
                f"{len(self.classes)}-class taxonomy"
            )
        return self.classes[label_index_zero_based].name

    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.classes)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Load taxonomy YAML and return a frozen Taxonomy.

    With `path=None`, reads the bundled `scvterrascope/data/coda_taxonomy.yaml`.
    """
    import yaml

    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
        source: Path | str = path
    else:
        # importlib.resources keeps this working when SCVTerraScope is
        # installed as a wheel (no filesystem path) as well as when it's
        # an editable install or source tree.
        text = ir.files(_BUNDLED_YAML[0]).joinpath(_BUNDLED_YAML[1]).read_text(
            encoding="utf-8"
        )
        source = "/".join(_BUNDLED_YAML)

    data = yaml.safe_load(text)
    return _parse(data, source=source)


def _parse(data: dict, *, source: Path | str) -> Taxonomy:
    raw = data.get("operational_classes")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source}: 'operational_classes' must be a non-empty list")
    classes = []
    for entry in raw:
        if not isinstance(entry, dict) or "id" not in entry or "name" not in entry:
            raise ValueError(f"{source}: bad entry {entry!r}")
        classes.append(OperationalClass(id=int(entry["id"]), name=str(entry["name"])))
    expected = list(range(1, len(classes) + 1))
    if [c.id for c in classes] != expected:
        raise ValueError(
            f"{source}: operational_classes ids must be 1..{len(classes)} contiguous, "
            f"got {[c.id for c in classes]}"
        )
    return Taxonomy(classes=tuple(classes))
