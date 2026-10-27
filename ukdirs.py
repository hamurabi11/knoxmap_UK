"""KnoxMap UK: where the UK chain (Building_types) and its assets (asset_creator) are,
from KnoxMap's side, in either layout (Building_types/src/ukpaths.py has the same).

- The KnoxMap UK workspace: Building_types beside this folder, asset_creator beside
  their parent.
- The fork (KnoxMap UK, branch uk-mode): the chain in uk/ here; asset_creator
  beside the repository or in uk/asset_creator, if at all.

KNOXMAP_UK_DIR and KNOXMAP_UK_ASSETS (environment) win over both.
"""
from __future__ import annotations

import os

BASE = os.path.dirname(os.path.abspath(__file__))


def _first(*candidates, inside: str):
    for c in candidates:
        if c and os.path.exists(os.path.join(c, inside)):
            return os.path.normpath(c)
    return None


UK = _first(os.environ.get("KNOXMAP_UK_DIR"), os.path.join(BASE, "..", "Building_types"),
            os.path.join(BASE, "uk"), inside="src") or os.path.normpath(os.path.join(BASE, "..", "Building_types"))
UK_SRC = os.path.join(UK, "src")
ASSETS = _first(os.environ.get("KNOXMAP_UK_ASSETS"), os.path.join(BASE, "..", "..", "asset_creator"),
                os.path.join(BASE, "..", "asset_creator"), os.path.join(UK, "asset_creator"),
                inside="listed_building_maker")


def assets(*parts: str) -> str | None:
    """A path inside asset_creator, or None without one."""
    return os.path.join(ASSETS, *parts) if ASSETS else None
