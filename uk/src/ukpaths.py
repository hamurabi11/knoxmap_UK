"""Where the UK chain's folders are, in either layout it is kept in.

- The KnoxMap UK workspace: Mapping/Knoxmap (KnoxMap), Mapping/Building_types
  (this), and asset_creator beside Mapping.
- The fork of KnoxMap (KnoxMap UK, branch uk-mode): KnoxMap at the
  repository's root and this folder as uk/ in it; asset_creator, if there is one,
  beside the repository or in uk/asset_creator.

KNOXMAP_DIR and KNOXMAP_UK_ASSETS (environment) win over both. ASSETS is None when
there is no asset_creator: the steps that need it (listed buildings, the UK tile
pack) go without, and maps are made with UK assets off.
"""
from __future__ import annotations

import os

SRC = os.path.dirname(os.path.abspath(__file__))
UK = os.path.dirname(SRC)                         # Building_types, or the fork's uk/


def _first(*candidates: str | None, inside: str) -> str | None:
    for c in candidates:
        if c and os.path.exists(os.path.join(c, inside)):
            return os.path.normpath(c)
    return None


KNOX = _first(os.environ.get("KNOXMAP_DIR"), os.path.join(UK, "..", "Knoxmap"), os.path.join(UK, ".."),
              inside="knoxbuild") or os.path.normpath(os.path.join(UK, "..", "Knoxmap"))
ASSETS = _first(os.environ.get("KNOXMAP_UK_ASSETS"), os.path.join(UK, "..", "..", "asset_creator"),
                os.path.join(KNOX, "..", "asset_creator"), os.path.join(UK, "asset_creator"),
                inside="listed_building_maker")


def assets(*parts: str) -> str | None:
    """A path inside asset_creator, or None without one."""
    return os.path.join(ASSETS, *parts) if ASSETS else None
