"""Where KnoxMap and the UK chain (Building_types) are, from asset_creator, in either
layout it is kept in.

- The KnoxMap UK workspace: asset_creator beside Mapping/ (Mapping/Knoxmap,
  Mapping/Building_types).
- The fork of KnoxMap (KnoxMap UK, branch uk-mode): asset_creator as uk/asset_creator,
  KnoxMap at the repository's root and the chain in uk/.

KNOXMAP_DIR and KNOXMAP_UK_DIR (environment) win over both.
"""
from __future__ import annotations

import os

AC = os.path.dirname(os.path.abspath(__file__))


def _first(*candidates, inside: str) -> str | None:
    for c in candidates:
        if c and os.path.exists(os.path.join(c, inside)):
            return os.path.normpath(c)
    return None


KNOX = _first(os.environ.get("KNOXMAP_DIR"), os.path.join(AC, "..", "Mapping", "Knoxmap"),
              os.path.join(AC, "..", ".."), inside="knoxbuild") \
    or os.path.normpath(os.path.join(AC, "..", "Mapping", "Knoxmap"))
UK = _first(os.environ.get("KNOXMAP_UK_DIR"), os.path.join(AC, "..", "Mapping", "Building_types"),
            os.path.join(AC, ".."), inside=os.path.join("src", "uk_pipeline.py")) \
    or os.path.normpath(os.path.join(AC, "..", "Mapping", "Building_types"))
TOOLS_TILES = os.path.join(KNOX, "vendor", "PZMappingTools", "Tiles", "2x")     # the map tools' sheets
KERBS_TOPDOWN = os.path.join(UK, "data", "kerbs", "kerbs_topdown.json")
