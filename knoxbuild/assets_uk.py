"""KnoxMap UK: a map made with no UK tile, for players without the KnoxMapUK_Tiles mod.

`uk_assets.off` in a map's folder (written by Building_types/src/uk_pipeline.py
from uk.json's `uk_assets`) turns them off:

- Build reads regional.json through `vanilla()` (regional_uk.load): each UK
  style becomes its vanilla stand-in (the palette's `uk_stand_ins`, a lookup;
  STAND_INS when a map's palette has none), the palette's UK styles and shop
  signs go, and every building's UK parts go - those naming a UK sheet, and
  the kits named only by key (UK_KEYS: bays, porches, doors, downpipes,
  garages, trims, listed builds);
- the Build steps that lay UK pieces of their own ask `off()` (fences,
  shop fronts, the police box...);
- the map steps before Build swap or leave out theirs (kerbs, markings, signs,
  furniture: Building_types), and the ground is compiled by the vanilla rules
  (tools/terrain_source.rules_path);
- `uk_tiles()` lists any UK tile left in a built map (uk_pipeline.py's check).
"""
from __future__ import annotations

import glob
import json
import os
import re

OFF_FILE = "uk_assets.off"
UK = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_uk_\d\d[A-Za-z0-9_]*")
# Kits named by key rather than by sheet (regional_uk, housing_kit, listed_building_maker).
UK_KEYS = ("bay", "door", "door_surround", "drains", "garage", "wall_trim", "upper_wall", "upper_sides",
           "fascia", "uk_front", "loading", "landmark", "listed")
# A UK style -> the palette's vanilla look-alike, when the palette has no `uk_stand_ins`.
STAND_INS = {"uk_victorian_terrace": "victorian", "uk_interwar_30s": "cream", "uk_vernacular_timber": "cream"}
DEFAULT_STAND_IN = "brick"


def off(map_dir: str | None) -> bool:
    return bool(map_dir) and os.path.exists(os.path.join(map_dir, OFF_FILE))


def has_uk(value) -> bool:
    return bool(UK.search(json.dumps(value) if not isinstance(value, str) else value))


def _base(style: str, uk_styles: set[str]) -> str | None:
    """A style's UK palette style, if it is one or a variant of one (regional_uk names a
    variant "uk_<style>__<roof>__<window>")."""
    if style in uk_styles:
        return style
    if style.startswith("uk_") and "__" in style:
        base = style[3:].split("__")[0]
        return base if base in uk_styles or base.startswith("uk_") else None
    return None


def vanilla(regional: dict) -> dict:
    """regional.json with no UK tile: styles to their stand-ins, UK parts left out."""
    pal = dict(regional.get("palette") or {})
    adds = pal.get("add") or []
    uk_styles = {a["name"] for a in adds if has_uk(a)}
    stand_ins = {k: v for k, v in {**STAND_INS, **(pal.get("uk_stand_ins") or {})}.items() if k != "_about"}
    pal["add"] = [a for a in adds if a["name"] not in uk_styles]
    # a stand-in the palette drops (KnoxMap's timber and panel on an English map) is back
    pal["drop"] = [d for d in pal.get("drop") or [] if d not in set(stand_ins.values())]
    pal.pop("shop_signs", None)
    kept = {a["name"] for a in pal["add"]}
    out = {}
    for oid, e in (regional.get("buildings") or {}).items():
        e = {k: v for k, v in e.items() if k not in UK_KEYS and (k == "style" or not has_uk(v))}
        base = _base(str(e.get("style") or ""), uk_styles)
        if base is not None:
            s = stand_ins.get(base, DEFAULT_STAND_IN)
            e["style"] = s if s in kept or s in ("brick", "render", "painted", "stucco", "panel", "timber") \
                else DEFAULT_STAND_IN
        out[oid] = e
    # the rest (the forecourt kit, the pubs' props) only where it names no UK tile
    rest = {k: v for k, v in regional.items() if k not in ("palette", "buildings", "pub_props") and not has_uk(v)}
    return {**rest, "palette": pal, "buildings": out}


def uk_tiles(map_dir: str) -> dict[str, int]:
    """Every UK tile name in a built map's buildings and loose tiles, with how often."""
    found: dict[str, int] = {}
    name = os.path.basename(os.path.normpath(map_dir))
    files = glob.glob(os.path.join(map_dir, "buildings", "*.tbx")) + \
        [os.path.join(map_dir, f"{name}_structures.json")]
    for f in files:
        if os.path.exists(f):
            for m in UK.findall(open(f, encoding="utf-8", errors="replace").read()):
                found[m] = found.get(m, 0) + 1
    return found
