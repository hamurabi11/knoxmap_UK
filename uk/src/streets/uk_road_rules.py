"""Carriageways in vanilla's dark asphalt (street2) for the map tools.

    python uk_road_rules.py [<Rules.txt>] [--undo]

Vanilla B42 paves its roads with street2, the dark asphalt
(blends_street_01_80-87: 55k squares in Muldraugh against 1.7k of street3,
LOG 2026-10-10), and its kerb overlays (street_curbs_01_blend_street) are
drawn for it. KnoxMap paints town streets 150 150 150, which Rules.txt makes
street3, the smooth grey asphalt. This rule draws that colour with street2
tiles instead: every step keeps reading carriageways by their colour, and car
parks (100 100 100, street2 already) stay told apart from them in the
bitmaps. Main roads and service lanes (80 80 80, street4, the worn speckled
tarmac) are left as they are.

And Blends.txt: the pavement's slabs (street) no longer blend onto the road
surfaces (street2, street3, street4). Road and pavement meet at a kerb
(kerbs.py); the blend laid a strip of slab over the road beyond every
straight kerb and a slab corner on the road at every step of a 45-degree
edge - the pale triangles beside the straight 45-degree kerbs (LOG
2026-10-11).

Run once on the map tools, as uk_tree_rules.py; a map takes it when it is
next compiled (clear its lots/ and compile_state.json: the incremental
compile does not see a change of rules). The files are backed up first as
*.pre-uk-roads; --undo puts street3 and the blends back.
"""
from __future__ import annotations

import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
RULES = os.path.join(ukpaths.KNOX, "vendor", "PZMappingTools", "config", "Rules.txt")
CARRIAGEWAY = re.compile(r"(color = 150 150 150\s*\n\s*tiles = )(street[23])\b")
# A blend whose main tile is the pavement, up to its exclude list.
PAVEMENT_BLEND = re.compile(r"(mainTile = street[ \t]*\n(?:[^\n]*\n)*?[ \t]*exclude = )([^\n]*)")
ROADS = ["street2", "street3", "street4"]


def _read(path: str) -> tuple[str, bool]:
    raw = open(path, "rb").read()
    return raw.decode("utf-8").replace("\r\n", "\n"), b"\r\n" in raw[:4096]


def _write(path: str, text: str, crlf: bool) -> None:
    backup = path + ".pre-uk-roads"
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    with open(path, "w", encoding="utf-8", newline="\r\n" if crlf else "\n") as f:
        f.write(text)


def blends(path: str, undo: bool) -> int:
    """The pavement's blends kept off the road surfaces (or let back on).
    How many blends there are."""
    text, crlf = _read(path)

    def fix(m):
        words = [w for w in m.group(2).split() if w not in ROADS]
        return m.group(1) + " ".join(words + ([] if undo else ROADS))
    new, n = PAVEMENT_BLEND.subn(fix, text)
    _write(path, new, crlf)
    return n


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else RULES
    undo = "--undo" in sys.argv
    want = "street3" if undo else "street2"
    text, crlf = _read(path)
    if not CARRIAGEWAY.search(text):
        print(f"no rule for 150 150 150 in {path}")
        return 1
    _write(path, CARRIAGEWAY.sub(lambda m: m.group(1) + want, text), crlf)
    print(f"150 150 150 drawn with {want} in {path}")
    n = blends(os.path.join(os.path.dirname(path), "Blends.txt"), undo)
    print(f"{n} pavement blends {'let back onto' if undo else 'kept off'} the road surfaces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
