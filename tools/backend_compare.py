"""Compile a copy of a built map with the Rust compiler and compare it, square by
square, to the original's WorldEd lots (KnoxMap UK).

    python tools/backend_compare.py output/<map> [--keep]

The original is never written to: copied (without lots, tmx, pictures) to a temp
folder, its .pzw repointed (as tools/compile_compare.py), compiled with knoxlots
(KNOXLOTS, else the usual places), and each square's tile stack read back from both
(tools/cell_check.read_cell) and compared. Prints the share of squares that differ,
and the tiles that differ most by sheet (so a missing roof type shows as roofs_*).
The ground's random picks (which of a blend's variants, grass overlay or tree a rule
puts on a square) are set aside: the two compilers draw them differently, and the
game shows the same ground either way. Exit 0 when every square matches.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


RANDOM = ("blends_", "e_", "vegetation_foliage", "vegetation_groundcover", "vegetation_trees",
          "jumbo_tree", "d_generic", "d_plants", "d_floorleaves", "d_streetcracks")


def _fixed(tiles: list[str]) -> list[str]:
    """A square's tiles without the ground's random picks."""
    return [t for t in tiles if not t.startswith(RANDOM)]


def _cells(lots: Path) -> list[tuple[int, int]]:
    return sorted(tuple(int(v) for v in p.stem.split("_")) for p in lots.glob("*.lotheader"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", help="a map folder compiled with WorldEd, e.g. output/mymap")
    ap.add_argument("--keep", action="store_true", help="keep the temp copy")
    args = ap.parse_args(argv)
    import compile_map
    from cell_check import read_cell

    src = Path(args.project).resolve()
    if not (src / "lots").is_dir() or not any((src / "lots").glob("*.lotheader")):
        print(f"{src} has no compiled lots to compare against.", file=sys.stderr)
        return 1
    work = Path(tempfile.mkdtemp(prefix="knoxmap-backend-"))
    dst = work / src.name
    try:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("lots", "tmx", "pictures", ".knoxlots"))
        for f in ("compile_state.json", "compile_failures.json"):
            (dst / f).unlink(missing_ok=True)
        pzw = dst / f"{src.name}.pzw"
        text = pzw.read_text(encoding="utf-8")
        pzw.write_text(text.replace(src.as_posix(), dst.as_posix()), encoding="utf-8")
        t0 = time.time()
        compile_map.compile_map(str(dst), backend="rust", fresh=True)
        print(f"knoxlots compiled the copy in {time.time() - t0:.0f}s")

        a_cells, b_cells = _cells(src / "lots"), _cells(dst / "lots")
        print(f"cells: WorldEd {len(a_cells)}, knoxlots {len(b_cells)}")
        squares = differ = 0
        only_a, only_b = collections.Counter(), collections.Counter()
        examples = []
        for cx, cy in sorted(set(a_cells) & set(b_cells)):
            a, b = read_cell(str(src / "lots"), cx, cy), read_cell(str(dst / "lots"), cx, cy)
            for key in set(a) | set(b):
                squares += 1
                ta, tb = _fixed(a.get(key, [])), _fixed(b.get(key, []))
                if ta == tb:
                    continue
                differ += 1
                if len(examples) < 12:
                    examples.append(((cx, cy) + key, ta, tb))
                ca, cb = collections.Counter(ta), collections.Counter(tb)
                for t in (ca - cb):
                    only_a[t.rsplit("_", 1)[0]] += 1
                for t in (cb - ca):
                    only_b[t.rsplit("_", 1)[0]] += 1
        print(f"squares {squares}, differ {differ} ({100 * differ / max(squares, 1):.3f}%)")
        print("tiles only WorldEd has, by sheet:", only_a.most_common(12))
        print("tiles only knoxlots has, by sheet:", only_b.most_common(12))
        for e in examples:
            print("  e.g.", e)
        return 0 if differ == 0 else 1
    finally:
        if args.keep:
            print(f"kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
