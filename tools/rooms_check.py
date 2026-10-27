"""Real houses through the room solver, as Build sees them (ROOMS.md step 2).

    KNOXMAP_ROOMS_DUMP=<dump> python -m knoxbuild <map>        (writes <dump>.<pid> files)
    python tools/rooms_check.py <dump> [--sample 20] [--all] [--out output/rooms_check]

Each house in the dump has Build's own footprint and storeys (attics cut), its
street side, party walls and garage, its 1993 look and its EPCs' habitable
rooms. A sample (spread over the programmes) is solved and drawn on one sheet,
with a line each: programme, variant, seconds, rule breaks, habitable rooms
against the EPC's. --all solves every house through the cache (rooms_uk.prewarm)
and reports how many have a layout, by programme.
"""
import argparse
import collections
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from PIL import Image, ImageDraw  # noqa: E402

from knoxbuild import rooms_uk as R  # noqa: E402


def habitable(lay) -> int:
    return sum(1 for st in lay.storeys for r in st["rooms"] if r["kind"] in R.RULES["habitable"])


def spread(houses, n, seed):
    """n houses, as even a share of each programme as there is."""
    by = collections.defaultdict(list)
    for h in houses:
        by[h[1]].append(h)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def sample(houses, a):
    os.makedirs(a.out, exist_ok=True)
    pics, lines = [], []
    for house, prog, rec in spread(houses, a.sample, a.seed):
        t = time.time()
        lay = R.solve(house, prog, workers=a.workers)
        took = time.time() - t
        look = (rec.get("rooms") or {}).get("look", "")
        head = f"{rec['id']} {look} -> {prog}"
        if lay is None:
            lines.append(f"{head}: NO LAYOUT ({took:.1f}s)")
            continue
        broken = R.check(lay)
        epc = rec.get("epc_rooms")
        lines.append(f"{head} v{lay.variant} {took:.1f}s  habitable {habitable(lay)}"
                     f"{f' (EPC {epc})' if epc else ''}  {'OK' if not broken else broken}")
        path = os.path.join(a.out, f"{rec['id']}.png")
        R.render(lay, path, f"{rec['id']} {look} street {house.street}"
                 f"{' EPC ' + str(epc) + ' rooms' if epc else ''}")
        pics.append(path)
    print("\n".join(lines))
    if pics:
        ims = [Image.open(p) for p in pics]
        cols = 3
        cw, ch = max(i.width for i in ims), max(i.height for i in ims)
        sheet = Image.new("RGB", (cw * cols, ch * ((len(ims) + cols - 1) // cols)), "white")
        for k, im in enumerate(ims):
            sheet.paste(im, ((k % cols) * cw, (k // cols) * ch))
        out = os.path.join(a.out, "sheet.png")
        R._save(sheet, out)
        with open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("sheet:", out)


def everything(houses, a):
    cache = R.LayoutCache(a.cache)
    print("stale entries dropped:", cache.prune())
    got = R.prewarm([(h, p) for h, p, _ in houses], cache, processes=a.processes, workers=a.workers,
                    progress=lambda i, n, t: print(f"  {i}/{n} footprints, {t:.0f}s", flush=True))
    print(got)
    by, ok = collections.Counter(), collections.Counter()
    off = []
    for h, p, rec in houses:
        by[p] += 1
        lay = R.layout_for(h, p, rec["id"], cache)
        if lay:
            ok[p] += 1
            if rec.get("epc_rooms"):
                off.append(habitable(lay) - rec["epc_rooms"])
    print(f"laid out {sum(ok.values())}/{len(houses)}")
    for p in sorted(by):
        print(f"  {p:18s} {ok[p]}/{by[p]}")
    if off:
        c = collections.Counter(off)
        print("habitable rooms against the EPC's (plan minus EPC):", dict(sorted(c.items())))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join("output", "rooms_check"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--cache")
    a = ap.parse_args(argv)
    houses = R.load_dump(a.dump)
    print(f"{len(houses)} houses in the dump;",
          dict(collections.Counter(p for _, p, _ in houses).most_common()))
    if a.all:
        a.workers = min(a.workers, 4)
        everything(houses, a)
    else:
        sample(houses, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
