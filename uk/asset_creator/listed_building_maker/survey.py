"""Every listed building of an area read at once: what assets they need in common.

    python survey.py <area data dir> [--make] [--min-share 0.05]

Reads each entry's list text (listing.py; fetched once into cache/) with the
lexicon (lexicon.py), counts the features of the buildings' fronts (old work
only: a "modern shopfront" asks for nothing), and maps them to assets
(assets.py). Writes survey/<area>.json and survey/<area>.md: the features by
share of buildings, each asset with its state (ours / a sheet not yet wired /
a vanilla stand-in / Build / missing), and the assets still to make, the most
wanted first.

--make: for each asset still to make that at least --min-share of the
buildings need, runs its recipe where it has one and its sheet is not on disk,
and writes a brief (briefs/<asset>.md: what it is, how many buildings, the
listing's own words, the photos that may be looked at) for the rest.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import assets as AS
import lexicon as L
import listing as LI

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_ROOT = os.path.dirname(HERE)
KEEP_QUOTES = 6


def survey(area: str) -> dict:
    rows = LI.load(area)
    buildings, structures = [], []
    for row in rows:
        r = L.read(row["text"], row["name"])
        rec = {"entry": row["entry"], "name": row["name"], "grade": row["grade"], "kind": r.kind,
               "era": r.era, "storeys": r.storeys, "attic": r.attic, "bays": r.bays,
               "front": r.features("front"), "rear": r.features("rear"),
               "modern": sorted(f for f, hs in r.hits.items() if any(h.modern for h in hs)),
               "quotes": {f: hs[0].quote for f, hs in r.hits.items()},
               "photos": row["photos"]}
        (buildings if r.group == "building" else structures).append(rec)
    n = len(buildings)
    feat_count, feat_any = Counter(), Counter()
    for b in buildings:
        feat_count.update(b["front"])
        feat_any.update(set(b["front"]) | set(b["rear"]))
    need = defaultdict(list)
    for b in buildings:
        for f in b["front"]:
            if f in AS.BY_ID:
                need[f].append(b["entry"])
    for s in structures:
        need[f"structure.{s['kind']}"].append(s["entry"])
    table = []
    for aid, entries in need.items():
        a = AS.BY_ID.get(aid)
        if a is None:
            continue
        on_disk = {s: AS.sheet_on_disk(s) for s in a.sheets}
        table.append({"asset": aid, "what": a.what, "state": a.state, "buildings": len(entries),
                      "share": round(len(entries) / max(1, n if not aid.startswith("structure.") else len(structures)), 3),
                      "stand_in": a.stand_in, "recipe": a.recipe, "plan": a.plan,
                      "sheets": on_disk, "entries": entries})
    table.sort(key=lambda t: (-t["buildings"], t["asset"]))
    uses = Counter(p["use"] for row in rows for p in row["photos"])
    with_photo = sum(1 for row in rows if any(p["use"] in ("look", "texture") for p in row["photos"]))
    return {"area": os.path.basename(os.path.normpath(area)), "entries": len(rows), "buildings": n,
            "structures": len(structures),
            "kinds": dict(Counter(b["kind"] for b in buildings + structures).most_common()),
            "eras": dict(sorted(Counter((b["era"] // 50 * 50) if b["era"] else None for b in buildings).items(),
                                key=lambda kv: kv[0] or 0)),
            "features": {f: {"front": c, "share": round(c / n, 3), "any": feat_any[f]}
                         for f, c in feat_count.most_common()},
            "assets": table, "to_make": [t["asset"] for t in table if t["state"] in ("missing", "uk_sheet")],
            "photos": {"by_use": dict(uses), "entries_with_a_usable_photo": with_photo},
            "records": buildings + structures}


def write_report(s: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{s['area']}.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    lines = [f"# Listed buildings of {s['area']}: what they need",
             "", f"{s['entries']} list entries: {s['buildings']} buildings, {s['structures']} structures. "
             f"Kinds: {', '.join(f'{k} {v}' for k, v in s['kinds'].items())}.",
             "", "Built (earliest date in the text, by half century): "
             + ", ".join(f"{k or '?'}: {v}" for k, v in s["eras"].items()) + ".",
             "", f"Photos: {s['photos']['by_use']} (look = may be looked at and measured, never copied into a "
             f"tile; texture = may be; link = Historic England's page, view only). "
             f"{s['photos']['entries_with_a_usable_photo']} entries have a photo to look at.",
             "", "## Features of the fronts (share of buildings)", "", "| Feature | Buildings | Share |", "|---|---|---|"]
    lines += [f"| {f} | {v['front']} | {v['share']:.0%} |" for f, v in s["features"].items()]
    lines += ["", "## Assets, most wanted first", "",
              "| Asset | Buildings | State | Stand-in now | Recipe / plan |", "|---|---|---|---|---|"]
    for t in s["assets"]:
        lines.append(f"| {t['asset']} | {t['buildings']} ({t['share']:.0%}) | {t['state']} | {t['stand_in']} | "
                     f"{t['recipe'] or t['plan']} |")
    lines += ["", "## Still to make (missing, or drawn but not wired), most wanted first", ""]
    lines += [f"- {a}" for a in s["to_make"]]
    path = os.path.join(out_dir, f"{s['area']}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def brief(s: dict, t: dict, out_dir: str) -> str:
    recs = {r["entry"]: r for r in s["records"]}
    lines = [f"# {t['asset']}: {t['what']}", "",
             f"Wanted by {t['buildings']} listed buildings of {s['area']} ({t['share']:.0%}); state: {t['state']}; "
             f"Build lays meanwhile: {t['stand_in']}.", ""]
    if t["plan"]:
        lines += [f"Planned: {t['plan']}", ""]
    feature = t["asset"]
    lines += ["## In the listings' own words", ""]
    for e in t["entries"][:KEEP_QUOTES]:
        r = recs[e]
        q = r["quotes"].get(feature) or ""
        lines.append(f"- {r['name']} (grade {r['grade']}, {r['era'] or '?'}): \"{q}\"")
    lines += ["", "## Photos of those buildings", "",
              "Commons photos are CC BY-SA unless marked: look and measure only, never copy or trace into a tile "
              "(the pack is CC BY). Historic England's are view-only.", ""]
    for e in t["entries"]:
        for p in recs[e]["photos"]:
            if p["use"] in ("look", "texture"):
                lines.append(f"- {recs[e]['name']}: [{p['file']}]({p['page']}) - {p['licence']}, {p['author']} "
                             f"({p['use']})")
    lines += [f"- Historic England's photos: {recs[e]['photos'][-1]['page']}" for e in t["entries"][:3]]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{t['asset']}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def make(s: dict, min_share: float) -> None:
    ran = set()
    for t in s["assets"]:
        if t["recipe"] and not all(t["sheets"].values()):
            # A sheet the table names is not on disk: its maker draws it again.
            cmd = t["recipe"].split("   ")[0].split()
            if tuple(cmd) not in ran:
                print(f"  {t['asset']}: running {' '.join(cmd)}")
                subprocess.run([sys.executable] + cmd[1:], cwd=ASSET_ROOT, check=True)
                ran.add(tuple(cmd))
            continue
        if t["state"] not in ("missing", "uk_sheet") or t["share"] < min_share:
            continue
        print(f"  {t['asset']}: brief {os.path.relpath(brief(s, t, os.path.join(HERE, 'briefs')), HERE)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("area")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--min-share", type=float, default=0.05)
    a = ap.parse_args()
    s = survey(a.area)
    path = write_report(s, os.path.join(HERE, "survey"))
    print(f"wrote {path}: {s['buildings']} buildings, {s['structures']} structures")
    for t in s["assets"][:15]:
        print(f"  {t['asset']:24} {t['buildings']:4} ({t['share']:.0%})  {t['state']}")
    if a.make:
        make(s, a.min_share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
