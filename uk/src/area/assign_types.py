"""Join EPCs to buildings, predict era where there is none, assign types.

    python assign_types.py <area data dir> [<epc dir>] [plots.parquet]

Reads <area>/buildings.csv (features.py) and <epc dir>/epc_domestic.csv
(extract_epc.py; by default the area's council's, area_conf.epc_dir). At the
osm_lidar data level, or with no EPC, there is no era but a listing's: the
homes take a sub-type by built form (subtypes.unknown_era). Writes <area>/types.csv and adds era/type to
buildings.geojson, then prints how well era is predicted for buildings that
have an EPC when their own certificate is hidden (leave-one-out).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

ERAS = ["pre1900", "1900_1929", "1930_1949", "1950_1966", "1967_1982",
        "1983_1995", "1996_plus"]
ERA_INDEX = {e: i for i, e in enumerate(ERAS)}

# Development clusters: neighbours on the same street that look alike.
LINK_M = 25.0
AREA_RATIO = 1.5
ORIENT_DEG = 15.0
KNN = 6
KNN_RADIUS_M = 120.0


def era_of(band: str) -> str | None:
    band = (band or "").lower()
    if "before 1900" in band:
        return "pre1900"
    years = [int(y) for y in re.findall(r"(1[89]\d\d|20\d\d)", band)]
    if not years:
        return None
    y = years[0]
    for era, top in (("pre1900", 1899), ("1900_1929", 1929), ("1930_1949", 1949),
                     ("1950_1966", 1966), ("1967_1982", 1982), ("1983_1995", 1995)):
        if y <= top:
            return era
    return "1996_plus"


def band_year(band: str) -> int | None:
    """The first year of an EPC age band (before 1900: 1890)."""
    band = (band or "").lower()
    if "before 1900" in band:
        return 1890
    years = [int(y) for y in re.findall(r"(1[89]\d\d|20\d\d)", band)]
    return years[0] if years else None


def wall_kind(desc: str) -> str:
    d = (desc or "").lower()
    for key, kind in (("timber frame", "timber_frame"), ("cavity", "cavity"),
                      ("solid brick", "solid_brick"), ("sandstone", "stone"),
                      ("granite", "stone"), ("system built", "system_built"),
                      ("cob", "cob"), ("park home", "park_home")):
        if key in d:
            return kind
    return "other" if d else ""


def roof_kind(desc: str) -> str:
    d = (desc or "").lower()
    for key, kind in (("thatch", "thatched"), ("flat", "flat"), ("roof room", "roof_rooms"),
                      ("pitched", "pitched"), ("another dwelling above", "dwelling_above")):
        if key in d:
            return kind
    return "other" if d else ""


def load_epc(path: str) -> dict[str, dict]:
    """Latest certificate per UPRN."""
    latest: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = (row.get("uprn") or "").strip()
            if not u:
                continue
            u = u.split(".")[0]
            if u not in latest or row["lodgement_date"] > latest[u]["lodgement_date"]:
                latest[u] = row
    return latest


def load_plots(path: str | None) -> dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    import pandas as pd
    df = pd.read_parquet(path)
    return {str(u): float(a) for u, a in zip(df["uprn"], df["plot_area_sqm"])}


def majority(values):
    values = [v for v in values if v]
    return Counter(values).most_common(1)[0][0] if values else ""


def summarise_epc(b: dict, epc: dict, plots: dict) -> None:
    rows = [epc[u] for u in b["uprn_list"] if u in epc]
    b["epc_count"] = len(rows)
    eras = [era_of(r["construction_age_band"]) for r in rows]
    b["epc_era"] = majority(eras)
    b["epc_form"] = majority(r["built_form"] for r in rows)
    b["epc_type"] = majority(r["property_type"] for r in rows)
    b["epc_walls"] = majority(wall_kind(r["walls_description"]) for r in rows)
    b["epc_roof"] = majority(roof_kind(r["roof_description"]) for r in rows)
    # For the sub-types (subtypes.py): the finer age band, social tenancies,
    # floor height.
    years = sorted(y for y in (band_year(r["construction_age_band"]) for r in rows) if y)
    b["epc_year"] = years[len(years) // 2] if years else None
    tenures = [(r.get("tenure") or "").lower() for r in rows]
    known = [t for t in tenures if t and t not in ("unknown", "not defined", "no data!")]
    b["epc_social_n"] = sum(1 for t in known if "social" in t)
    b["epc_tenure_n"] = len(known)
    b["epc_social"] = b["epc_social_n"] / len(known) if known else 0.0
    heights = sorted(float(r["floor_height"]) for r in rows if (r.get("floor_height") or "").replace(".", "", 1).isdigit())
    b["epc_floor_h"] = heights[len(heights) // 2] if heights else 0.0
    areas = [float(r["total_floor_area"]) for r in rows if r.get("total_floor_area")]
    b["epc_floor_m2"] = round(sum(areas) / len(areas), 1) if areas else ""
    # habitable rooms, the median over the certificates (rooms_uk: how many rooms a house's plan has)
    rooms = sorted(int(r["number_habitable_rooms"]) for r in rows
                   if (r.get("number_habitable_rooms") or "").strip().isdigit())
    b["epc_rooms"] = rooms[len(rooms) // 2] if rooms else ""
    plot = [plots[u] for u in b["uprn_list"] if u in plots]
    b["plot_m2"] = round(sum(plot), 1) if plot else ""


def residential(b) -> bool:
    return b["form"] not in ("outbuilding", "non_residential")


def clusters(blds) -> None:
    """Link alike neighbours on the same street; components are developments."""
    parent = list(range(len(blds)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_street = defaultdict(list)
    for i, b in enumerate(blds):
        if residential(b) and b["street"]:
            by_street[b["street"]].append(i)
    for idx in by_street.values():
        for a_pos, i in enumerate(idx):
            a = blds[i]
            for j in idx[a_pos + 1:]:
                c = blds[j]
                if math.dist(a["xy"], c["xy"]) > LINK_M + (a["long_m"] + c["long_m"]) / 2:
                    continue
                big, small = max(a["area_m2"], c["area_m2"]), min(a["area_m2"], c["area_m2"])
                if small <= 0 or big / small > AREA_RATIO:
                    continue
                d = abs(a["orient_deg"] - c["orient_deg"]) % 90
                if min(d, 90 - d) > ORIENT_DEG:
                    continue
                parent[find(i)] = find(j)
    for i, b in enumerate(blds):
        b["dev"] = find(i)
    sizes = Counter(b["dev"] for b in blds)
    for b in blds:
        b["dev_size"] = sizes[b["dev"]]


def predict_era(i, blds, by_dev, by_street, hide_self=True):
    """Era for building i from its development, its street, then the nearest
    buildings with a known era. Returns (era, method)."""
    b = blds[i]
    if b.get("listed_grade"):
        # Listed buildings are old whatever an assessor's age band says: in
        # Odiham 42 of the 58 with an EPC say pre-1900 and most others are noise.
        return "pre1900", "listing"
    others = lambda ids: [blds[k]["epc_era"] for k in ids if k != i and blds[k]["epc_era"]]
    dev = others(by_dev[b["dev"]]) if b["dev_size"] > 1 else []
    if dev:
        return majority(dev), "development"
    near = sorted((math.dist(b["xy"], blds[k]["xy"]), k) for k in range(len(blds))
                  if k != i and blds[k]["epc_era"] and residential(blds[k]))
    near = [(d, k) for d, k in near[:KNN] if d <= KNN_RADIUS_M]
    if near:
        votes = Counter()
        for d, k in near:
            votes[blds[k]["epc_era"]] += 1.0 / (1.0 + d / 20.0)
        return votes.most_common(1)[0][0], "nearby"
    street = others(by_street.get(b["street"], []))
    if street:
        return majority(street), "street"
    return "", "none"


def type_of(b) -> str:
    """Draft taxonomy (PLAN.md, Tier 1) from era, form, listing and EPC."""
    if b["form"] == "outbuilding":
        return "outbuilding"
    if b["btag"] in ("farm_auxiliary", "barn", "farmhouse", "stable", "cowshed"):
        return "farm"
    if b["form"] == "non_residential":
        return "non_residential"
    era = b["era"]
    flats = b.get("epc_type") in ("Flat", "Maisonette") or b["form"] == "multi_unit"
    terrace = b["form"] in ("terrace_row", "mid_terrace", "end_terrace")
    if b.get("listed_grade"):
        if b.get("epc_walls") == "timber_frame":
            return "timber_framed"
        if (b.get("setback_m") or 99) <= 4 and (terrace or b["street"] == "High Street"):
            return "georgian_townhouse"
        return "listed_cottage"
    if not era:
        return "unknown"
    if flats:
        return "flats_old" if ERA_INDEX[era] <= ERA_INDEX["1930_1949"] else "flats_modern"
    if era == "pre1900":
        return "victorian_terrace" if terrace else "victorian_cottage"
    return {"1900_1929": "edwardian", "1930_1949": "interwar_semi",
            "1950_1966": "postwar_council", "1967_1982": "estate_60s70s",
            "1983_1995": "estate_80s90s", "1996_plus": "new_build"}[era]


def listings_read(blds) -> dict:
    """{list entry: lexicon Reading} for the listed buildings, from
    asset_creator's listed_building_maker cache (none if it is not there)."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
    import ukpaths  # noqa: E402
    maker = ukpaths.assets("listed_building_maker") or ""
    if not os.path.isdir(maker):
        return {}
    sys.path.insert(0, maker)
    try:
        import lexicon
    except ImportError:
        return {}
    out = {}
    for b in blds:
        e = str(b.get("list_entry") or "").split(".")[0]
        path = os.path.join(maker, "cache", "text", f"{e}.json")
        if e and os.path.exists(path):
            d = json.load(open(path, encoding="utf-8"))
            out[e] = lexicon.read(d.get("description") or "", d.get("entry_name") or "")
    return out


def main() -> int:
    import area_conf
    area = sys.argv[1]
    full = area_conf.level(area) == "full"
    epc_dir = (sys.argv[2] if len(sys.argv) > 2 else area_conf.epc_dir(area)) if full else None
    plots = load_plots(sys.argv[3] if len(sys.argv) > 3 and full else None)
    epc_path = os.path.join(epc_dir, "epc_domestic.csv") if epc_dir else ""
    epc = load_epc(epc_path) if os.path.exists(epc_path) else {}
    if not epc:
        print(f"no domestic EPCs ({area_conf.level(area)}, {epc_dir or 'no council'}): eras from listings only")
    with open(os.path.join(area, "buildings.csv"), encoding="utf-8") as f:
        blds = list(csv.DictReader(f))
    from pyproj import Transformer
    to_bng = Transformer.from_crs(4326, 27700, always_xy=True)
    for b in blds:
        for k in ("area_m2", "long_m", "orient_deg", "aspect", "rectangularity"):
            b[k] = float(b.get(k) or 0)
        b["setback_m"] = float(b["setback_m"]) if b.get("setback_m") else None
        b["uprn_list"] = [u for u in (b.get("uprns") or "").split(";") if u]
        b["xy"] = to_bng.transform(float(b["lon"]), float(b["lat"]))
        summarise_epc(b, epc, plots)
    clusters(blds)
    by_dev, by_street = defaultdict(list), defaultdict(list)
    for i, b in enumerate(blds):
        if residential(b):
            by_dev[b["dev"]].append(i)
            by_street[b["street"]].append(i)

    # Hold-out: hide each building's own EPC and predict it.
    res = [i for i, b in enumerate(blds) if residential(b)]
    known = [i for i in res if blds[i]["epc_era"]]
    hits = near_hits = 0
    by_method = defaultdict(lambda: [0, 0])
    confusion = defaultdict(Counter)
    for i in known:
        guess, method = predict_era(i, blds, by_dev, by_street)
        truth = blds[i]["epc_era"]
        ok = guess == truth
        hits += ok
        near_hits += bool(guess) and abs(ERA_INDEX[guess] - ERA_INDEX[truth]) <= 1
        by_method[method][0] += ok
        by_method[method][1] += 1
        confusion[truth][guess or "none"] += 1

    for i, b in enumerate(blds):
        if b.get("listed_grade") and residential(b):
            b["era"], b["era_source"] = "pre1900", "listing"
        elif b["epc_era"]:
            b["era"], b["era_source"] = b["epc_era"], "epc"
        elif residential(b):
            b["era"], b["era_source"] = predict_era(i, blds, by_dev, by_street)
        else:
            b["era"], b["era_source"] = "", ""
        b["type"] = type_of(b)

    # Sub-types (subtypes.py): with each development's social share and
    # median EPC year, the LiDAR's roofs and the listings' text.
    import subtypes as ST
    dev_soc, dev_n, dev_years = Counter(), Counter(), defaultdict(list)
    for b in blds:
        dev_soc[b["dev"]] += b["epc_social_n"]
        dev_n[b["dev"]] += b["epc_tenure_n"]
        if b["epc_year"]:
            dev_years[b["dev"]].append(b["epc_year"])
    roofs = {}
    rp = os.path.join(area, "roofs.csv")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            roofs = {r["osm_id"]: r for r in csv.DictReader(f)}
    readings = listings_read(blds)
    for b in blds:
        ys = sorted(dev_years[b["dev"]])
        b["subtype"], b["subtype_why"] = ST.subtype_of(
            b, readings.get(str(b.get("list_entry") or "").split(".")[0]), roofs.get(str(b["osm_id"])),
            dev_soc[b["dev"]] / dev_n[b["dev"]] if dev_n[b["dev"]] and b["dev_size"] > 1 else 0.0,
            ys[len(ys) // 2] if ys else None)
        b["look"] = ST.look_of(b["subtype"], b)

    cols = ["osm_id", "form", "type", "subtype", "subtype_why", "look", "era", "era_source", "epc_count", "epc_era",
            "epc_form", "epc_type", "epc_walls", "epc_roof", "epc_floor_m2", "epc_rooms", "plot_m2",
            "listed_grade", "list_name", "street", "dev", "dev_size", "area_m2",
            "uprn_count"]
    with open(os.path.join(area, "types.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(blds)
    gj_path = os.path.join(area, "buildings.geojson")
    with open(gj_path, encoding="utf-8") as f:
        gj = json.load(f)
    extra = {str(b["osm_id"]): b for b in blds}
    for ft in gj["features"]:
        b = extra.get(str(ft["properties"]["osm_id"]))
        if b:
            for k in ("type", "subtype", "look", "era", "era_source", "epc_walls", "epc_roof", "epc_type", "dev"):
                ft["properties"][k] = b.get(k)
    with open(gj_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)

    print(f"EPC certificates (latest per UPRN): {len(epc)}")
    print(f"residential buildings: {len(res)}, with an EPC: {len(known)} "
          f"({100 * len(known) / max(1, len(res)):.0f}%)")
    print(f"developments with 2+ buildings: "
          f"{len({b['dev'] for b in blds if residential(b) and b['dev_size'] > 1})}")
    print(f"\nhold-out era prediction ({len(known)} buildings with an EPC, own EPC hidden):")
    print(f"  exact era      : {hits}/{len(known)} = {100 * hits / max(1, len(known)):.0f}%")
    print(f"  within one era : {near_hits}/{len(known)} = {100 * near_hits / max(1, len(known)):.0f}%")
    for m, (ok, n) in sorted(by_method.items()):
        print(f"  via {m:12s}: {ok}/{n} = {100 * ok / max(1, n):.0f}%")
    print("\nconfusion (rows = true era, columns = predicted):")
    print("  " + " ".join(f"{e[:9]:>9}" for e in ERAS + ["none"]))
    for t in ERAS:
        print(f"  {t[:9]:>9} " + " ".join(f"{confusion[t][g]:>9}" for g in ERAS + ["none"]))
    print("\nera source for all residential:", dict(Counter(blds[i]["era_source"] for i in res)))
    print("types:", dict(Counter(b["type"] for b in blds).most_common()))
    print("sub-types:", dict(Counter(b["subtype"] for b in blds).most_common()))
    print("looks (as in 1993):", dict(Counter(b["look"] for b in blds).most_common()))
    print("EPC walls:", dict(Counter(b["epc_walls"] for b in blds if b["epc_walls"])))
    print("EPC roofs:", dict(Counter(b["epc_roof"] for b in blds if b["epc_roof"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
