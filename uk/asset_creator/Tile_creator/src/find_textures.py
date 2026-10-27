"""Find CC0 and public-domain pictures of UK building materials, by EPC period.

    python find_textures.py --epc "<construction_age_band>" "<walls_description>"
    python find_textures.py --query "pebbledash" [--query ...]
    python find_textures.py --test        (the periods below, as a trial)
    python find_textures.py --towns <KnoxMap map folder | lat,lon>   (photos of the main towns round it)

Searches, for each material, four places whose licence is known per item
(only CC0 and public domain kept, so anything found can go in the free pack
with no share-alike - APPROACH.md section 5):
- Openverse (api.openverse.org; Flickr, Wikimedia and others), asked for
  license=cc0,pdm only;
- Wikimedia Commons (files' own LicenseShortName read; CC0 and public
  domain kept, everything else dropped);
- ambientCG (all CC0: scanned and made PBR materials);
- Poly Haven (all CC0; its texture list matched by name and tag).

What a period means in materials is a lookup table (EPC_MATERIALS: an
age band and a wall type from the EPC register -> UK materials of that
time and their search words), not a guess per query. Writes
reference/found/<slug>/results.json (each find: source, title, page, image,
thumbnail, licence, creator, size) and contact.png (thumbnails, source and
licence under each). --towns: OpenStreetMap's biggest places within 15 km of
an area and Wikimedia Commons photos geotagged in each (any licence, marked;
for ideas, not for the pack), reference/found/towns_<area>/. Downloads thumbnails only; a picture is fetched whole
only when chosen, and recorded in Mapping/SOURCES.md before use.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FOUND = os.path.join(HERE, "..", "reference", "found")
UA = "KnoxMapUK-texture-search/0.1 (Project Zomboid map modding; personal use)"
PER_SOURCE = 6

# EPC (construction_age_band, walls_description) -> what such houses in
# north Hampshire are built of, and the words to search for each. The age
# bands are the register's own ("England and Wales: 1930-1949"); the wall
# types its first words ("Solid brick", "Cavity wall", "Timber frame",
# "Sandstone or limestone", "System built"). A cavity-wall house is brick
# outside; render, pebbledash and tile-hanging are what changes by decade.
EPC_MATERIALS = [
    ("before 1900", "Solid brick", "Georgian and Victorian red brick, flint, slate",
     ["red brick wall flemish bond", "flint wall", "slate roof", "clay roof tiles"]),
    ("before 1900", "Timber frame", "timber-framed cottages, lime render, plain tiles, thatch",
     ["timber framed wall", "lime render texture", "thatched roof"]),
    ("1900-1929", "Solid brick", "Edwardian red brick, roughcast, clay tiles",
     ["edwardian brick house", "roughcast render", "clay roof tiles"]),
    ("1930-1949", "Cavity wall", "interwar semis: pebbledash, render, clay or concrete tiles",
     ["pebbledash texture", "pebbledash house", "rendered wall texture", "stucco wall"]),
    ("1950-1966", "Cavity wall", "post-war council houses: plain brick, concrete tiles",
     ["brick wall texture", "concrete roof tiles"]),
    ("1967-1975", "Cavity wall", "1960s-70s estates: buff and orange brick, tile-hanging",
     ["tile hanging wall", "buff brick wall", "orange brick wall"]),
    ("1976-1999", "Cavity wall", "1980s-90s estates: buff or red brick, concrete tiles",
     ["buff brick wall", "concrete roof tiles"]),
    ("", "Timber frame", "modern timber frame: brick slips, render, weatherboarding",
     ["weatherboard cladding", "rendered wall texture"]),
]


def _get(url: str, binary: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    time.sleep(0.5)                                         # polite
    return data if binary else json.loads(data)


def openverse(q: str) -> list[dict]:
    d = _get("https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(
        {"q": q, "license": "cc0,pdm", "page_size": PER_SOURCE}))
    return [{"source": "Openverse/" + (r.get("provider") or ""), "title": r.get("title") or "",
             "page": r.get("foreign_landing_url"), "image": r.get("url"), "thumb": r.get("thumbnail"),
             "licence": f"{r.get('license', '').upper()} {r.get('license_version') or ''}".strip(),
             "creator": r.get("creator"), "size": [r.get("width"), r.get("height")]} for r in d.get("results", [])]


def commons(q: str) -> list[dict]:
    d = _get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": 6,
        "gsrlimit": 30, "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": 256,
        "iiextmetadatafilter": "LicenseShortName|Artist", "format": "json"}))
    out = []
    for p in sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
        ii = p["imageinfo"][0]
        lic = ii["extmetadata"].get("LicenseShortName", {}).get("value", "")
        if not (lic.upper().startswith(("CC0", "PD", "PUBLIC DOMAIN"))):
            continue
        out.append({"source": "Wikimedia Commons", "title": p["title"][5:], "page": ii.get("descriptionurl"),
                    "image": ii.get("url"), "thumb": ii.get("thumburl"), "licence": lic,
                    "creator": re.sub("<[^>]+>", "", ii["extmetadata"].get("Artist", {}).get("value", "")),
                    "size": [ii.get("width"), ii.get("height")]})
        if len(out) == PER_SOURCE:
            break
    return out


def ambientcg(q: str) -> list[dict]:
    d = _get("https://ambientcg.com/api/v2/full_json?" + urllib.parse.urlencode(
        {"type": "Material", "q": q, "limit": PER_SOURCE, "include": "displayData,imageData,tagData"}))
    return [{"source": "ambientCG", "title": a.get("displayName"), "page": a.get("shortLink"),
             "image": None, "thumb": (a.get("previewImage") or {}).get("256-PNG"), "licence": "CC0 1.0",
             "creator": "ambientCG (Lennart Demes)", "size": None, "tags": a.get("tags")}
            for a in d.get("foundAssets", [])]


_POLYHAVEN = None


def polyhaven(q: str) -> list[dict]:
    global _POLYHAVEN
    if _POLYHAVEN is None:
        _POLYHAVEN = _get("https://api.polyhaven.com/assets?t=textures")
    words = [w for w in q.lower().split() if len(w) > 2 and w not in {"wall", "texture", "house"}]
    hits = []
    for k, a in _POLYHAVEN.items():
        text = " ".join([k, a.get("name", "")] + a.get("tags", []) + a.get("categories", [])).lower()
        score = sum(w in text for w in words)
        if words and score == len(words):                     # every word (one in common matched river beds)
            hits.append((score, a.get("download_count", 0), k, a))
    hits.sort(reverse=True)
    return [{"source": "Poly Haven", "title": a.get("name"), "page": f"https://polyhaven.com/a/{k}", "image": None,
             "thumb": f"https://cdn.polyhaven.com/asset_img/thumbs/{k}.png?width=256", "licence": "CC0 1.0",
             "creator": ", ".join(a.get("authors", {})), "size": a.get("dimensions"), "tags": a.get("tags")}
            for _s, _d, k, a in hits[:PER_SOURCE]]


SOURCES = (openverse, commons, ambientcg, polyhaven)


def search(q: str) -> list[dict]:
    out = []
    for src in SOURCES:
        try:
            out += [dict(r, query=q, via=src.__name__) for r in src(q)]
        except Exception as e:                              # one source down is not the search down
            print(f"  {src.__name__}: {e}")
    return out


def contact(results: list[dict], path: str, title: str) -> None:
    cell, label = 170, 34
    queries = list(dict.fromkeys(r["query"] for r in results))
    cols = max(1, max(sum(r["query"] == q for r in results) for q in queries)) if results else 1
    img = Image.new("RGB", (cols * cell, 24 + len(queries) * (cell + label + 18)), (32, 32, 32))
    d = ImageDraw.Draw(img)
    d.text((6, 6), title, fill=(255, 255, 255))
    y = 24
    for q in queries:
        d.text((6, y), f"'{q}'", fill=(255, 220, 120))
        for i, r in enumerate([r for r in results if r["query"] == q]):
            x = i * cell
            try:
                try:
                    th = Image.open(io.BytesIO(_get(r["thumb"], binary=True))).convert("RGB")
                except Exception:                           # Openverse's own thumbnail can fail: the source's
                    th = Image.open(io.BytesIO(_get(r["image"], binary=True))).convert("RGB")
                th.thumbnail((cell - 8, cell - 8))
                img.paste(th, (x + 4, y + 16))
            except Exception:
                d.text((x + 8, y + 60), "(no thumbnail)", fill=(200, 80, 80))
            d.text((x + 4, y + 16 + cell - 4), r["source"][:26], fill=(170, 200, 255))
            d.text((x + 4, y + 30 + cell - 4), f"{r['licence'][:12]} {(r['title'] or '')[:12]}", fill=(200, 200, 200))
        y += cell + label + 18
    img.save(path)


def run(slug: str, title: str, queries: list[str]) -> list[dict]:
    os.makedirs(os.path.join(FOUND, slug), exist_ok=True)
    results = []
    for q in queries:
        found = search(q)
        print(f"  '{q}': " + ", ".join(f"{s.__name__} {sum(r['via'] == s.__name__ for r in found)}" for s in SOURCES))
        results += found
    json.dump({"title": title, "queries": queries, "results": results},
              open(os.path.join(FOUND, slug, "results.json"), "w", encoding="utf-8"), indent=1)
    contact(results, os.path.join(FOUND, slug, "contact.png"), title)
    return results


# ---- Pictures of the main towns round an area, for ideas ---------------------
# OpenStreetMap's places (Overpass) within TOWN_REACH of the area's centre,
# biggest first (city, town, village; then population); for each, Wikimedia
# Commons' photos geotagged within PHOTO_RADIUS of its centre, those whose
# titles name a street, house, cottage, church... first. Any licence is kept
# here - they are for looking at, not for the pack - but each is marked.
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
TOWN_REACH_M = 15000
PHOTO_RADIUS_M = 1200
TOWNS = 7
PHOTOS = 10
RANK = {"city": 0, "town": 1, "village": 2}
STREET_WORDS = ("street", "high st", "house", "cottage", "terrace", "church", "road", "lane", "inn",
                "pub", "shop", "square", "row", "listed", "hall", "farm", "green", "bridge")


def towns_near(lat: float, lon: float) -> list[dict]:
    q = (f'[out:json][timeout:60];node(around:{TOWN_REACH_M},{lat},{lon})'
         f'[place~"^(city|town|village)$"][name];out;')
    nodes, err = None, None
    for url in OVERPASS:                                # the main server is often busy: its mirrors in turn
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                nodes = json.loads(r.read())["elements"]
            break
        except Exception as e:
            err = e
    if nodes is None:
        raise RuntimeError(f"no Overpass server answered ({err})")
    def pop(n):
        try:
            return int(re.sub(r"[^0-9]", "", n["tags"].get("population", "")) or 0)
        except ValueError:
            return 0
    # The area's own place (the one nearest its centre: Odiham is a village) first, then the biggest.
    home = min(nodes, key=lambda n: (n["lat"] - lat) ** 2 + (n["lon"] - lon) ** 2) if nodes else None
    nodes.sort(key=lambda n: (n is not home, RANK.get(n["tags"]["place"], 9), -pop(n)))
    return [{"name": n["tags"]["name"], "place": n["tags"]["place"], "population": pop(n),
             "lat": n["lat"], "lon": n["lon"]} for n in nodes[:TOWNS]]


def town_photos(town: dict) -> list[dict]:
    d = _get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "generator": "geosearch", "ggscoord": f"{town['lat']}|{town['lon']}",
        "ggsradius": PHOTO_RADIUS_M, "ggsnamespace": 6, "ggslimit": 60, "prop": "imageinfo",
        "iiprop": "url|extmetadata|size", "iiurlwidth": 256, "iiextmetadatafilter": "LicenseShortName|Artist",
        "format": "json"}))
    out = []
    for p in d.get("query", {}).get("pages", {}).values():
        if "imageinfo" not in p:
            continue
        ii = p["imageinfo"][0]
        if not p["title"].lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lic = ii["extmetadata"].get("LicenseShortName", {}).get("value", "")
        score = sum(w in p["title"].lower() for w in STREET_WORDS)
        out.append({"source": "Wikimedia Commons", "title": p["title"][5:], "page": ii.get("descriptionurl"),
                    "image": ii.get("url"), "thumb": ii.get("thumburl"), "licence": lic,
                    "creator": re.sub("<[^>]+>", "", ii["extmetadata"].get("Artist", {}).get("value", "")),
                    "size": [ii.get("width"), ii.get("height")], "query": town["name"], "via": "commons-geo",
                    "_score": score})
    out.sort(key=lambda r: -r["_score"])
    return out[:PHOTOS]


def towns(lat: float, lon: float, slug: str) -> None:
    found = towns_near(lat, lon)
    print("towns:", ", ".join(f"{t['name']} ({t['place']}{', ' + str(t['population']) if t['population'] else ''})"
                              for t in found))
    results = []
    for t in found:
        photos = town_photos(t)
        print(f"  {t['name']}: {len(photos)} photos ({sum(p['licence'].upper().startswith(('CC0', 'PD', 'PUBLIC')) for p in photos)} CC0 / public domain)")
        results += photos
    out = os.path.join(FOUND, "towns_" + slug)
    os.makedirs(out, exist_ok=True)
    json.dump({"towns": found, "results": results}, open(os.path.join(out, "results.json"), "w", encoding="utf-8"), indent=1)
    contact(results, os.path.join(out, "contact.png"),
            f"Towns round {slug}: Wikimedia Commons photos for ideas (licence under each; reference only)")
    print(f"  -> reference/found/towns_{slug}/")


def area_centre(arg: str) -> tuple[float, float, str]:
    """A KnoxMap map folder (its info.json's box) or 'lat,lon'."""
    if os.path.isdir(arg):
        info = next(f for f in os.listdir(arg) if f.endswith("_info.json"))
        b = json.load(open(os.path.join(arg, info), encoding="utf-8"))["bbox"]
        return (b["south"] + b["north"]) / 2, (b["west"] + b["east"]) / 2, os.path.basename(arg.rstrip("/\\"))
    lat, lon = (float(v) for v in arg.split(","))
    return lat, lon, f"{lat:.3f}_{lon:.3f}"


def for_epc(age: str, walls: str) -> list[tuple]:
    """The table rows an EPC age band and wall description match."""
    age = age.replace("England and Wales: ", "")
    def in_band(row_band):
        if not row_band:
            return True
        if row_band == age or row_band in age:
            return True
        m = re.match(r"(\d{4})-(\d{4})", row_band)
        y = re.match(r"(\d{4})", age)
        return bool(m and y and int(m.group(1)) <= int(y.group(1)) <= int(m.group(2)))
    return [r for r in EPC_MATERIALS if in_band(r[0]) and walls.lower().startswith(r[1].lower())]


def main() -> int:
    args = sys.argv[1:]
    if "--towns" in args:
        lat, lon, slug = area_centre(args[args.index("--towns") + 1])
        towns(lat, lon, slug)
        return 0
    if "--test" in args:
        rows = EPC_MATERIALS
    elif "--epc" in args:
        i = args.index("--epc")
        rows = for_epc(args[i + 1], args[i + 2])
        if not rows:
            print("no materials in the table for that period and wall type")
            return 1
    else:
        qs = [args[i + 1] for i, a in enumerate(args) if a == "--query"]
        rows = [("", "", "queries", qs)]
    for band, walls, what, queries in rows:
        slug = re.sub(r"[^a-z0-9]+", "_", f"{band} {walls} {what}".lower()).strip("_")[:60]
        print(f"{band or 'any age'} / {walls or 'any walls'}: {what}")
        res = run(slug, f"{band or 'any age'} / {walls or 'any walls'}: {what}", queries)
        print(f"  -> {len(res)} CC0 / public domain finds, reference/found/{slug}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
