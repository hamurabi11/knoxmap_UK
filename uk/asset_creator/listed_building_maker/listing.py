"""A listed building's own evidence: the official list text and the photos of it.

    python listing.py <area data dir> [--entries N ...] [--refresh]

For every list entry in `<area>/nhle_polygons.geojson` (src/area/fetch_nhle.py),
cached under `cache/` here (fetched once; `--refresh` fetches again):

- `cache/text/<entry>.json`: the official list entry text (Historic England,
  OGL v3 since 2022). Historic England's own pages sit behind a Cloudflare
  check that refuses scripts (403), and its ArcGIS service carries no text, so
  the text is read from British Listed Buildings' copy of the same entry
  (britishlistedbuildings.co.uk/10<entry>; its robots.txt allows the pages),
  one request a second.
- `cache/photos/<entry>.json`: the photos linked to the entry, each with what
  it may be used for (USES):
  * Wikimedia Commons, through Wikidata's NHLE number (P1216) and image (P18):
    the file's own licence read from Commons;
  * Historic England's (Images of England, Enrich the List): a link only. Its
    archive terms grant a licence "solely to view ... for non-commercial
    purposes" and forbid copying or making it available, so nothing of it is
    downloaded or measured.

Run with KnoxMap's Python (`Mapping/Knoxmap/.venv/Scripts/python.exe`).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/130 Safari/537.36")
BOT = "KnoxMapUK-listed-building-maker/0.1 (Project Zomboid map research)"
BLB = "https://britishlistedbuildings.co.uk/10{entry}"
HE = "https://historicengland.org.uk/listing/the-list/list-entry/{entry}"
WAIT_S = 1.0

# What a photo may be used for, by licence (the pack is CC BY 4.0: SOURCES.md).
#   texture   - its pixels may go into a tile (CC0, public domain)
#   look      - looked at and measured (storeys, bays, colours, which way the
#               roof runs: facts), never copied or traced into a tile
#               (CC BY-SA would make the pack share-alike; CC BY needs credit
#               in the pack - kept out of tiles as Geograph's are)
#   link      - opened by a person in a browser only (Historic England's archive)
USES = [(re.compile(r"^(cc0|public domain|pd)", re.I), "texture"),
        (re.compile(r"^cc[ -]by", re.I), "look")]


def _get(url: str, agent: str = BROWSER, timeout: int = 60, tries: int = 6) -> bytes:
    """GET politely: WAIT_S before each request, and on 429 (too many - Commons,
    a city's 1,673 entries, 2026-10-22) waiting what the server asks, longer each time."""
    req = urllib.request.Request(url, headers={"User-Agent": agent, "Accept-Language": "en-GB"})
    for k in range(tries):
        time.sleep(WAIT_S)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 0) or 10 * 2 ** k)


def entries_of(area: str) -> list[dict]:
    """The area's list entries (properties of nhle_polygons.geojson), one each."""
    with open(os.path.join(area, "nhle_polygons.geojson"), encoding="utf-8") as f:
        feats = json.load(f)["features"]
    seen, out = set(), []
    for ft in feats:
        p = ft["properties"]
        if p["ListEntry"] not in seen:
            seen.add(p["ListEntry"])
            out.append(p)
    return out


def _page_text(raw: str) -> str:
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.S)
    body = re.sub(r"<br\s*/?>", "\n", body)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", body))
    return re.sub(r"[ \t\r\f\v]*\n[\s]*", "\n", text)


def parse_blb(raw: str) -> dict:
    """The list text and the fields beside it, from a British Listed Buildings page."""
    text = _page_text(raw)
    out = {}
    m = re.search(r"\nDescription\n(.*?)\n(?:External Links|Recommended Books|Sponsored Link|"
                  r"Other nearby listed buildings)\n", text, re.S)
    out["description"] = m.group(1).strip() if m else ""
    for field in ("Entry Name", "Listing Date", "Last Amended", "Grade"):
        m = re.search(rf"\n{field}:\n\s*(.+)\n", text)
        if m:
            out[field.lower().replace(" ", "_")] = m.group(1).strip()
    return out


def fetch_text(entry: int, refresh: bool = False) -> dict:
    path = os.path.join(CACHE, "text", f"{entry}.json")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    url = BLB.format(entry=entry)
    out = {"entry": entry, "source": url, "licence": "OGL v3 (Historic England list entry text)"}
    try:
        out.update(parse_blb(_get(url).decode("utf-8", "replace")))
    except Exception as e:            # noqa: BLE001 - a missing page is recorded, not fatal
        out["error"] = str(e)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def use_of(licence: str) -> str:
    return next((u for rx, u in USES if rx.search(licence or "")), "link")


def commons_photos(entries: list[int]) -> dict[int, list[dict]]:
    """Commons photos Wikidata links to each entry, with licence and use."""
    found: dict[int, list[str]] = {}
    for i in range(0, len(entries), 200):
        ids = " ".join(f'"{e}"' for e in entries[i:i + 200])
        q = f"SELECT ?id ?img WHERE {{ VALUES ?id {{ {ids} }} ?item wdt:P1216 ?id . ?item wdt:P18 ?img }}"
        res = json.loads(_get("https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(q),
                              agent=BOT))
        for b in res["results"]["bindings"]:
            name = urllib.parse.unquote(b["img"]["value"].split("FilePath/")[1])
            found.setdefault(int(b["id"]["value"]), []).append(name)
    files = sorted({n for v in found.values() for n in v})
    meta = {}
    for i in range(0, len(files), 40):
        titles = "|".join("File:" + n for n in files[i:i + 40])
        d = json.loads(_get("https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo"
                            "&iiprop=url|extmetadata|size&iiurlwidth=800&format=json&titles="
                            + urllib.parse.quote(titles), agent=BOT))
        for p in d["query"]["pages"].values():
            info = (p.get("imageinfo") or [{}])[0]
            m = info.get("extmetadata", {})
            lic = m.get("LicenseShortName", {}).get("value", "")
            meta[p["title"][5:]] = {
                "source": "Wikimedia Commons", "file": p["title"][5:],
                "page": info.get("descriptionurl"), "thumb": info.get("thumburl"),
                "licence": lic, "use": use_of(lic),
                "author": re.sub(r"<[^>]+>", "", m.get("Artist", {}).get("value", "")).strip(),
                "credit_required": m.get("AttributionRequired", {}).get("value") == "true"}
    return {e: [meta[n] for n in names if n in meta] for e, names in found.items()}


def he_photo_link(entry: int) -> dict:
    return {"source": "Historic England", "page": HE.format(entry=entry) + "?section=images",
            "licence": "Historic England Archive terms (view only, non-commercial; no copying)",
            "use": "link"}


def fetch_photos(entries: list[int], refresh: bool = False) -> dict[int, list[dict]]:
    todo = [e for e in entries if refresh or not os.path.exists(os.path.join(CACHE, "photos", f"{e}.json"))]
    got = commons_photos(todo) if todo else {}
    out = {}
    os.makedirs(os.path.join(CACHE, "photos"), exist_ok=True)
    for e in entries:
        path = os.path.join(CACHE, "photos", f"{e}.json")
        if e in todo:
            photos = got.get(e, []) + [he_photo_link(e)]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(photos, f, indent=1)
        with open(path, encoding="utf-8") as f:
            out[e] = json.load(f)
    return out


def load(area: str, refresh: bool = False, only: set[int] | None = None) -> list[dict]:
    """Every entry of the area with its text and photos (fetching what is not cached)."""
    entries = [p for p in entries_of(area) if not only or p["ListEntry"] in only]
    photos = fetch_photos([p["ListEntry"] for p in entries], refresh)
    out = []
    for k, p in enumerate(entries):
        t = fetch_text(p["ListEntry"], refresh)
        out.append({"entry": p["ListEntry"], "name": p["Name"], "grade": p["Grade"],
                    "text": t.get("description", ""), "text_source": t.get("source"),
                    "photos": photos[p["ListEntry"]], "nhle": p})
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(entries)} entries", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("area")
    ap.add_argument("--entries", type=int, nargs="*")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    rows = load(a.area, a.refresh, set(a.entries) if a.entries else None)
    no_text = [r["entry"] for r in rows if not r["text"]]
    uses = {}
    for r in rows:
        for p in r["photos"]:
            uses[p["use"]] = uses.get(p["use"], 0) + 1
    print(f"{len(rows)} entries; text for {len(rows) - len(no_text)} (none: {no_text[:10]})")
    print(f"photos by use: {uses} (link = Historic England's page, one an entry)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
