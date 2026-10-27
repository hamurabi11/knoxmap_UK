"""Reference photos for the commercial kit, from Wikimedia Commons (PLAN.md).

    python fetch_refs.py [topic ...] [--per N]   (KnoxMap's Python)

For each topic (TOPICS: a folder and its search words) it asks Commons'
search for photos, reads each file's own licence, and keeps only free ones:
CC0, public domain and CC BY may be used for textures (the pack is CC BY,
credited); CC BY-SA (most Geograph photos) is for looking and measuring only,
never copied into a tile (as the listed maker's photos: README "Pictures and
text"). Anything else is dropped. Writes reference/<topic>/<n>.jpg (1024 px
wide at most), reference/<topic>/sources.json (title, page, author, licence,
date, use) and reference/<topic>/contact.png. Reruns skip files already got.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference")
UA = "KnoxMapUK-reference/0.1 (Project Zomboid map modding; personal use)"
API = "https://commons.wikimedia.org/w/api.php"
WIDTH = 1024
MAX_BYTES = 15_000_000   # originals are fetched (Commons rate-limits made-on-demand thumbnails), up to this size
PER = 6

# topic -> search words (each a Commons search; the first PER free photos of each kept)
TOPICS = {
    "high_street_1990s": ["high street 1990 England shops", "high street 1992 shops", "town centre 1990s England"],
    "chains_1990s": ["Woolworths shop front England", "Our Price records shop", "Rumbelows shop", "Dixons shop front",
                     "Boots chemist shop front", "WH Smith shop front", "Victoria Wine shop", "Threshers off licence",
                     "Blockbuster video shop UK", "Midland Bank branch", "Abbey National branch", "Littlewoods store",
                     "C&A store England", "Lunn Poly", "Wimpy restaurant", "Spar shop England"],
    "shopfront_traditional": ["Victorian shopfront England", "traditional shop front fascia", "shopfront Hampshire",
                              "shop front Alton Hampshire", "shop front Odiham"],
    "shopfront_modern": ["shopping parade England", "1960s shopping parade", "aluminium shopfront", "shop roller shutter"],
    "supermarkets": ["Kwik Save", "Safeway supermarket United Kingdom", "Tesco superstore", "Sainsbury's supermarket",
                     "Asda superstore", "Morrisons supermarket", "Waitrose supermarket", "Iceland frozen food shop",
                     "supermarket trolley shelter", "bottle bank recycling UK"],
    "industrial": ["industrial estate units England", "industrial unit cladding", "roller shutter door warehouse",
                   "palisade fence", "corrugated asbestos cement roof", "warehouse England",
                   "maltings England", "Alton brewery"],
    "farm": ["Dutch barn England", "weatherboarded barn Hampshire", "black weatherboarded barn",
             "granary staddle stones", "corrugated iron barn", "cattle shed farm England", "hay bales stack England"],
    "petrol_garage": ["petrol station England", "petrol station 1990s", "garage MOT England", "car repair garage England"],
    # second pass (2026-10-19): Commons categories, and photos filed under the 1990s
    "period_1990s": ['deepcategory:"1990s in England" shop', 'deepcategory:"1990s in England" high street',
                     'deepcategory:"1990s in England" supermarket', 'deepcategory:"1980s in England" shop',
                     'deepcategory:"1990s in England" petrol', 'deepcategory:"1990s in England" farm'],
    "chain_stores": ['incategory:"Former Woolworths Group stores in England"', 'incategory:"Kwik Save"',
                     'incategory:"Kwiksave in the United Kingdom"', 'incategory:"Safeway (UK)"',
                     'incategory:"Safeway (UK) petrol stations"'],
    "industrial_estates": ['incategory:"Industrial estates in England"', 'deepcategory:"Industrial estates in England" unit'],
}
TEXTURE = ("cc0", "public domain", "pd", "cc by 2.0", "cc by 3.0", "cc by 4.0", "cc by 2.5", "cc by 1.0")


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _plain(html: str | None) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def use_of(licence: str) -> str | None:
    """'texture' (may go into a tile), 'look' (share-alike: look only) or None (not free)."""
    lic = licence.lower().strip()
    if lic.startswith("cc by-sa") or lic.startswith("cc-by-sa"):
        return "look"
    if lic.startswith(TEXTURE) or lic in TEXTURE:
        return "texture"
    return None


def _download(url: str, tries: int = 4):
    """The picture, waiting longer after each 429 (Commons' rate limit)."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return None
            time.sleep(10 * (k + 1))
        except Exception:
            return None
    return None


def search(words: str, n: int) -> list[dict]:
    data = _get({"action": "query", "generator": "search", "gsrnamespace": 6,
                 "gsrsearch": f"{words} filetype:bitmap", "gsrlimit": 30,
                 "prop": "imageinfo", "iiprop": "url|extmetadata|size"})
    pages = sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        licence = _plain((meta.get("LicenseShortName") or {}).get("value"))
        use = use_of(licence)
        if not use or (info.get("width") or 0) < 640 or (info.get("size") or 0) > MAX_BYTES:
            continue
        out.append({"title": p["title"], "page": info.get("descriptionurl"), "image": info.get("url"),
                    "author": _plain((meta.get("Artist") or {}).get("value"))[:200], "licence": licence,
                    "date": _plain((meta.get("DateTimeOriginal") or {}).get("value"))[:40],
                    "description": _plain((meta.get("ImageDescription") or {}).get("value"))[:300],
                    "search": words, "use": use})
        if len(out) >= n:
            break
    return out


def fetch(topic: str, per: int) -> None:
    folder = os.path.join(REF, topic)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "sources.json")
    have = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    titles = {h["title"] for h in have}
    for words in TOPICS[topic]:
        try:
            finds = search(words, per)
        except Exception as e:  # a failed search is reported and skipped
            print(f"  {topic}: '{words}' failed: {e}")
            continue
        for f in finds:
            if f["title"] in titles:
                continue
            name = f"{len(have) + 1:03d}.jpg"
            im = _download(f["image"])
            if im is None:
                print(f"  {f['title']}: not downloaded")
                continue
            im.thumbnail((WIDTH, WIDTH))
            im.save(os.path.join(folder, name), quality=88)
            f["file"] = name
            have.append(f)
            titles.add(f["title"])
            json.dump(have, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            time.sleep(4.0)                      # gently: Commons rate-limits a quick run
        print(f"  {topic}: '{words}' -> {len(finds)}", flush=True)
        time.sleep(3.0)
    json.dump(have, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    contact(folder, have)


def contact(folder: str, items: list[dict], cell: int = 240) -> None:
    """Thumbnails in a grid, the file, licence and use under each."""
    cols = 6
    rows = max(1, (len(items) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 30)), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        im = Image.open(os.path.join(folder, it["file"]))
        im.thumbnail((cell - 8, cell - 8))
        x, y = (i % cols) * cell, (i // cols) * (cell + 30)
        sheet.paste(im, (x + 4, y + 4))
        draw.text((x + 4, y + cell), f"{it['file']} {it['licence'][:14]} {it['use']}", fill=(230, 230, 230))
        draw.text((x + 4, y + cell + 13), it["title"][5:40], fill=(170, 170, 170))
    sheet.save(os.path.join(folder, "contact.png"))


def main(argv: list[str]) -> int:
    per = PER
    if "--per" in argv:
        k = argv.index("--per")
        per = int(argv[k + 1])
        del argv[k:k + 2]
    for topic in argv or list(TOPICS):
        print(topic)
        fetch(topic, per)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
