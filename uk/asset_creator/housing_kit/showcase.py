"""Example houses for each housing style, rendered from the compiled lots.
   python ../../asset_creator/housing_kit/showcase.py out_dir [per_style] [style,...]   (from Mapping/Knoxmap/)"""
import json
import os
import sys

sys.path.insert(0, 'tools')
sys.path.insert(0, '.')
import render_lots as R  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

STYLES = [  # (style, label)
    ("uk_vernacular_timber", "Vernacular: timber frame (vern_timber; listed examples)"),
    ("uk_georgian_town", "Georgian town house (georgian_townhouse; listed examples)"),
    ("uk_victorian_terrace", "Victorian terrace (vic_terrace)"),
    ("uk_edwardian", "Edwardian terrace (edw_terrace)"),
    ("uk_interwar_30s", "Inter-war detached (iw_detached)"),
    ("uk_postwar_50s", "Post-war private (pw_private)"),
    ("uk_estate_70s", "1960s-70s estate house (e70_house)"),
    ("uk_estate_80s", "1980s-90s estate house (e80_house; stained timber or white uPVC windows)"),
    # K09.26 (2026-10-19): council housing, flats, bungalows, 1980s terraces; K09.23 mock-Tudor
    ("uk_council_iw", "Inter-war council: brick (iw_council)"),
    ("uk_council_iw_rough", "Inter-war council: roughcast (iw_council)"),
    ("uk_council_pw", "Post-war council (pw_council)"),
    ("uk_council_70s", "1960s-70s council (e70_council)"),
    ("uk_social_80s", "1980s housing association (e80_social)"),
    ("uk_terrace_80s", "1980s starter terrace (e80_terrace; stained timber or uPVC)"),
    ("uk_bungalow_iw", "1930s bungalow (iw_bungalow)"),
    ("uk_bungalow_pw", "1950s-60s bungalow (pw_bungalow)"),
    ("uk_bungalow_70s", "1960s-70s bungalow (e70_bungalow)"),
    ("uk_flats_pw", "Post-war flats (flats_postwar)"),
    ("uk_flats_80s", "1980s flats: buff (flats_modern)"),
    ("uk_flats_80s_red", "1980s flats: red (flats_modern)"),
    ("uk_edwardian_tudor", "Edwardian semi: mock-Tudor gables (edw_semi)"),
    ("uk_interwar_tudor", "1930s semi: mock-Tudor gables (iw_semi)"),
]
MAPS = (os.environ.get("SHOWCASE_MAPS") or "output/Odiham_town_graph,output/Alton_town_graph").split(",")
out_dir = sys.argv[1]
per = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ONLY = sys.argv[3].split(',') if len(sys.argv) > 3 else None
os.makedirs(out_dir, exist_ok=True)

# Unlisted ones are rare (the listed maker builds most): its own styles of the same kit stand in.
ALSO = {"uk_timber_white__tile__casement_leaded": "uk_vernacular_timber",
        "uk_timber_brick__tile__casement_leaded": "uk_vernacular_timber",
        "uk_brick_flemish__tile__sash_georgian": "uk_georgian_town",
        "uk_estate_80s_timber": "uk_estate_80s", "uk_terrace_80s_upvc": "uk_terrace_80s"}
cands = {s: [] for s, _ in STYLES}
for md in MAPS:
    # Build's own record of each building it placed (style, box, storeys):
    # every building, with a roof plan or not.
    import csv
    with open(os.path.join(md, os.path.basename(md) + "_placements.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = ALSO.get(row["style"], row["style"])
            if st not in cands:
                continue
            x0, y0, w, h = (int(row[k]) for k in ("tile_x", "tile_y", "width", "height"))
            cands[st].append((md, row["name"], x0, y0, w, h, w * h, int(row["levels"] or 1)))

rows = []
font = ImageFont.load_default()
for st, label in STYLES:
    if ONLY and st not in ONLY:
        continue
    c = cands[st]
    if not c:
        print(st, "none built")
        continue
    pick = []
    # Two-storey houses of a typical size first, then any (a bungalow style's)
    for pool in ([k for k in c if k[7] >= 2], c):
        pool = sorted(pool, key=lambda k: k[6])
        mid = len(pool) // 2
        for k in pool[mid:] + pool[:mid]:                # typical size first
            if len(pick) == per:
                break
            if 6 <= max(k[4], k[5]) <= 18 and k not in pick                     and all(abs(k[2] - p[2]) + abs(k[3] - p[3]) > 30 or k[0] != p[0] for p in pick):
                pick.append(k)
    tiles = []
    for md, oid, x0, y0, w, h, area, lv in pick:
        cx, cy = x0 + w // 2, y0 + h // 2
        S = 40
        # Drawn at the sprites' own size, then the whole picture shrunk once: render_lots
        # shrinks sprite by sprite below that, which draws a line at every square's edge
        # that the game does not (K09.25, 2026-10-19)
        im = R.render(md, cx - S // 2, cy - S // 2, S, S, scale=2.0)
        if im is not None:
            im = im.resize((round(im.width * 0.375), round(im.height * 0.375)), Image.LANCZOS)
        if im is None:
            continue
        W_, H_ = im.size
        im = im.crop((W_ // 2 - 330, H_ // 2 - 330, W_ // 2 + 330, H_ // 2 + 230)).convert("RGB")
        tiles.append(im)
        print(st, os.path.basename(md), oid, (x0, y0, w, h), "levels", lv)
    if not tiles:
        continue
    row = Image.new("RGB", (sum(t.width for t in tiles) + 10 * (len(tiles) - 1), tiles[0].height + 30), (24, 26, 28))
    ImageDraw.Draw(row).text((10, 8), label + f"  - style {st}", fill=(230, 230, 230), font=font)
    x = 0
    for t in tiles:
        row.paste(t, (x, 30))
        x += t.width + 10
    row.save(os.path.join(out_dir, f"{st}.png"))
    rows.append(row)

if rows:
    W_ = max(r.width for r in rows)
    sheet = Image.new("RGB", (W_, sum(r.height for r in rows) + 10 * len(rows)), (24, 26, 28))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 10
    sheet.save(os.path.join(out_dir, "all_styles.png"))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(out_dir, "all_styles_half.png"))
    print("wrote", out_dir)
