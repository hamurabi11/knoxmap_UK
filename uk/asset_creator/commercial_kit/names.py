"""The high street's names as they stood in 1993, tweaked (user, 2026-10-19; PLAN.md "Names and signs").

    python names.py <map dir>     every shop on a map and the sign it would get (a check; writes nothing)

A lookup, not a guess:
- a shop whose OSM / FSA name is a chain gets that chain's 1993 sign (Tesco -> Tisco);
- a brand that opened after 1993 (Starbucks...) takes a 1993 business of its trade;
- a shop whose trade the data gives gets a chain of that trade (at most one of each
  chain a town) or an independent;
- a shop whose trade the data does not give (most of the non-domestic EPCs say only
  "retail") takes a trade from a 1993 market town's mix (TRADE_MIX), by its id;
- hotels, surgeries and churches get no fascia (they have their own signs or none); a
  pub gets its own name's board in its brewery's livery (pub_<key>, export_regional.uk_pubs).

The signs (`signs()`, in sheet order: a sign's place fixes its tile ids, so once
the sheet is drawn new ones go in ADDED, at the end) are drawn by
fascias.py; export_regional.py writes the chosen one as the building's `fascia`.

Each chain's sign looks like it the way a discount supermarket's own brand looks
like the premium one beside it: its colours, layout and kind of lettering, a name
a letter or two off. Nothing is traced: the lettering is a free font (DejaVu,
fonts/LICENSE_DEJAVU).
"""
from __future__ import annotations

import hashlib
import re

# Fonts: "sans" bold, "serif" bold, "italic" (sans bold oblique), "script" (serif bold italic).
# Styles: "plain" letters on the ground; "stripe" an accent band under them; "box" the
# name in an accent box; "gilt" a traditional painted board (a bead round it, shaded letters).
W, CREAM, GOLD = (250, 250, 248), (236, 222, 176), (214, 178, 92)

# key: (text, trade, ground, ink, font, accent, style). The user's supermarkets first.
CHAINS = {
    "tisco": ("TISCO", "supermarket", (246, 246, 242), (16, 52, 140), "sans", (212, 32, 40), "stripe"),
    "lainburys": ("LAINBURYS", "supermarket", (240, 238, 232), (206, 88, 10), "serif", None, "plain"),
    "norrisons": ("NORRISONS", "supermarket", (252, 210, 40), (20, 92, 52), "sans", None, "plain"),
    "greenes": ("Greene's", "supermarket", (30, 112, 58), W, "script", None, "plain"),
    "safeway": ("SAFEWAY", "supermarket", (236, 236, 232), (196, 26, 34), "sans", (196, 26, 34), "stripe"),
    "kwiksave": ("KWIK SAVE", "supermarket", (250, 206, 20), (204, 30, 30), "italic", None, "plain"),
    # in the towns' OSM, proposed (PLAN.md: to confirm)
    "waitrows": ("WAITROWS", "supermarket", (244, 244, 240), (38, 92, 34), "serif", None, "plain"),
    "marks_spender": ("MARKS & SPENDER", "supermarket", (22, 70, 52), (240, 236, 220), "serif", None, "plain"),
    "greenland": ("GREENLAND", "frozen_food", (228, 70, 30), W, "italic", None, "plain"),
    "wessex_coop": ("WESSEX CO-OP", "convenience", (240, 244, 248), (20, 80, 160), "sans", None, "plain"),
    "spur": ("SPUR", "convenience", (16, 120, 64), W, "sans", (212, 40, 40), "box"),
    # the high street's chains
    "woolies": ("WOOLIES", "variety", (200, 20, 32), W, "sans", None, "plain"),
    "littlewards": ("LITTLEWARDS", "variety", (30, 40, 110), W, "sans", None, "plain"),
    "boats": ("Boats", "chemist", (16, 60, 150), W, "script", None, "plain"),
    "supadrug": ("SUPADRUG", "chemist", (40, 40, 120), W, "sans", (220, 40, 120), "stripe"),
    "wh_smythe": ("WH SMYTHE", "newsagent", (20, 40, 100), W, "sans", None, "plain"),
    "martyns": ("MARTYNS", "newsagent", (200, 30, 40), W, "sans", None, "plain"),
    "our_prize": ("our prize", "music", (40, 120, 180), W, "italic", None, "plain"),
    "hmb": ("HMB", "music", (200, 20, 40), W, "sans", None, "plain"),
    "blockbluster": ("BLOCKBLUSTER VIDEO", "video", (20, 40, 120), (252, 200, 30), "sans", None, "box"),
    "dixens": ("Dixens", "electrical", (20, 30, 90), (240, 70, 40), "italic", None, "plain"),
    "curries": ("CURRIES", "electrical", (200, 30, 40), W, "sans", None, "plain"),
    "rumblers": ("RUMBLERS", "electrical", (30, 60, 140), W, "sans", None, "plain"),
    "victory_wine": ("Victory Wine", "off_licence", (20, 60, 40), GOLD, "script", None, "gilt"),
    "thrashers": ("THRASHERS", "off_licence", (120, 20, 40), W, "serif", None, "plain"),
    "griggs": ("GRIGGS", "bakery", (24, 70, 140), (250, 210, 40), "sans", None, "plain"),
    "barton": ("BARTON", "clothes", (20, 20, 22), W, "serif", None, "plain"),
    "dorothy_parkins": ("DOROTHY PARKINS", "clothes", (240, 236, 236), (150, 20, 82), "serif", None, "plain"),
    "c_and_e": ("C&E", "clothes", (20, 60, 150), W, "sans", (200, 30, 40), "box"),
    "clerks": ("CLERKS", "shoes", (20, 30, 60), W, "serif", None, "plain"),
    "mothercore": ("mothercore", "baby", (30, 90, 170), W, "sans", None, "plain"),
    "rattners": ("RATTNERS", "jewelry", (20, 20, 60), GOLD, "serif", None, "gilt"),
    "h_samuels": ("H. SAMUELS", "jewelry", (110, 20, 30), GOLD, "serif", None, "gilt"),
    "highland_bank": ("HIGHLAND BANK", "bank", (230, 230, 228), (190, 20, 40), "sans", (20, 40, 110), "stripe"),
    "nateast": ("NatEast", "bank", (240, 240, 240), (60, 40, 120), "sans", (200, 30, 40), "stripe"),
    "barkleys": ("BARKLEYS", "bank", (0, 120, 190), W, "sans", None, "plain"),
    "floyds": ("FLOYDS BANK", "bank", (20, 90, 60), W, "serif", None, "plain"),
    "tbs": ("TBS", "bank", (40, 110, 190), W, "sans", (250, 210, 40), "stripe"),
    "abbey_rational": ("ABBEY RATIONAL", "bank", (200, 30, 30), W, "sans", None, "plain"),
    "halifix": ("HALIFIX", "bank", (20, 50, 130), W, "sans", None, "plain"),
    "lunn_polly": ("LUNN POLLY", "travel_agency", (240, 200, 20), (200, 30, 30), "sans", None, "plain"),
    "thomas_crook": ("THOMAS CROOK", "travel_agency", (20, 50, 120), W, "serif", None, "plain"),
    "ladbrooks": ("LADBROOKS", "bookmaker", (200, 20, 30), W, "sans", None, "plain"),
    "william_hall": ("WILLIAM HALL", "bookmaker", (20, 50, 120), W, "sans", None, "plain"),
    "clintons_cardz": ("CLINTONS CARDZ", "gift", (120, 20, 80), W, "serif", None, "plain"),
    "wimpey": ("WIMPEY", "fast_food", (200, 20, 30), W, "sans", None, "plain"),
    "kfg": ("KENTUCKY FRIED GOOSE", "fast_food", (196, 24, 32), W, "sans", None, "plain"),
    "pizza_hutt": ("PIZZA HUTT", "pizza", (190, 20, 30), W, "script", None, "plain"),
    "dominoes": ("DOMINOES PIZZA", "pizza", (20, 90, 160), W, "sans", (200, 30, 40), "box"),
    # the game's own burger chain: its own front and neon (shopfronts.SPIFFOS_FRONT); this
    # board is only drawn for its place in the sheets
    "spiffos": ("Spiffo's", "spiffos", (196, 24, 32), (252, 214, 60), "script", None, "plain"),
    "post_office": ("POST OFFICE", "post_office", (200, 20, 30), (250, 210, 40), "sans", None, "plain"),
    "halfwords": ("HALFWORDS", "car_parts", (230, 120, 20), (8, 12, 40), "sans", None, "plain"),
    "texeco": ("TEXECO", "fuel", (200, 20, 30), W, "sans", None, "plain"),
    "kwik_fix": ("KWIK FIX", "car_repair", (20, 40, 110), (250, 210, 40), "italic", None, "plain"),
    "holland_barratt": ("HOLLAND & BARRATT", "health_food", (20, 80, 50), W, "serif", None, "plain"),
    "peacox": ("PEACOX", "clothes", (20, 20, 22), W, "sans", None, "plain"),
}

# Independents: per trade, the board's look and a few names (three a trade:
# a street of them reads as many shops, not one).
_GILT = ((60, 30, 34), (24, 50, 40), (26, 36, 70), (40, 40, 44), (100, 24, 34))
INDEPENDENTS = {
    "butcher": ("gilt", "serif", ["J. PEARCE & SON  BUTCHERS", "FOSTER'S  FAMILY BUTCHER", "H. GREEN  QUALITY MEATS"]),
    "bakery": ("gilt", "serif", ["HOLLOWAY'S BAKERY", "THE CORN DOLLY  BAKERS", "R. WEBB  BAKER"]),
    "greengrocer": ("gilt", "serif", ["F. MANNING  FRUITERER", "COUNTY FRUIT & VEG", "BARNES  GREENGROCER"]),
    "newsagent": ("plain", "sans", ["NEWSAGENT & TOBACCONIST", "THE PAPER SHOP", "KUMAR NEWS"]),
    "hardware": ("gilt", "serif", ["CHANDLER'S HARDWARE", "T. ELLIS  IRONMONGER", "HOME & GARDEN"]),
    "chemist": ("gilt", "serif", ["A. BRYANT  CHEMIST", "PHARMACY", "TOWN PHARMACY"]),
    "hairdresser": ("plain", "script", ["Salon Joanne", "Cut Above", "Gents Barber"]),
    "estate_agent": ("plain", "serif", ["HARDING & CO  ESTATE AGENTS", "MARSH & PARSONS", "COUNTY HOMES"]),
    "video": ("plain", "italic", ["STAR VIDEO", "VIDEO CITY", "MOVIE MAGIC"]),
    "chinese": ("plain", "sans", ["GOLDEN DRAGON  TAKEAWAY", "JADE GARDEN", "THE LUCKY STAR"]),
    "fish_chips": ("plain", "sans", ["THE CODFATHER  FISH & CHIPS", "THE FRYING PAN", "TOWN CHIPPY"]),
    "kebab": ("plain", "sans", ["ISTANBUL KEBAB HOUSE", "KEBAB & PIZZA", "THE KEBAB SHOP"]),
    "indian": ("plain", "serif", ["TAJ MAHAL TANDOORI", "THE BOMBAY", "SPICE OF INDIA"]),
    "italian": ("plain", "script", ["La Trattoria", "Luigi's", "Bella Napoli"]),
    "thai": ("plain", "serif", ["BANGKOK HOUSE", "THAI ORCHID", "SIAM"]),
    "restaurant": ("plain", "serif", ["THE BISTRO", "WINE BAR & BISTRO", "THE STEAKHOUSE"]),
    "cafe": ("plain", "serif", ["THE KETTLE CAFE", "TEA ROOMS", "BETTY'S CAFE"]),
    "laundry": ("plain", "sans", ["LAUNDERETTE", "DRY CLEANERS", "SPEEDY CLEAN"]),
    "florist": ("plain", "script", ["Petals", "The Flower Shop", "Blooms"]),
    "clothes": ("gilt", "serif", ["MADELEINE  LADIES' FASHIONS", "GENTLEMEN'S OUTFITTERS", "THE BOUTIQUE"]),
    "shoes": ("gilt", "serif", ["SHOE REPAIRS & KEYS", "BROWN'S SHOES", "FOOTWEAR"]),
    "charity": ("plain", "sans", ["HOSPICE SHOP", "OXFAN", "SUE RYDE SHOP"]),
    "gift": ("plain", "script", ["The Gift Box", "Cards & Gifts", "Treasures"]),
    "jewelry": ("gilt", "serif", ["WATCHMAKER & JEWELLER", "P. HUNT  JEWELLER", "GOLDSMITHS"]),
    "music": ("plain", "italic", ["RECORD CORNER", "SOUNDS", "MUSIC SHOP"]),
    "electrical": ("plain", "sans", ["TV & RADIO REPAIRS", "TOWN ELECTRICAL", "HI-FI CENTRE"]),
    "off_licence": ("gilt", "serif", ["WINES & SPIRITS", "THE WINE CELLAR", "OFF LICENCE"]),
    "travel_agency": ("plain", "sans", ["TOWN TRAVEL", "SUNSEEKER HOLIDAYS", "GLOBE TRAVEL"]),
    "bookmaker": ("plain", "sans", ["TURF ACCOUNTANT", "J. COLE  BOOKMAKER", "FAIRPLAY"]),
    "office": ("gilt", "serif", ["SOLICITORS", "ACCOUNTANTS", "INSURANCE BROKERS"]),
    "bank": ("plain", "serif", ["BUILDING SOCIETY", "COUNTY BUILDING SOCIETY", "WESSEX BUILDING SOCIETY"]),
    "convenience": ("plain", "sans", ["MINI MARKET", "LATE SHOP", "VILLAGE STORES"]),
    "general": ("gilt", "serif", ["GENERAL STORES", "THE EMPORIUM", "BRIC-A-BRAC"]),
    "variety": ("plain", "sans", ["DISCOUNT STORE", "POUND SAVER", "BARGAIN CENTRE"]),
    "baby": ("plain", "script", ["Little Ones", "Tiny Tots", "Baby World"]),
    "car_parts": ("plain", "sans", ["MOTOR SPARES", "AUTO PARTS", "TYRES & EXHAUSTS"]),
    "pizza": ("plain", "sans", ["PIZZA PALACE", "PIZZA EXPRESSO", "NAPOLI PIZZA"]),
    "car_repair": ("plain", "sans", ["MOT TESTING STATION", "EXHAUST CENTRE", "MOTOR REPAIRS"]),
    "health_food": ("plain", "serif", ["HEALTH FOODS", "NATURE'S WAY", "WHOLEFOODS"]),
}
_PLAIN = ((250, 248, 240), (24, 40, 80), (180, 20, 30), (20, 20, 22), (30, 100, 60))


def _ind_colours(k: int, style: str):
    if style == "gilt":
        return _GILT[k % len(_GILT)], CREAM
    ground = _PLAIN[k % len(_PLAIN)]
    ink = (24, 40, 80) if sum(ground) > 600 else W
    return ground, ink


def signs() -> dict:
    """Every sign, in sheet order: the chains, then each trade's independents."""
    out = dict(CHAINS)
    n = 0
    for trade, (style, font, texts) in INDEPENDENTS.items():
        for k, text in enumerate(texts):
            ground, ink = _ind_colours(n, style)
            out[f"ind_{trade}_{k}"] = (text, trade, ground, ink, font, None, style)
            n += 1
    out.update(ADDED)
    return out


# A sign's shortest form, for a narrow shop (fascias.py: 2-3 squares hold about 8
# letters at a size the game can show). A chain's own short name, else its trade's word.
CHAIN_SHORT = {"blockbluster": "BLUSTER", "dorothy_parkins": "D.PARKINS", "marks_spender": "M&SPENDER",
               "kfg": "K.F.G.", "dominoes": "DOMINOES", "pizza_hutt": "P.HUTT", "highland_bank": "HIGHLAND",
               "abbey_rational": "ABBEY", "clintons_cardz": "CLINTONS", "thomas_crook": "T.CROOK",
               "william_hall": "W.HALL", "holland_barratt": "H&B", "wessex_coop": "CO-OP",
               "littlewards": "LITTLEWARDS", "victory_wine": "WINES", "lunn_polly": "LUNN POLLY",
               "floyds": "FLOYDS", "h_samuels": "SAMUELS", "wh_smythe": "SMYTHE"}
TRADE_SHORT = {"butcher": "BUTCHER", "bakery": "BAKER", "greengrocer": "FRUIT", "newsagent": "NEWS",
               "hardware": "HARDWARE", "chemist": "CHEMIST", "hairdresser": "SALON", "estate_agent": "AGENTS",
               "video": "VIDEO", "chinese": "CHINESE", "fish_chips": "CHIPS", "kebab": "KEBABS", "indian": "INDIAN",
               "italian": "ITALIAN", "thai": "THAI", "restaurant": "BISTRO", "cafe": "CAFE", "laundry": "LAUNDRY",
               "florist": "FLOWERS", "clothes": "FASHION", "shoes": "SHOES", "charity": "CHARITY", "gift": "GIFTS",
               "jewelry": "JEWELLER", "music": "MUSIC", "electrical": "TV & HI-FI", "off_licence": "WINES",
               "travel_agency": "TRAVEL", "bookmaker": "BETTING", "office": "OFFICES", "bank": "BANK",
               "convenience": "STORES", "general": "STORES", "variety": "BARGAINS", "baby": "BABY",
               "car_parts": "SPARES", "pizza": "PIZZA", "car_repair": "MOT", "health_food": "HEALTH"}


def short_of(key: str) -> str:
    if key in CHAIN_SHORT:
        return CHAIN_SHORT[key]
    text, trade = SIGNS[key][0], SIGNS[key][1]
    if not key.startswith("ind_"):
        return text
    return TRADE_SHORT.get(trade, text.partition("  ")[0])


# Signs added after the first sheet was drawn go here, at the end, so no drawn
# sign's tile ids move (key: as CHAINS; an independent as ("text", trade, ...) too).
ADDED: dict = {}

# Pubs (PLAN step 5): each pub that has its hanging sign (asset_creator/src/pubs.py
# PUBS, in its order) gets a fascia with its name in its brewery's livery: a lookup by
# its key over the 1993 Hampshire mix (OSM names no brewery). The breweries' names a
# letter or two off, as the chains': Gale's of Horndean, Courage (Alton's own brewery
# until 1990s), Whitbread. Style "pub": a painted board, shaded serif letters, the
# brewery's name at its end where the board has room (fascias.board).
# (key, its name, ground, ink, its share of the pubs, its pubs' front: shopfronts.py's finish)
# - appended only, and the shares left be: a pub's brewery sizes its board (fascias.py), so
# a change moves the pub signs' tile ids (fascias_uk_28 on)
BREWERIES = [("gayles", "GAYLE'S", (28, 66, 46), GOLD, 35, "pub_green"),
             ("kourage", "KOURAGE", (26, 40, 92), CREAM, 35, "pub_navy"),
             ("whitebread", "WHITEBREAD", (96, 30, 36), GOLD, 30, "pub_maroon")]


def brewery_of(pub_key: str) -> tuple:
    h = int(hashlib.md5(("brewery" + pub_key).encode()).hexdigest()[:8], 16) % sum(b[4] for b in BREWERIES)
    for b in BREWERIES:
        if h < b[4]:
            return b
        h -= b[4]
    return BREWERIES[-1]


def _pub_signs() -> None:
    import os
    import sys
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
    from pubs import PUBS
    for key, (name, _emblem) in PUBS.items():
        _b, label, ground, ink, _w, _front = brewery_of(key)
        ADDED[f"pub_{key}"] = (name.upper(), "pub", ground, ink, "serif", label, "pub")


_pub_signs()
SIGNS = signs()

# A chain in OSM / FSA (brand or name, lower case, at the start) -> its 1993 sign.
OSM_CHAINS = [
    (r"tesco", "tisco"), (r"sainsbury", "lainburys"), (r"morrisons", "norrisons"), (r"asda", "greenes"),
    (r"lidl", "safeway"), (r"aldi", "kwiksave"), (r"waitrose", "waitrows"),
    (r"m ?& ?s\b|marks (and|&) spencer", "marks_spender"), (r"iceland", "greenland"),
    (r"(the )?co-?op(erative)?( food)?$", "wessex_coop"), (r"spar\b", "spur"),
    (r"boots", "boats"), (r"superdrug", "supadrug"), (r"w ?h ?smith", "wh_smythe"),
    (r"martin mccoll|martins newsagent", "martyns"), (r"greggs", "griggs"), (r"mcdonald", "spiffos"),
    (r"post office", "post_office"), (r"barclays", "barkleys"), (r"lloyds", "floyds"),
    (r"natwest", "nateast"), (r"hsbc|midland", "highland_bank"), (r"tsb\b", "tbs"),
    (r"santander|abbey national", "abbey_rational"), (r"halifax", "halifix"), (r"ladbrokes", "ladbrooks"),
    (r"william hill", "william_hall"), (r"thomas cook|tui\b", "thomas_crook"), (r"woolworth", "woolies"),
    (r"clinton", "clintons_cardz"), (r"halfords", "halfwords"), (r"kfc|kentucky", "kfg"),
    (r"pizza hut", "pizza_hutt"), (r"domino", "dominoes"), (r"texaco|valero", "texeco"),
    (r"clarks\b", "clerks"), (r"burton\b", "barton"), (r"tg jones", "wh_smythe"), (r"kwik.?fit", "kwik_fix"),
    (r"holland (&|and) barrett", "holland_barratt"), (r"peacocks", "peacox"),
]
# A name that is no shop (a hall, a club, a museum, a bar's other rooms): no fascia.
NOT_A_SHOP = re.compile(r"town hall|council|museum|gallery|church|chapel|\bhall\b|centre\b|\bclub\b|register office|"
                        r"jobcentre|clinic|surgery|\bvets?\b|school|\bbar\b|\barms\b|\binn\b|tavern|hotel|"
                        r"assembly rooms|masonic|library|police|health", re.I)
# Brands that came after 1993 -> the trade they stand in for.
LATE = {"starbucks": "cafe", "caffe nero": "cafe", "costa": "cafe", "pret": "cafe", "subway": "fast_food",
        "card factory": "gift", "savers": "chemist", "grape tree": "convenience", "esquires": "cafe"}

SHOP_TRADE = {
    "supermarket": "supermarket", "convenience": "convenience", "butcher": "butcher", "bakery": "bakery",
    "greengrocer": "greengrocer", "newsagent": "newsagent", "chemist": "chemist", "pharmacy": "chemist",
    "hardware": "hardware", "doityourself": "hardware", "hairdresser": "hairdresser", "beauty": "hairdresser",
    "barber": "hairdresser", "estate_agent": "estate_agent", "video": "video", "florist": "florist",
    "clothes": "clothes", "shoes": "shoes", "charity": "charity", "variety_store": "variety",
    "department_store": "variety", "electronics": "electrical", "books": "newsagent", "stationery": "newsagent",
    "jewelry": "jewelry", "travel_agency": "travel_agency", "bookmaker": "bookmaker", "gift": "gift",
    "music": "music", "alcohol": "off_licence", "wine": "off_licence", "laundry": "laundry",
    "dry_cleaning": "laundry", "frozen_food": "frozen_food", "car_parts": "car_parts", "baby_goods": "baby",
}
AMENITY_TRADE = {"fast_food": "fast_food", "cafe": "cafe", "restaurant": "restaurant", "bank": "bank",
                 "pharmacy": "chemist", "post_office": "post_office", "fuel": "fuel"}
CUISINE = {"chinese": "chinese", "indian": "indian", "italian": "italian", "pizza": "pizza", "thai": "thai",
           "kebab": "kebab", "fish_and_chips": "fish_chips", "burger": "fast_food", "chicken": "fast_food"}
BY_NAME = [(r"fish|chip|\bcod", "fish_chips"), (r"chinese|oriental|wok|dragon|china", "chinese"),
           (r"kebab", "kebab"), (r"pizza", "pizza"), (r"thai|siam|kwai", "thai"),
           (r"tandoori|indian|spice|gurkha|curry|balti", "indian"), (r"cafe|coffee|tea room", "cafe"),
           (r"\bnews", "newsagent"), (r"hair|barber|salon|face", "hairdresser"), (r"butcher|meats", "butcher"),
           (r"bakery|bakehouse|\bbaker\b|\bbake\b", "bakery"), (r"florist|flowers", "florist"),
           (r"jewel", "jewelry"), (r"optic", "chemist"), (r"travel", "travel_agency"),
           (r"estate agent|lettings|strutt|parker", "estate_agent"), (r"stitch|sew|fabric|wool", "gift")]
# No fascia: they have their own signs (pubs: pub_signs_uk_01) or none.
NO_FASCIA = {"pub", "bar", "nightclub", "hotel", "guest_house", "doctors", "dentist", "clinic",
             "place_of_worship", "police", "library", "townhall", "community_centre", "advice"}
# ...and shops with a sign of their own that no 1993 board here stands in for.
NO_FASCIA_SHOPS = {"funeral_directors"}

# A 1993 market town's high street, by trade (a lookup for shops the data gives no
# trade): the share of units each kind held.
TRADE_MIX = [("clothes", 10), ("estate_agent", 6), ("newsagent", 5), ("chemist", 5), ("bank", 6),
             ("hairdresser", 6), ("bakery", 4), ("butcher", 3), ("greengrocer", 3), ("hardware", 3),
             ("video", 3), ("electrical", 4), ("off_licence", 4), ("jewelry", 3), ("shoes", 3), ("gift", 4),
             ("charity", 4), ("travel_agency", 3), ("bookmaker", 2), ("florist", 2), ("music", 2),
             ("cafe", 4), ("laundry", 1), ("variety", 1), ("baby", 1), ("general", 3)]
OFFICE_MIX = [("office", 5), ("estate_agent", 3), ("bank", 2), ("travel_agency", 1)]

BY_TRADE: dict[str, list[str]] = {}
for _key, (_t, _trade, *_r) in SIGNS.items():
    BY_TRADE.setdefault(_trade, []).append(_key)


def _hash(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _pick(mix, h):
    total = sum(w for _t, w in mix)
    k = h % total
    for trade, w in mix:
        if k < w:
            return trade
        k -= w
    return mix[-1][0]


def chain_of(name: str | None) -> str | None:
    """The 1993 sign of a chain named in OSM / FSA, if it is one."""
    n = (name or "").strip().lower()
    for pat, key in OSM_CHAINS:
        if re.match(pat, n):
            return key
    return None


def trade_of(bid: str, uses: list[dict], name: str = "") -> str | None:
    """A shop's trade from its tags, its name, or the town's mix; None: no fascia."""
    names = [n for n in [name] + [t.get("name") or "" for t in uses] if n]
    if any(NOT_A_SHOP.search(n) for n in names):
        return None
    for t in uses:
        for k in ("amenity", "tourism", "healthcare", "leisure"):
            if (t.get(k) or "").lower() in NO_FASCIA:
                return None
    for n in names:
        low = n.lower()
        for brand, trade in LATE.items():
            if low.startswith(brand):
                return trade
    # The business's own name before its tags (a food hygiene record says only
    # "retail" or "other": P.A.R.K. Butchers was a convenience store).
    for n in names:
        for pat, by_name in BY_NAME:
            if re.search(pat, n, re.I):
                return by_name
    office = False
    for t in uses:
        shop = (t.get("shop") or "").lower()
        amen = (t.get("amenity") or "").lower()
        cuisine = (t.get("cuisine") or "").split(";")[0].lower()
        if cuisine in CUISINE and amen in ("fast_food", "restaurant"):
            return CUISINE[cuisine]
        if shop in SHOP_TRADE:
            return SHOP_TRADE[shop]
        if shop in NO_FASCIA_SHOPS:
            return None
        if shop and shop not in ("retail", "yes", "no", "vacant"):
            return "general"        # a trade of its own with no 1993 sign (funeral directors): not the town's mix
        if amen in AMENITY_TRADE:
            return AMENITY_TRADE[amen]
        office = office or bool(t.get("office") or t.get("craft"))
    return _pick(OFFICE_MIX if office else TRADE_MIX, _hash(bid, "trade"))


# A building's own OSM tags that say what it is (a building society mapped on its
# outline: amenity=bank); its knox_uses are the points and records inside it.
OWN_TAG_KEYS = ("amenity", "shop", "office", "craft", "tourism", "healthcare", "leisure", "cuisine", "brand", "name")


def shop_tags(props: dict) -> list[dict]:
    """A shop's uses for sign_for: its building's own tags first, then its knox_uses
    (shops.py). Read from the knox_uses alone, a building society whose only record
    was its business rates' "retail" drew a butcher's sign from the town's mix."""
    own = {k: v for k, v in props.items() if k in OWN_TAG_KEYS}
    return ([own] if own else []) + list(props.get("knox_uses") or [])


def sign_for(bid: str, uses: list[dict], name: str = "", taken: set | None = None) -> str | None:
    """The sign a shop gets, or None. `taken`: chains given so far in this town (updated)."""
    taken = taken if taken is not None else set()
    for t in [{"name": name}] + list(uses):
        for n in (t.get("brand"), t.get("name")):
            key = chain_of(n)
            if key:
                taken.add(key)
                return key
    trade = trade_of(bid, uses, name)
    if trade is None:
        return None
    pool = BY_TRADE.get(trade) or BY_TRADE["general"]
    chains = [k for k in pool if not k.startswith("ind_") and k not in taken and k != "spiffos"]
    inds = [k for k in pool if k.startswith("ind_")]
    h = _hash(bid)
    if chains and (not inds or h % 3 != 0):       # a town's chains first, while there are some
        key = chains[h % len(chains)]
        taken.add(key)
        return key
    if not inds:
        return None
    # Each of a trade's independents once before any comes again (review 2026-10-20:
    # three neighbouring units were all "County Fruit & Veg").
    free = [k for k in inds if k not in taken]
    if not free:
        taken.difference_update(inds)
        free = inds
    key = free[h % len(free)]
    taken.add(key)
    return key


def main(argv: list[str]) -> int:
    import collections
    import json
    import os
    m = argv[0].rstrip("/\\")
    name = os.path.basename(m)
    feats = json.load(open(os.path.join(m, f"{name}_buildings.geojson"), encoding="utf-8"))["features"]
    taken: set = set()
    got = collections.Counter()
    for ft in sorted(feats, key=lambda f: str(f["properties"].get("osm_id"))):
        p = ft["properties"]
        if p.get("knox_dwelling") not in ("shop", "shop_flats"):
            continue
        key = sign_for(str(p.get("osm_id")), shop_tags(p), p.get("name") or "", taken)
        got[key] += 1
        was = (p.get("name") or next((u.get("name") for u in p.get("knox_uses") or [] if u.get("name")), None) or "-")[:32]
        print(f"{str(p.get('osm_id')):>14}  {was:32}  -> {SIGNS[key][0] if key else '(none)'}")
    print(len(got), "signs:", dict(got.most_common()))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
