"""Reading a list entry: the terms of the listing text as a lookup table.

List descriptions are written to a formula ("C18. 2 storeys and attic, 3
windows. Red tile roof, hipped at east end. Red brick walling in Flemish
bond..."), so they are read with a table of the trade's terms, not a model:
each feature is a few patterns, matched sentence by sentence.

`read(text)` gives a `Reading`:
  kind      what the entry is (house, shop, inn, church, barn... or a structure
            that is not a building: wall, memorial, tomb, bridge, kiosk...)
  era       the earliest century or year named (C15 -> 1450, "early C19" -> 1810)
  storeys   the main range's storeys (and attic / basement)
  bays      windows across the front ("3 windows", "2.1.3.2 windows" -> 8)
  hits      every feature found: {feature: [Hit(scope, modern, quote)]}

Each sentence has a scope: `front` (the default), `rear` (the rear, back,
outshots, extensions, a barn behind), `interior` (inside, never built from),
and a feature in a sentence that says it is modern ("modern shopfront", "C20
dormer") is marked `modern`, so a modern detail does not choose the old look.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUM = {"one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
       "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
N = r"(\d+|" + "|".join(NUM) + r")"

# What an entry is, first match wins (the entry's name is read before its text).
# Structures (not buildings) are made as objects or boundaries, not built.
KINDS = [
    ("kiosk", "structure", r"\bk6\b|telephone (kiosk|box)"),
    ("memorial", "structure", r"\bwar memorial\b|\bmemorial\b(?! hall)|\bobelisk\b"),
    ("tomb", "structure", r"\b(chest |table |pedestal )?tombs?\b|headstones?|\bmonument to\b"),
    ("milestone", "structure", r"\bmile ?(stone|post)\b|boundary (stone|post)"),
    ("pump", "structure", r"\bpump\b|\bwell ?head\b"),
    ("bridge", "structure", r"\bbridge\b|\bculvert\b"),
    ("lock", "structure", r"\block\b(?! up)|\bcanal\b"),
    ("stocks", "structure", r"\bstocks\b|\bwhipping post\b|\bpound\b"),
    ("wall", "structure", r"^(the )?(boundary |garden |front |retaining )?walls?\b|\bwalls? (to|of|at|round|enclosing)\b|"
                          r"\bgate ?(piers|way)\b|\bgates\b|\brailings\b"),
    ("church", "building", r"\bchurch\b(?! (centre|hall|rooms?))|\bchapels?\b|\bmeeting house\b"),
    ("castle", "building", r"\bcastle\b"),
    # A farm building before the inn ("Stable Block at Rear of George Hotel"),
    # but a home named after one is a home ("Forge House", "Mill Cottage").
    ("barn", "building", r"\bbarns?\b|\bgranary\b|\bstables?\b|\bstable block\b|\bcart ?shed\b|\bmaltings?\b|"
                         r"\boasts?\b|\bdovecote\b|\bcoach ?house\b|\boutbuildings?\b|\bwarehouse\b|\bcow ?house\b|"
                         r"\bbyre\b|"
                         r"\b(forge|mill)\b(?!\s+(house|cottage|farm))(?!.*\b(house|cottage)\b)"),
    ("inn", "building", r"\binn\b|\bhotel\b|public house|\btavern\b|\barms\b"),
    ("school", "building", r"\bschool\b|\bhall\b(?! house)|\bworkhouse\b"),
    ("shop", "building", r"\bshop\b|\bbank\b|\bpremises\b"),
    ("farmhouse", "building", r"\bfarm ?house\b|\bfarm\b"),
    # almshouses: a row of small homes (Odiham's of 1628, round a court)
    ("cottage", "building", r"\bcottages?\b|\blodge\b|\balmshouses?\b"),
    ("house", "building", r".*"),
]
# Kinds the opening sentences may give a plain address ("Barn of 4 bays.");
# not halls, shops or inns, which a text names as history ("former open hall").
TEXT_KINDS = {"church", "castle", "barn", "farmhouse", "cottage"}
HOME_KINDS = {"inn", "farmhouse", "cottage", "house"}

# Feature -> patterns. Grouped by what they decide (the group is the prefix).
FEATURES = {
    # walls: the main material of the outside walls
    "wall.red_brick": r"^\s*brick\s*$|\bred brick|\bbrick(work|walling| wall(s|ing)?| fronts?| facade| with)\b|\bin brick\b|\bof brick\b",
    "wall.blue_brick": r"\bblue (brick|headers)|vitrified headers|grey headers|chequer(ed)? brick",
    "wall.painted_brick": r"\bpainted brick|\bpainted walls|colour ?washed brick|whitewashed brick|\bpainted\b(?=.*walling)",
    "wall.render": r"\brender(ed)?\b|\bstucco(ed)?\b|\broughcast\b|\bpebbledash|\bplaster(ed)? (walls|walling|front|infill)|"
                   r"\bcolour ?washed\b|\bwhite ?washed\b|\bcement\b(?= render)",
    # "sashes in exposed frames" are window frames: not "exposed frames"
    "wall.timber_frame": r"timber[- ]fram(e|ed|ing)|\bexposed (timber|framing|frame\b(?!s))|\bbox[- ]fram|\bclose[- ]studd|"
                         r"\bcruck|\bbrick ?nogg|\bstuds? and|\bwattle",
    "wall.flint": r"\bflint",
    "wall.stone": r"\b(lime|sand|rag|bath|ashlar|malm|clunch|chalk|ironstone|portland)\s*stone|\bashlar\b|"
                  r"\bstone (walls|walling|rubble|front|built)\b|\brubble\b|\bchalk\b(?! pit)|\bclunch\b",
    "wall.tile_hung": r"tile[- ]hung|tile[- ]hanging|hung tiles|\bmathematical tiles",
    "wall.weatherboard": r"weather[- ]?board|\bboarded (walls|gables|upper)|\bfeather[- ]?edged board|\bclad\b",
    "bond.flemish": r"flemish bond",
    "bond.english": r"english bond|english garden wall",
    "bond.header": r"header bond",
    # roof covering and form
    "roof.slate": r"\bslated?\b(?! hanging)",
    "roof.plain_tile": r"\b(red |plain |clay |old |brown |handmade )?tiled? roof|\btiled\b|\btile roof|\bred tiles?\b|"
                       r"\bplain tiles?\b|\bpeg ?tiles?\b|\btiles\b(?=.*roof)",
    "roof.thatch": r"\bthatch(ed)?\b",
    "roof.stone_slate": r"stone slates?|horsham slab",
    "roof.lead": r"\blead(ed)? (roof|flat|covered)|\blead[- ]covered",
    "roof.hipped": r"\bhipped\b(?!.*dormer)|\bhips?\b",
    "roof.half_hipped": r"half[- ]hip|(1/2|½)[- ]?hip|gambrel|jerkin",
    "roof.mansard": r"\bmansard",
    "roof.catslide": r"\bcatslide",
    "roof.gabled": r"\bgabled?\b|\bgable ends?\b",
    "roof.parapet": r"\bparapet",
    "roof.dormer": r"\bdormers?\b",
    "roof.chimney": r"\bchimneys?\b|\bstacks?\b|\bflues?\b",
    # windows
    "window.sash": r"\bsash(es)?\b|\bsash windows\b",
    "window.casement": r"\bcasements?\b|\beasements\b",
    "window.leaded": r"\bleaded|\bsquare[- ]leaded|\bdiamond[- ]leaded|\blattice",
    "window.mullion": r"\bmullion(ed)?\b|\btransom(ed)?\b",
    "window.bay": r"\bbays?\b(?= window| with)|\b(canted|splayed|rectangular|square|curved|bow|segmental)[- ](fronted )?bays?\b|"
                  r"\bbow[- ]windows?\b|\bbow[- ]shaped\b|\bprojecting bay\b|\bsolid bay\b",
    "window.oriel": r"\boriel",
    "window.venetian": r"\bvenetian\b",
    "window.gothic": r"\bgothick?\b|\bpointed (arch|window|head)|\btracery\b|\blancet",
    # ("an arched opening" is a doorway's: a doorcase's, a carriage entry's)
    "window.round_headed": r"\bround[- ]headed\b(?! door)|\barched (window|head)|\bsemi[- ]circular(?! fanlight)",
    "window.french": r"\bfrench (windows?|doors?)\b",
    # doors and entrances
    "door.six_panel": r"\b6[- ]panel|\bsix[- ]panel",
    "door.four_panel": r"\b4[- ]panel|\bfour[- ]panel",
    "door.panelled": r"\bpanelled doors?\b|\bflush[- ]panelled\b|\bfielded panel",
    "door.boarded": r"\b(ledged|boarded|plank|battened|braced) doors?\b|\bboarded and ledged\b",
    "door.fanlight": r"\bfanlights?\b|\bradial (glazing|fanlight)|\bsemi[- ]circular fanlight",
    "door.doorcase": r"\bdoorcases?\b|\bdoor ?case\b|\bdoorway\b.*\b(pilasters|pediment|entablature)",
    "door.pediment": r"\bpediment(ed)?\b",
    "door.porch": r"\bporch(es)?\b|\bportico\b",
    "door.hood": r"\bhood\b|\bcanopy\b|\bshell hood\b",
    # walls' trim
    "trim.cornice": r"\bcornice|\bmodillion|\bdentil|\beaves (band|course|cornice)",
    "trim.band": r"\bbands?\b|\bstring ?course|\bfirst[- ]floor band|\bplat band",
    "trim.plinth": r"\bplinth",
    "trim.quoins": r"\bquoins?\b|\bpilaster strips|\brusticated (corners|angles)",
    "trim.pilasters": r"\bpilasters?\b|\bcolumns?\b|\bdoric|\bionic|\btuscan",
    "trim.flat_arches": r"\b(rubbed|gauged|flat|cambered|segmental|brick) arch(es|ed)?\b|\bkeystones?\b",
    "trim.cills": r"\bcills?\b|\bsills?\b",
    "trim.bargeboards": r"\bbarge ?boards?\b|\bbargeboarded",
    "trim.jetty": r"\bjett(y|ied|ies)\b|\boversail|\boverhang(ing)? (upper|first)",
    "trim.balcony": r"\bbalcon(y|ies)\b|\bverandah|\bveranda",
    "trim.railings": r"\brailings?\b|\bguard ?rail|\bwrought[- ]iron\b|\bcast[- ]iron\b",
    "trim.sign_bracket": r"\bsign ?brackets?\b|\bprojecting (wrought-iron )?brackets?\b|\binn sign",
    # the ground floor
    "ground.shopfront": r"\bshop ?[- ]?fronts?\b|\bshop windows?\b|\bfascia\b",
    "ground.carriage_entry": r"\bcarriage (entry|entrance|way|arch)|\bcarriageway\b|\barchway\b|"
                             r"\bthrough (passage|archway)|\bcoach (entry|entrance|arch)",
    # churches and big buildings
    "church.tower": r"\btower\b",
    "church.spire": r"\bspire\b|\bbroach\b|\bshingled\b",
    "church.buttress": r"\bbuttress",
    "church.battlements": r"\bbattlement|\bcrenellat|\bembattled",
    "church.bellcote": r"\bbell ?cote|\bbelfry|\bcupola|\blantern\b|\bweather ?vane|\bclock\b",
}
FEATURES_RX = {k: re.compile(v, re.I) for k, v in FEATURES.items()}

# A sentence about a modern part: its features do not choose the historic look.
MODERN = re.compile(r"\bmodern\b|\bc20\b|\bc21\b|\b(19[2-9]\d|20\d\d)\b|\brecent(ly)?\b|\breplacement\b|"
                    r"\bnew\b|\bupvc\b|\balter(ed|ation)", re.I)
# Scope from how a sentence starts: listing texts open a sentence with the part
# it describes ("At the rear...", "The rear wing...", "Within is...").
REAR = re.compile(r"^(at |to |on |from )?(the )?(rear|back)\b|^(a |an |the |two |2 )?(later |older |long |c1\d |c20 |modern |"
                  r"small |low |lower |single[- ]storey(ed)? |one[- ]storey(ed)? |lean[- ]to |brick |timber[- ]framed )*"
                  r"(rear |back |side )?(wings?|ranges?|additions?|extensions?|outshots?|lean[- ]tos?)\b|"
                  r"\bat the back\b", re.I)
INTERIOR = re.compile(r"^(within|inside|interior|internally)\b|^the interior\b|"
                      r"\binterior (has|have|retains|contains|features|includes|is|not)\b|\bfireplaces?\b|"
                      r"\bstair(case)?s?\b|\bpanelled (walls|rooms?)\b|\bceilings?\b|\binglenook", re.I)
# A feature whose clause also names one of these belongs to that part, not to
# the main walls or roof ("dormer with hipped roof", "brick chimney").
NOT_WITH = {"roof.hipped": r"dormer|porch|bay|canopy|hood|outshot",
            "roof.gabled": r"dormer|porch|bay|projecting|wing",
            "roof.slate": r"dormer|porch|bay|outshot|lean",
            "roof.plain_tile": r"dormer|porch|tile[- ]hung|hood",
            "wall.red_brick": r"chimney|stack|plinth|nogg|infill|dentil|band|arch|cills?|dressings|quoins|piers|"
                              r"flues|shafts?|garden wall|boundary",
            "wall.timber_frame": r"interior|internal|inside",
            "window.round_headed": r"door|fanlight|entrance|entry|carriage|porch|gateway"}
NOT_WITH_RX = {k: re.compile(v, re.I) for k, v in NOT_WITH.items()}
# Headings in the newer entries; a heading sets the scope of what follows it.
HEADINGS = {"interior": "interior", "exterior": "front", "plan": "skip", "materials": "front",
            "history": "skip", "reasons for designation": "stop", "reasons for listing": "stop", "sources": "stop", "legacy": "stop",
            "summary of building": "skip", "summary": "skip", "listing ngr": "stop"}

ERA = re.compile(r"\b(early|mid|late|later|c\.?|circa|first half of the|second half of the|"
                 r"first quarter of the|last quarter of the)?\s*-?\s*c(1[0-9]|20)\b|"
                 r"\b(dated|built|circa|c\.?|in)\s+(1[4-9]\d\d)\b|\b(1[4-9]\d\d)\b|\b(medieval|tudor|"
                 r"elizabethan|jacobean|georgian|regency|victorian|edwardian)\b", re.I)
PERIOD_YEAR = {"medieval": 1400, "tudor": 1530, "elizabethan": 1580, "jacobean": 1615, "georgian": 1760,
               "regency": 1820, "victorian": 1870, "edwardian": 1905}
EDGE = {"early": 10, "first quarter of the": 12, "first half of the": 25, "mid": 50, "c.": 50, "c": 50,
        "circa": 50, "second half of the": 75, "late": 85, "later": 85, "last quarter of the": 88, None: 50}

STOREYS = re.compile(N + r"(?:\s*and\s*(?:a\s*)?half)?\s*[- ]?storey(?:s|ed)?\b(.{0,40})", re.I)
WINDOWS = re.compile(r"\b((?:\d+\.)+\d+|" + N + r")\s+(?:first[- ]floor\s+)?windows?\b", re.I)
HEADER_END = re.compile(r"(?:^|\s)(?:GV\s*)?(?:I|II|II\*|III)\s*$", re.M)


@dataclass
class Hit:
    feature: str
    scope: str
    modern: bool
    quote: str


@dataclass
class Reading:
    kind: str = "house"
    group: str = "building"
    era: int | None = None
    storeys: int | None = None
    attic: bool = False
    basement: bool = False
    bays: int | None = None
    hits: dict[str, list[Hit]] = field(default_factory=dict)
    body: str = ""

    def has(self, feature: str, scope: str = "front", modern: bool = False) -> bool:
        """The feature is found in this scope ('any' for every scope), old
        unless `modern`."""
        return any((scope == "any" or h.scope == scope) and (modern or not h.modern)
                   for h in self.hits.get(feature, ()))

    def features(self, scope: str = "front") -> list[str]:
        return sorted(f for f in self.hits if self.has(f, scope))


def body_of(text: str) -> str:
    """The description without the old card's header (grid squares, street,
    'formerly listed as', dates and grade) or the new format's notes after it."""
    t = text.replace("\r", "")
    m = None
    for m in HEADER_END.finditer(t[: max(1500, len(t) // 2)]):
        pass
    if m:
        t = t[m.end():]
    t = re.sub(r"Listing NGR:.*", "", t)
    # Old cards are typed with hard line breaks mid-sentence: a line that does
    # not end a sentence (or is not a heading) runs on into the next.
    t = re.sub(r"(?<![.:;\n])\n(?!\n|[A-Z][A-Z ]{3,30}:?\n)", " ", t)
    t = re.sub(r"This list entry was subject to a Minor Amendment.*?\n", "", t)
    return t.strip()


def sentences(body: str):
    """(scope, sentence) in order, scope from the headings and the sentence."""
    scope = "front"
    for para in re.split(r"\n+", body):
        head = para.strip().rstrip(":").lower()
        if head in HEADINGS:
            scope = HEADINGS[head]
            if scope == "stop":
                return
            continue
        if scope == "skip":
            continue
        m = re.match(r"^\s*([A-Z][A-Z ]{3,30}):\s*(.*)$", para)
        if m and m.group(1).strip().lower() in HEADINGS:
            scope = HEADINGS[m.group(1).strip().lower()]
            if scope == "stop":
                return
            para = m.group(2)
        for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", para):
            s = s.strip()
            if not s or scope == "skip":
                continue
            here = scope
            if here == "front" and INTERIOR.search(s):
                here = "interior"
            elif here == "front" and REAR.search(s):
                here = "rear"
            yield here, s


def era_of(body: str) -> int | None:
    """The building's date: the opening sentence's where it gives one ("Quaker
    meeting house, 1672."), else the earliest in the description. The history
    and the reasons for listing do not count: their dates are other buildings'
    and events' (George Fox "from 1647")."""
    said = [s for scope, s in sentences(body)]
    for text in (said[:1], said):
        years = _years(" ".join(text))
        if years:
            return min(years)
    return None


def _years(body: str) -> list[int]:
    years = []
    for m in ERA.finditer(body):
        if m.group(2):
            edge = (m.group(1) or "").lower().strip() or None
            years.append((int(m.group(2)) - 1) * 100 + EDGE.get(edge, 50))
        elif m.group(4) or m.group(5):
            y = int(m.group(4) or m.group(5))
            if 1400 <= y <= 1945:
                years.append(y)
        elif m.group(6):
            years.append(PERIOD_YEAR[m.group(6).lower()])
    return years


def kind_of(name: str, first: str) -> tuple[str, str]:
    """(kind, group) from the entry's name, else its opening sentences.

    Only the name's head counts: "Barn at Hatchwood to South West of House" is
    a barn, "Wall Between the Bell Inn and Bury Cottage" a wall. A home word in
    the head makes a home ("Castle Bridge Cottages", "Barn Cottage", "Old
    Church House"), except the farm and trade buildings named "... house"."""
    head = re.split(r"\b(at|to|of|between|including|adjacent|adjoining|behind|in front)\b", name)[0].strip() or name
    # a street named after a building is an address, not the building ("1, Anstey Mill Lane")
    street = r"\b(church|chapel|mill|barn|forge|maltings?) (street|lane|road|path|hill|close|row|yard|way)\b"
    head = re.sub(street, "", head)
    first = re.sub(street, "", first)
    if re.search(r"\b((?<!oast )(?<!coach )(?<!cow )(?<!charnel )(?<!meeting )house|cottages?|farmhouse|lodge|villas?)\b",
                 head) and not re.match(r"^\s*(the )?(boundary |garden )?(walls?|gate|railings|tomb|pump)\b", head):
        return next((k, "building") for k, g, rx in KINDS if k in HOME_KINDS and re.search(rx, head))
    named = next(((k, g) for k, g, rx in KINDS[:-1] if re.search(rx, head)), None)
    # A house that keeps a farm building's name ("King's Barn": "Elegant facade of
    # 2 storeys, ... 3 windows"): its text describes a front of storeys and
    # windows and never calls it a barn (its own name, repeated, aside).
    # Its own name is set aside only where it is more than the kind word ("King's
    # Barn", not "Barn" or "The Barn", whose text's own "Barn." must still count).
    if named and named[0] == "barn" and re.sub(r"^(the )?", "", head.strip()) not in ("barn", "barns", "stable block",
                                                                                      "stables", "granary", "oast"):
        text = first.replace(head.strip(), " ", 1)
        if not re.search(dict((k, rx) for k, _g, rx in KINDS)["barn"], text) \
                and re.search(r"\b(facade|front)\b.*\bstoreys?\b|\bstoreys?\b.*\bwindows?\b", text):
            return "house", "building"
    if named is None:
        named = next(((k, g) for k, g, rx in KINDS[:-1] if k in TEXT_KINDS and re.search(rx, first)),
                     ("house", "building"))
    return named


def read(text: str, name: str = "") -> Reading:
    body = body_of(text)
    r = Reading(body=body)
    low_name = name.lower()
    first = " ".join(body.split(".")[:3]).lower()
    r.kind, r.group = kind_of(low_name, first)
    r.era = era_of(body)
    for scope, s in sentences(body):
        # "Modern" and the NOT_WITH parts are read clause by clause: "Sashes
        # in reveals: ..., modern ground floor window" keeps its old sashes.
        clauses = re.split(r"[,:;(]|\bwith\b|\b(?:approached|reached)\b|\band(?= flanked\b)", s)
        for feat, rx in FEATURES_RX.items():
            for c in clauses:
                if rx.search(c) and not (feat in NOT_WITH_RX and NOT_WITH_RX[feat].search(c)):
                    r.hits.setdefault(feat, []).append(Hit(feat, scope, bool(MODERN.search(c)), s[:240]))
                    break
        if scope != "front":
            continue
        if r.storeys is None:
            m = STOREYS.search(s)
            if m:
                n = m.group(1).lower()
                r.storeys = int(n) if n.isdigit() else NUM.get(n)
                tail = m.group(2).lower()
                r.attic = bool(re.search(r"\battics?\b|\bgarret", tail))
                r.basement = bool(re.search(r"\bbasement|\bcellar", tail))
        if r.bays is None:
            m = WINDOWS.search(s)
            if m:
                n = m.group(1).lower()
                r.bays = (sum(int(x) for x in n.split(".")) if "." in n else
                          int(n) if n.isdigit() else NUM.get(n))
    if not r.attic:
        r.attic = r.has("roof.dormer") and bool(re.search(r"\battic", body, re.I))
    return r
