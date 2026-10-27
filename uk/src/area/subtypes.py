"""Housing sub-types: about thirty, each one a tile set of its own (user,
2026-10-19; LOG 2026-10-19 #3, to-do K05.17 / K09.22).

A lookup on the evidence a building has, not a learner. In order of trust:
the listing (its date and features, read by asset_creator's
listed_building_maker lexicon); the EPC majority over the building's
certificates (age band, built form, property type, walls, tenure, floor
height); the development cluster and street (council estates by their
share of social tenancies); the footprint and the LiDAR (storeys, roof
shape); OSM's tag. assign_types.py calls subtype_of for each building and
writes `subtype` beside the older `type` (which Build's palette still reads).

Decisions (user, 2026-10-19): the EPC's 1900-1929 band is split by built
form and footprint (Edwardian or early inter-war); council housing is a
sub-type of its own in each era; listed buildings keep the listed maker
(Grade I / II* the landmark tool) - their sub-type here is for the record and
for unlisted look-alikes.
"""
from __future__ import annotations

# id: (type, what it looks like) - the tile set each one needs (K09.22).
SUBTYPES = {
    # vernacular, before about 1714
    "vern_timber": ("vernacular", "medieval / Tudor / 17th c. timber frame: jetties, close studding, brick nogging"),
    "vern_thatched": ("vernacular", "thatched cottage"),
    "vern_flint": ("vernacular", "flint with brick dressings"),
    "vern_brick": ("vernacular", "17th - early 18th c. brick: plain tile, casements"),
    # Georgian and Regency, 1714-1837
    "georgian_townhouse": ("georgian", "town house: symmetrical, sashes, parapet or hipped roof"),
    "georgian_house": ("georgian", "detached or rural Georgian house / Regency villa"),
    # Victorian, 1837-1901
    "vic_terrace": ("victorian", "terrace with rear outrigger (flat or bay fronted)"),
    "vic_villa": ("victorian", "semi or detached villa"),
    "vic_cottage": ("victorian", "estate / rural cottage"),
    # Edwardian, 1901-1914 (the 1900-1929 band, by built form and footprint)
    "edw_terrace": ("edwardian", "bay-and-gable terrace"),
    "edw_semi": ("edwardian", "semi, mock-Tudor gable, roughcast"),
    "edw_villa": ("edwardian", "detached Edwardian house"),
    # Inter-war, 1919-1939
    "iw_semi": ("interwar", "1930s private semi: hipped roof, two-storey bay, pebbledash"),
    "iw_detached": ("interwar", "inter-war detached house"),
    "iw_bungalow": ("interwar", "1930s bungalow"),
    "iw_council": ("interwar", "inter-war council cottage (Addison / Wheatley): plain brick or render, hipped"),
    # Post-war, 1945-1966
    "pw_system": ("postwar", "system-built (Airey, Cornish, Swedish)"),
    "pw_council": ("postwar", "brick council semi or terrace"),
    "pw_private": ("postwar", "1950s-60s private house"),
    "pw_bungalow": ("postwar", "1950s-60s bungalow"),
    # 1967-1982
    "e70_house": ("estate_6070", "open-plan estate house: shallow pitch, tile-hung or weatherboard panels"),
    "e70_terrace": ("estate_6070", "estate terrace / link-detached"),
    "e70_bungalow": ("estate_6070", "chalet bungalow"),
    "e70_council": ("estate_6070", "1960s-70s council housing"),
    # 1983-1995
    "e80_house": ("estate_8090", "neo-vernacular estate house: small gables, mock bays, brown tiles"),
    "e80_terrace": ("estate_8090", "starter terrace"),
    "e80_social": ("estate_8090", "housing association"),
    # 1996 on
    "nb_pastiche": ("new_build", "1996-2006 neo-Georgian pastiche"),
    "nb_contemporary": ("new_build", "2007 on: render panels, grey frames, cladding"),
    # flats
    "flats_conversion": ("flats", "flats in a converted old house"),
    "flats_period": ("flats", "purpose-built before 1950 (mansion block)"),
    "flats_postwar": ("flats", "1950-1982 block (council or private)"),
    "flats_modern": ("flats", "1983 on"),
    # not homes
    "non_domestic": ("other", "shop, pub, church, hall, school, works (shops.py, listed maker)"),
    "farm_building": ("other", "barn, stable, cowshed"),
    "outbuilding": ("other", "garage, shed"),
    "unknown": ("other", "no era"),
    # no era (no EPC: the osm_lidar data level, or no certificate near): by built form
    # alone, styled from every era's styles for that form (palette subtype_styles) -
    # the shape and roof give no era away (LOG 2026-10-27 #U1: 23% across towns)
    "unknown_terrace": ("unknown", "terrace, era unknown"),
    "unknown_semi": ("unknown", "semi, era unknown"),
    "unknown_detached": ("unknown", "detached house, era unknown"),
    "unknown_bungalow": ("unknown", "bungalow, era unknown"),
    "unknown_flats": ("unknown", "flats, era unknown"),
}

SOCIAL_SHARE = 0.5        # a building's certificates this share social tenancies: council / housing association
DEV_SOCIAL_SHARE = 0.3    # ...or its development's this share (Right to Buy sold many since)
TALL_FLOOR_M = 2.75       # median EPC floor height this tall or more: a Georgian or Victorian house
TERRACES = ("terrace_row", "mid_terrace", "end_terrace")
SEMIS = ("semi", "semi_pair")

# Project Zomboid is set in 1993 (the Knox Event, July 1993): a building put
# up since looks as one of the 1990s would (user, 2026-10-19). Its sub-type is
# kept for the record; its `look`, which the palette styles, is the 1983-1995
# sub-type of the same built form.
GAME_YEAR = 1993
AS_IN_1993 = {"nb_pastiche": "e80_house", "nb_contemporary": "e80_house"}


def look_of(subtype: str, b: dict) -> str:
    """The sub-type as it would stand in the game's 1993."""
    if subtype not in AS_IN_1993:
        return subtype
    epc_form = (b.get("epc_form") or "").lower()
    if (b.get("epc_social") or 0) >= SOCIAL_SHARE:
        return "e80_social"
    if b.get("form") in TERRACES or "terrace" in epc_form:
        return "e80_terrace"
    return AS_IN_1993[subtype]


def _year_type(year: int) -> str | None:
    if year < 1714:
        return "vernacular"
    if year <= 1837:
        return "georgian"
    if year <= 1901:
        return "victorian"
    if year <= 1918:
        return "edwardian"
    return None


def listed_subtype(b: dict, reading) -> tuple[str, str] | None:
    """A listed building's sub-type from its listing's date and features."""
    if reading is None or not reading.era:
        return None
    year = reading.era
    kind = _year_type(year)
    terrace = b["form"] in TERRACES or (b.get("setback_m") is not None and b["setback_m"] <= 4)
    if kind == "vernacular" or (kind == "georgian" and reading.has("wall.timber_frame", "any")):
        if reading.has("roof.thatch", "any"):
            return "vern_thatched", f"listing c.{year}, thatch"
        if reading.has("wall.timber_frame", "any"):
            return "vern_timber", f"listing c.{year}, timber frame"
        if reading.has("wall.flint", "any"):
            return "vern_flint", f"listing c.{year}, flint"
        return "vern_brick", f"listing c.{year}"
    if kind == "georgian":
        if reading.has("wall.flint", "any"):
            return "vern_flint", f"listing c.{year}, flint"
        return ("georgian_townhouse" if terrace else "georgian_house"), f"listing c.{year}"
    if kind == "victorian":
        return ("vic_terrace" if b["form"] in TERRACES else "vic_villa"), f"listing c.{year}"
    if kind == "edwardian":
        return ("edw_terrace" if b["form"] in TERRACES else "edw_villa"), f"listing c.{year}"
    return None


def unknown_era(b: dict, storeys: int | None) -> tuple[str, str]:
    """No era: the built form (outlines, OSM's tag, the EPC's form where there is one)
    and the LiDAR's storeys."""
    epc_form = (b.get("epc_form") or "").lower()
    if b["form"] == "multi_unit" or b.get("epc_type") in ("Flat", "Maisonette"):
        return "unknown_flats", "no era, flats"
    if storeys == 1 or b.get("epc_type") == "Bungalow" or b.get("btag") == "bungalow":
        return "unknown_bungalow", f"no era, {'one storey (LiDAR)' if storeys == 1 else 'bungalow'}"
    if b["form"] in TERRACES or "terrace" in epc_form:
        return "unknown_terrace", "no era, terrace"
    if b["form"] in SEMIS or "semi" in epc_form:
        return "unknown_semi", "no era, semi"
    return "unknown_detached", "no era, detached"


def subtype_of(b: dict, reading=None, roof: dict | None = None, dev_social: float = 0.0,
               dev_year: int | None = None) -> tuple[str, str]:
    """(sub-type, why) for one building. `b` is assign_types' record (form,
    btag, era, listed_grade, setback_m, area_m2, rectangularity, aspect, the
    epc_* summary); `reading` the listing's lexicon Reading, if listed;
    `roof` its roofs.csv row (storeys_lidar, roof_shape); `dev_social` and
    `dev_year` its development's social share and median EPC year."""
    if b["form"] == "outbuilding":
        return "outbuilding", "form"
    if b.get("btag") in ("farm_auxiliary", "barn", "farmhouse", "stable", "cowshed"):
        return "farm_building", "OSM tag"
    if b["form"] == "non_residential":
        return "non_domestic", "form"
    if b.get("listed_grade"):
        got = listed_subtype(b, reading)
        if got:
            return got
    era = b.get("era") or ""
    roof = roof or {}
    storeys = int(roof.get("storeys_lidar") or 0) or None
    if not era:
        return unknown_era(b, storeys)
    hipped = roof.get("roof_shape") == "hipped"
    epc_form = (b.get("epc_form") or "").lower()
    terrace = b["form"] in TERRACES or "terrace" in epc_form
    semi = b["form"] in SEMIS or "semi" in epc_form
    bungalow = b.get("epc_type") == "Bungalow" or storeys == 1
    flats = b.get("epc_type") in ("Flat", "Maisonette") or b["form"] == "multi_unit"
    social_n = b.get("epc_social") or 0.0
    social = social_n >= SOCIAL_SHARE or dev_social >= DEV_SOCIAL_SHARE
    why_social = f"social {social_n:.0%} / development {dev_social:.0%}"
    floor_h = b.get("epc_floor_h") or 0.0
    year = b.get("epc_year") or dev_year

    if flats:
        if era in ("pre1900", "1900_1929") and b["form"] != "multi_unit":
            return "flats_conversion", f"flats in a {era} house"
        if era in ("pre1900", "1900_1929", "1930_1949"):
            return "flats_period", era
        if era in ("1950_1966", "1967_1982"):
            return "flats_postwar", era
        return "flats_modern", era

    if era == "pre1900":
        tall = floor_h >= TALL_FLOOR_M or (storeys or 0) >= 3
        if tall and terrace and (b.get("setback_m") or 99) <= 4:
            return "georgian_townhouse", f"pre-1900, tall ({floor_h or storeys}), on the street line"
        if terrace:
            return "vic_terrace", "pre-1900 terrace"
        if (b.get("epc_floor_m2") or 0) >= 130 or b["area_m2"] >= 120 or tall:
            return "vic_villa", "pre-1900, large"
        return "vic_cottage", "pre-1900, small"

    if era == "1900_1929":
        # The band is Edwardian and early inter-war: built form and footprint
        # decide (user, 2026-10-19).
        if social:
            return "iw_council", f"1900-1929, {why_social}"
        if bungalow:
            return "iw_bungalow", "1900-1929 bungalow"
        if terrace:
            return "edw_terrace", "1900-1929 terrace"
        boxy = (b.get("aspect") or 9) < 1.4 and (b.get("rectangularity") or 0) >= 0.9
        if semi:
            if hipped or boxy or b.get("epc_walls") == "cavity":
                return "iw_semi", "1900-1929 semi, hipped / square / cavity: early inter-war"
            return "edw_semi", "1900-1929 semi, deep or solid-walled"
        if (b.get("rectangularity") or 1) < 0.85 or floor_h >= 2.6:
            return "edw_villa", "1900-1929 detached, L-shaped or tall"
        return "iw_detached", "1900-1929 detached, square: early inter-war"

    if era == "1930_1949":
        if social:
            return "iw_council", f"1930-1949, {why_social}"
        if bungalow:
            return "iw_bungalow", "1930-1949 bungalow"
        if semi or terrace:
            return "iw_semi", "1930-1949 semi / terrace"
        return "iw_detached", "1930-1949 detached"

    if era == "1950_1966":
        if b.get("epc_walls") == "system_built":
            return "pw_system", "EPC walls system built"
        if social:
            return "pw_council", f"1950-1966, {why_social}"
        if bungalow:
            return "pw_bungalow", "1950-1966 bungalow"
        return "pw_private", "1950-1966"

    if era == "1967_1982":
        if social:
            return "e70_council", f"1967-1982, {why_social}"
        if bungalow:
            return "e70_bungalow", "1967-1982 bungalow"
        if terrace:
            return "e70_terrace", "1967-1982 terrace"
        return "e70_house", "1967-1982"

    if era == "1983_1995":
        if social:
            return "e80_social", f"1983-1995, {why_social}"
        if terrace:
            return "e80_terrace", "1983-1995 terrace"
        return "e80_house", "1983-1995"

    # 1996 on: the EPC's finer bands (or the development's) split it.
    if year and year >= 2007:
        return "nb_contemporary", f"built {year} on"
    return "nb_pastiche", f"built 1996-2006{'' if year else ' (no finer band: the earlier)'}"
