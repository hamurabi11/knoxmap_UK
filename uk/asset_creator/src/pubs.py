"""The pubs whose signs pub_signs_uk_01 carries (townscape.py), by the name
OSM gives them in the maps' areas (Mapping/Building_types/data/<area>/
road_nodes.geojson, amenity=pub). A sign's sprites are its place in PUBS:
a new pub goes at the end (as roadgraph/town_centre.py's PUBS, the map's side), never
in between. Each with the emblem painted on its board (sign_faces.py)."""

PUBS = {
    # Odiham
    "anchor_inn": ("The Anchor Inn", "anchor"),
    "waterwitch": ("The Waterwitch", "boat"),
    "red_lion": ("The Red Lion", "lion"),
    "bell_inn": ("Bell Inn", "bell"),
    # Alton
    "swan": ("The Swan", "swan"),
    "greyfriar": ("The Greyfriar", "friar"),
    "ivy_house": ("The Ivy House", "ivy"),
    "white_horse": ("The White Horse", "white_horse"),
    "wheatsheaf": ("The Wheatsheaf", "sheaf"),
    "market_hotel": ("Market Hotel", "monogram"),
    "bakers_arms": ("The Bakers Arms", "loaf"),
    "george": ("The George", "st_george"),
    "eight_bells": ("The Eight Bells", "bells"),
    "crown_hotel": ("The Crown Hotel", "crown"),
    "railway_arms": ("The Railway Arms", "engine"),
    "french_horn": ("The French Horn", "horn"),
    "white_hart_inn": ("The White Hart Inn", "hart"),
}


def slug(name: str) -> str:
    """An OSM pub name -> its key in PUBS (as furniture.py's)."""
    s = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    s = s[4:] if s.startswith("the_") else s
    return "_".join(p for p in s.split("_") if p)
