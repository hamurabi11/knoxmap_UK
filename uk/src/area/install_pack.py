"""The UK tile pack (the KnoxMapUK_Tiles mod) downloaded and installed into the game.

    python install_pack.py [--from <url of the zip>]

The pack is a GitHub release of the KnoxMap UK fork (KNOXMAP_UK_RELEASES, else the
fork's latest release): KnoxMapUK_Tiles.zip, and KnoxMapUK_Tiles.zip.sha256 beside it.
In order, each a step the app's UK panel shows:

== download   the zip and its checksum
== verify     the zip's SHA-256 against the checksum (a mismatch stops here); every file
              in it inside KnoxMapUK_Tiles/ (no path out of the mods folder)
== install    into the game's mods folder (~/Zomboid/mods, or the one KnoxMap is set
              to), in place of any older copy
== rules      the UK ground, lamps, drains and foliage into the map tools' rules
              (asset_creator's register.py --rules-only), so maps compiled from now on
              use them; skipped without asset_creator or the map tools

The pack is CC BY 4.0 (its LICENSE.txt).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths
import ukpaths  # noqa: E402

PACK = "KnoxMapUK_Tiles"
RELEASES = os.environ.get("KNOXMAP_UK_RELEASES",
                          "https://github.com/hamurabi11/knoxmap_UK/releases/latest/download/")
UA = {"User-Agent": "KnoxMap-UK install_pack"}


def mods_dir() -> str:
    sys.path.insert(0, ukpaths.KNOX)
    import knoxpaths
    return os.path.join(str(knoxpaths.zomboid_user_dir()), "mods")


def fetch(url: str, path: str) -> None:
    # requests (KnoxMap's own): its certificate bundle, where Windows' store lacks the one
    # GitHub's download server needs (urllib failed on it, 2026-10-27)
    import requests
    with requests.get(url, headers=UA, stream=True, timeout=(15, 600)) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)


def main() -> int:
    a = sys.argv
    url = a[a.index("--from") + 1] if "--from" in a else RELEASES + PACK + ".zip"
    if not url.startswith("https://"):
        print(f"not an https download: {url}")
        return 2
    with tempfile.TemporaryDirectory() as t:
        print("== download", flush=True)
        z = os.path.join(t, PACK + ".zip")
        fetch(url, z)
        fetch(url + ".sha256", z + ".sha256")
        print(f"  {os.path.getsize(z) / 1e6:.0f} MB", flush=True)

        print("== verify", flush=True)
        want = open(z + ".sha256", encoding="utf-8").read().split()[0].lower()
        h = hashlib.sha256()
        with open(z, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != want:
            print(f"  checksum mismatch: the download is not the release's ({h.hexdigest()[:12]} != {want[:12]})")
            return 1
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            bad = [n for n in names if not n.startswith(PACK + "/") or ".." in n.split("/") or n.startswith("/")]
            if bad:
                print(f"  refused: {len(bad)} files outside {PACK}/ (first {bad[0]})")
                return 1
            print(f"  {len(names)} files, checksum good", flush=True)

            print("== install", flush=True)
            mods = mods_dir()
            os.makedirs(mods, exist_ok=True)
            staging = os.path.join(t, "unpacked")
            zf.extractall(staging)
        dest = os.path.join(mods, PACK)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(os.path.join(staging, PACK), dest)
        print(f"  {dest}", flush=True)

    print("== rules", flush=True)
    register = ukpaths.assets("Tile_creator", "src", "register.py")
    if not register or not os.path.exists(register):
        print("  no asset_creator here: maps with UK assets use the pack's tiles, the ground stays vanilla")
        return 0
    r = subprocess.run([sys.executable, register, "--rules-only"], capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print("  " + (r.stdout.strip().splitlines() or ["(no output)"])[-1], flush=True)
    if r.returncode:
        print("  the rules were not changed (no map tools yet? run Setup.bat, then install again)")
    print(f"{PACK} installed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
