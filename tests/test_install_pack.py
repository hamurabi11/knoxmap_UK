"""The UK tile pack's installer (KnoxMap UK: Building_types/src/area/install_pack.py): a
release zip whose checksum matches is installed into the mods folder in place of an older
copy; a checksum that does not match, or a file outside the pack's folder, is refused and
nothing is installed. The release is stood in for by local files (no network)."""
import hashlib
import io
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, os.path.join(ukdirs.UK_SRC, "area"))
sys.path.insert(0, ukdirs.UK_SRC)

import install_pack as IP  # noqa: E402


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class Install(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.mods = os.path.join(t, "mods")
        self.release = os.path.join(t, "release")
        os.makedirs(self.release)
        self.was = IP.mods_dir, IP.fetch, IP.ukpaths.assets
        IP.mods_dir = lambda: self.mods
        IP.fetch = lambda url, path: shutil.copy(os.path.join(self.release, url.rsplit("/", 1)[1]), path)
        IP.ukpaths.assets = lambda *p: None                 # no rules step in a test

    def tearDown(self):
        IP.mods_dir, IP.fetch, IP.ukpaths.assets = self.was
        self.tmp.cleanup()

    def run_(self, members, checksum=None):
        z = os.path.join(self.release, "KnoxMapUK_Tiles.zip")
        h = make_zip(z, members)
        open(z + ".sha256", "w").write(f"{checksum or h}  KnoxMapUK_Tiles.zip\n")
        sys.argv = ["install_pack.py", "--from", "https://example.invalid/KnoxMapUK_Tiles.zip"]
        with redirect_stdout(io.StringIO()) as out:
            code = IP.main()
        return code, out.getvalue()

    def test_installed_in_place_of_an_older_copy(self):
        os.makedirs(os.path.join(self.mods, "KnoxMapUK_Tiles", "old"))
        code, _out = self.run_({"KnoxMapUK_Tiles/42/mod.info": "id=KnoxMapUK_Tiles\n",
                                "KnoxMapUK_Tiles/LICENSE.txt": "CC BY 4.0"})
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(self.mods, "KnoxMapUK_Tiles", "42", "mod.info")))
        self.assertFalse(os.path.exists(os.path.join(self.mods, "KnoxMapUK_Tiles", "old")))

    def test_a_wrong_checksum_installs_nothing(self):
        code, out = self.run_({"KnoxMapUK_Tiles/42/mod.info": "x"}, checksum="0" * 64)
        self.assertEqual(code, 1)
        self.assertIn("checksum mismatch", out)
        self.assertFalse(os.path.exists(self.mods))

    def test_a_file_outside_the_pack_is_refused(self):
        code, out = self.run_({"KnoxMapUK_Tiles/42/mod.info": "x", "KnoxMapUK_Tiles/../../evil.txt": "x"})
        self.assertEqual(code, 1)
        self.assertIn("refused", out)
        self.assertFalse(os.path.exists(self.mods))

    def test_only_https(self):
        sys.argv = ["install_pack.py", "--from", "http://example.invalid/KnoxMapUK_Tiles.zip"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(IP.main(), 2)


if __name__ == "__main__":
    unittest.main()
