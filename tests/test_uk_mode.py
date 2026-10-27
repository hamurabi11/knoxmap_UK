"""KnoxMap UK's mode in the app (web/uk.py, PLAN U stage 4): the UK mode button is always
on the page, the panel hidden unless the app started in UK mode (the button shows it); the
data check lists every extra dataset with a download page; settings are checked into uk.json; a run of the UK chain is a job the page
can poll to its end (its last step on a built test map, when it is there)."""
import json
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from flask import Flask, render_template  # noqa: E402

import app as knoxapp  # noqa: E402
from web import uk  # noqa: E402

TEST_MAP = "Alton_osm_graph"          # LOG 2026-10-27 #U1-#U3


def client():
    a = Flask(__name__)
    a.register_blueprint(uk.bp)
    return a.test_client()


class Page(unittest.TestCase):
    def test_the_button_and_the_panel(self):
        with knoxapp.app.test_request_context("/"):
            on = render_template("index.html", version="t", in_window=False, uk=True)
            off = render_template("index.html", version="t", in_window=False, uk=False)
        for page in (on, off):
            self.assertIn('id="ukModeBtn"', page)
            self.assertIn("/static/js/uk.js", page)
        self.assertIn('id="ukCard">', on)                 # shown
        self.assertIn('id="ukCard" hidden>', off)         # there, for the button to show
        self.assertIn('data-start-on="1"', on)
        self.assertIn('data-start-on="0"', off)

    def test_the_routes_whatever_the_flag(self):
        self.assertIn("uk.api_data", knoxapp.app.view_functions)

    def test_the_flag(self):
        was = os.environ.get("KNOXMAP_UK")
        try:
            os.environ["KNOXMAP_UK"] = "1"
            self.assertTrue(uk.enabled())
            os.environ["KNOXMAP_UK"] = ""
            self.assertFalse(uk.enabled())
        finally:
            os.environ.pop("KNOXMAP_UK", None) if was is None else os.environ.__setitem__("KNOXMAP_UK", was)


class Data(unittest.TestCase):
    def test_every_dataset_has_a_download_page(self):
        d = client().get("/api/uk/data").get_json()
        self.assertEqual({x["key"] for x in d["datasets"]},
                         {"tile_pack", "os_roads", "uprn", "epc_domestic", "epc_nondomestic", "inspire"})
        for x in d["datasets"]:
            self.assertTrue(x["url"].startswith("https://"), x)
            self.assertIsInstance(x["have"], bool)
        self.assertTrue(d["automatic"])

    def test_only_the_listed_pages_open(self):
        self.assertEqual(client().post("/api/uk/open", json={"key": "http://evil"}).status_code, 400)


class Settings(unittest.TestCase):
    def test_checked(self):
        s = uk._clean_settings({"roads": "cost-front", "lanes": 1, "front_anchor": "auto", "streets": "High Street, ",
                                "world_cell": ["80", " 0"], "backend": "nonsense", "area": "../x", "dormers": 0})
        self.assertEqual(s, {"roads": "cost-front", "lanes": True, "front_anchor": None, "streets": ["High Street"],
                             "world_cell": [80, 0], "backend": None, "dormers": False})
        self.assertNotIn("roads", uk._clean_settings({"roads": "grid"}))

    def test_bad_requests(self):
        c = client()
        self.assertEqual(c.post("/api/uk/setup", json={"name": "a b"}).status_code, 400)
        self.assertEqual(c.post("/api/uk/setup", json={"name": "Town"}).status_code, 400)   # no box
        self.assertEqual(c.post("/api/uk/run", json={"map": "No_such_map_xyz"}).status_code, 400)
        self.assertEqual(c.get("/api/uk/job?map=No_such_map_xyz").get_json(), {"state": "idle"})


@unittest.skipUnless(os.path.exists(os.path.join(uk.OUTPUT_DIR, TEST_MAP, "uk_pipeline.json")),
                     f"needs the built {TEST_MAP}")
class Run(unittest.TestCase):
    def test_a_run_to_its_end(self):
        c = client()
        path = os.path.join(uk.OUTPUT_DIR, TEST_MAP, "uk.json")
        before = open(path, encoding="utf-8").read()
        try:
            r = c.post("/api/uk/settings", json={"map": TEST_MAP, "settings": json.loads(before)})
            self.assertEqual(r.status_code, 200)
            # the last step only (--from validate): the job's whole path, in seconds
            self.assertEqual(c.post("/api/uk/run", json={"map": TEST_MAP, "from": "validate"}).status_code, 200)
            self.assertEqual(c.post("/api/uk/run", json={"map": TEST_MAP}).status_code, 409)   # one at a time
            for _ in range(480):
                job = c.get(f"/api/uk/job?map={TEST_MAP}").get_json()
                if job["state"] not in ("running", "stopping"):
                    break
                time.sleep(0.5)
            self.assertEqual(job["state"], "done", job.get("lines"))
            self.assertEqual((job["k"], job["n"], job["step"]), (16, 16, "validate"))
            self.assertTrue(any("files valid" in ln for ln in job["lines"]))
        finally:
            open(path, "w", encoding="utf-8").write(before)


if __name__ == "__main__":
    unittest.main()
