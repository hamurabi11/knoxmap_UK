"""The UK pipeline runner (KnoxMap UK: Building_types/src/uk_pipeline.py; PLAN U stage 3): a
second run with nothing changed skips every step; a changed option runs its step and every
step after it; a step run alone (--only) leaves the steps after it to run next time; a
failed step keeps no stamp."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)

import uk_pipeline as U  # noqa: E402


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        os.makedirs(os.path.join(t, "bt", "data", "town"))
        os.makedirs(os.path.join(t, "knox", "output"))
        self.was = U.KNOX, U.BT, U.steps
        U.KNOX, U.BT = os.path.join(t, "knox"), os.path.join(t, "bt")
        self.ran = os.path.join(t, "ran.txt")
        code = os.path.join(t, "code.py")
        open(code, "w").write("# step code\n")

        def steps(m, a, s):
            say = lambda n: ["-c", f"open(r'{self.ran}', 'a').write('{n} ')" + (  # noqa: E731
                "; raise SystemExit(3)" if s.get("fail") == n else "")]
            return [("one", say("one"), [code], ()), ("two", say("two"), [code], ("dormers",)),
                    ("three", say("three"), [code], ())]
        U.steps = steps

    def tearDown(self):
        U.KNOX, U.BT, U.steps = self.was
        self.tmp.cleanup()

    def run_(self, *extra, **settings):
        if settings:
            path = os.path.join(U.KNOX, "output", "T", "uk.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            d = json.load(open(path)) if os.path.exists(path) else {}
            d.update(settings)
            json.dump(d, open(path, "w"))
        if os.path.exists(self.ran):
            os.remove(self.ran)
        sys.argv = ["uk_pipeline.py", "T", "--source", "S", "--area", "town", *extra]
        with redirect_stdout(io.StringIO()):
            code = U.main()
        return code, (open(self.ran).read().split() if os.path.exists(self.ran) else [])

    def test_unchanged_skips_and_an_option_reruns_from_its_step(self):
        self.assertEqual(self.run_(), (0, ["one", "two", "three"]))
        self.assertEqual(self.run_(), (0, []))
        self.assertEqual(self.run_(dormers=True), (0, ["two", "three"]))

    def test_a_step_run_alone_leaves_the_later_ones_to_run(self):
        self.run_()
        self.assertEqual(self.run_("--only", "two"), (0, ["two"]))
        self.assertEqual(self.run_(), (0, ["three"]))

    def test_a_failed_step_runs_again(self):
        self.assertEqual(self.run_(fail="two")[0], 3)
        self.assertEqual(self.run_(fail=None), (0, ["two", "three"]))


if __name__ == "__main__":
    unittest.main()
