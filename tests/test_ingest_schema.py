"""Tests for the ingest schema check in Updated_End_to_End_pipeline.py.

Run from the repository root:
    python -m unittest discover tests

They read dataset/source/ and dataset/raw/ but never write there: every
ingest run happens in a temporary copy of the project folders.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import Updated_End_to_End_pipeline as pl  # noqa: E402

SOURCE = REPO / "dataset" / "source"
RAW = REPO / "dataset" / "raw"


def source_frame(name: str) -> pd.DataFrame:
    return pl.load_one(pl.EXPECTED_FILES[name])


class CheckSchemaOnCurrentFiles(unittest.TestCase):
    """The files the reported results were built on must all pass."""

    def test_all_six_source_files_pass(self):
        for name in pl.EXPECTED_FILES:
            with self.subTest(file=name):
                self.assertEqual(pl.check_schema(name, source_frame(name)), [])


class CheckSchemaRejectsBrokenFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.crime = source_frame("crime_new").head(500).copy()
        cls.population = source_frame("population").head(500).copy()
        cls.alcohol = source_frame("alcohol_2025").copy()

    def assert_rejected(self, name, df, words):
        errors = pl.check_schema(name, df)
        self.assertTrue(errors, "a broken file passed the schema check")
        self.assertIn(words, " ".join(errors))

    def test_missing_column(self):
        self.assert_rejected("crime_new", self.crime.drop(columns="Number of offences"),
                             "missing required column")

    def test_renamed_column(self):
        renamed = self.crime.rename(columns={"DV involvement": "Family violence"})
        self.assert_rejected("crime_new", renamed, "missing required column")

    def test_new_flag_value(self):
        df = self.crime.copy()
        df.loc[df.index[0], "Alcohol involvement"] = "Y"
        self.assert_rejected("crime_new", df, "unexpected value")

    def test_blank_flag(self):
        df = self.crime.copy()
        df.loc[df.index[0], "DV involvement"] = None
        self.assert_rejected("crime_new", df, "unexpected value")

    def test_month_out_of_range(self):
        df = self.crime.copy()
        df.loc[df.index[0], "Month number"] = 13
        self.assert_rejected("crime_new", df, "outside 1-12")

    def test_text_in_offence_count(self):
        df = self.crime.copy()
        df["Number of offences"] = df["Number of offences"].astype(object)
        df.loc[df.index[0], "Number of offences"] = "n/a"
        self.assert_rejected("crime_new", df, "Number of offences")

    def test_negative_offence_count(self):
        df = self.crime.copy()
        df.loc[df.index[0], "Number of offences"] = -1
        self.assert_rejected("crime_new", df, "Number of offences")

    def test_blank_offence_category(self):
        df = self.crime.copy()
        df.loc[df.index[0], "Offence category"] = None
        self.assert_rejected("crime_new", df, "blank value")

    def test_empty_file(self):
        self.assert_rejected("crime_new", self.crime.iloc[0:0], "no rows")

    def test_negative_population(self):
        df = self.population.copy()
        df.loc[df.index[0], "Population"] = -5
        self.assert_rejected("population", df, "Population")

    def test_alcohol_date_not_a_date(self):
        df = self.alcohol.copy()
        df["Quarter Ending"] = df["Quarter Ending"].astype(str)
        df.loc[df.index[0], "Quarter Ending"] = "Q5 2025"
        self.assert_rejected("alcohol_2025", df, "not dates")


class CheckSchemaAllowsKnownDrift(unittest.TestCase):

    def test_header_spacing_and_case_drift(self):
        crime = source_frame("crime_old").head(200)
        drifted = crime.rename(columns={"Offence type": "Offence type ",
                                        "Reporting region": "REPORTING REGION"})
        self.assertEqual(pl.check_schema("crime_old", drifted), [])

    def test_blank_sa2_is_allowed(self):
        crime = source_frame("crime_new").head(200).copy()
        crime["Statistical Area 2"] = None
        self.assertEqual(pl.check_schema("crime_new", crime), [])


class IngestInTemporaryProject(unittest.TestCase):
    """Runs ingest_main against a temporary copy of dataset/, so the
    repository's own files are never touched."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "dataset" / "source"
        self.raw = self.tmp / "dataset" / "raw"
        shutil.copytree(SOURCE, self.source)
        shutil.copytree(RAW, self.raw)
        self.saved = {n: getattr(pl, n) for n in ("BASE_DIR", "SOURCE_DIR", "RAW_DIR")}
        pl.BASE_DIR, pl.SOURCE_DIR, pl.RAW_DIR = self.tmp, self.source, self.raw

    def tearDown(self):
        for n, value in self.saved.items():
            setattr(pl, n, value)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def raw_snapshot(self) -> dict:
        return {p.name: p.read_bytes() for p in sorted(self.raw.iterdir())}

    def run_ingest(self) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pl.ingest_main()
        return out.getvalue()

    def test_broken_file_stops_ingest_and_writes_nothing(self):
        before = self.raw_snapshot()
        crime = pd.read_csv(self.source / "nt_crime_statistics_latest.csv")
        crime.loc[crime.index[0], "DV involvement"] = "Y"
        crime.to_csv(self.source / "nt_crime_statistics_latest.csv", index=False)

        with self.assertRaises(SystemExit) as stop:
            self.run_ingest()
        self.assertEqual(stop.exception.code, 1)
        self.assertEqual(self.raw_snapshot(), before,
                         "dataset/raw/ changed although ingest was rejected")

    def test_missing_file_stops_ingest_and_writes_nothing(self):
        before = self.raw_snapshot()
        (self.source / "wholesale-alcohol-supply-by-quarter-2024.xlsx").unlink()
        with self.assertRaises(SystemExit):
            self.run_ingest()
        self.assertEqual(self.raw_snapshot(), before)

    def test_current_files_reproduce_the_committed_raw_copies(self):
        before = self.raw_snapshot()
        log = self.run_ingest()
        self.assertIn("finished OK", log)
        after = self.raw_snapshot()
        for name in before:
            if name == "manifest.json":   # holds the run time, so it always changes
                continue
            with self.subTest(file=name):
                # Git may check files out with CRLF on Windows; compare the text.
                self.assertEqual(after[name].replace(b"\r\n", b"\n"),
                                 before[name].replace(b"\r\n", b"\n"))


if __name__ == "__main__":
    unittest.main()
