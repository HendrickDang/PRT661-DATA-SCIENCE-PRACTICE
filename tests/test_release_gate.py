"""Tests for the forecast publication gate (release_gate.py, monitoring.py).

Run from the repository root:
    python -m unittest discover tests

Every test works in a temporary Outputs folder built from the committed
Outputs/F1_forecasts.csv; the repository's own files are never written.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import release_gate  # noqa: E402

APPROVED = REPO / "Outputs" / "F1_forecasts.csv"


class ReleaseGateTest(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "Outputs"
        (self.out / "staging").mkdir(parents=True)
        # approved = last published forecast; candidate = a new, different forecast
        shutil.copyfile(APPROVED, self.out / "F1_forecasts.csv")
        cand = pd.read_csv(APPROVED)
        cand["Forecast_rate_100k"] = cand["Forecast_rate_100k"] + 1.0
        cand[["Lower_80", "Upper_80"]] = cand[["Lower_80", "Upper_80"]] + 1.0
        cand.to_csv(self.out / "staging" / "F1_forecasts.csv", index=False)
        self.approved_bytes = (self.out / "F1_forecasts.csv").read_bytes()
        self.candidate_bytes = (self.out / "staging" / "F1_forecasts.csv").read_bytes()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def dashboard_view(self) -> pd.DataFrame:
        """What the Administrator Portal reads (core_data.FORECAST_FILES)."""
        return pd.read_csv(self.out / "F1_forecasts.csv")

    def test_committed_forecast_passes_integrity_checks(self):
        self.assertEqual(release_gate.candidate_problems(APPROVED), [])

    def test_failed_release_keeps_last_approved_forecast(self):
        before = self.dashboard_view()
        result = release_gate.publish(promote=False, out_dir=self.out)
        self.assertEqual(result["action"], "rejected")
        self.assertFalse(result["published"])
        self.assertEqual((self.out / "F1_forecasts.csv").read_bytes(), self.approved_bytes)
        pd.testing.assert_frame_equal(self.dashboard_view(), before)
        # the rejected candidate stays in staging for investigation
        self.assertEqual((self.out / "staging" / "F1_forecasts.csv").read_bytes(),
                         self.candidate_bytes)

    def test_passed_release_publishes_candidate(self):
        result = release_gate.publish(promote=True, out_dir=self.out)
        self.assertEqual(result["action"], "published")
        self.assertEqual((self.out / "F1_forecasts.csv").read_bytes(), self.candidate_bytes)
        self.assertFalse((self.out / ".F1_forecasts.csv.tmp").exists())

    def test_invalid_candidate_is_never_published(self):
        bad = pd.read_csv(self.out / "staging" / "F1_forecasts.csv")
        rec = bad["Recommended"].astype(str) == "True"
        bad.loc[rec, "Lower_80"] = bad.loc[rec, "Forecast_rate_100k"] + 50   # range misses the forecast
        bad.to_csv(self.out / "staging" / "F1_forecasts.csv", index=False)
        result = release_gate.publish(promote=True, out_dir=self.out)
        self.assertEqual(result["action"], "rejected")
        self.assertIn("80% range", result["reason"])
        self.assertEqual((self.out / "F1_forecasts.csv").read_bytes(), self.approved_bytes)

    def test_missing_region_is_rejected(self):
        cand = pd.read_csv(self.out / "staging" / "F1_forecasts.csv")
        drop = (cand["Recommended"].astype(str) == "True") & (cand["Horizon_months"] == 1)
        cand = cand.drop(cand[drop].index[:1])
        cand.to_csv(self.out / "staging" / "F1_forecasts.csv", index=False)
        result = release_gate.publish(promote=True, out_dir=self.out)
        self.assertEqual(result["action"], "rejected")
        self.assertEqual((self.out / "F1_forecasts.csv").read_bytes(), self.approved_bytes)

    def test_no_candidate_changes_nothing(self):
        (self.out / "staging" / "F1_forecasts.csv").unlink()
        result = release_gate.publish(promote=True, out_dir=self.out)
        self.assertEqual(result["action"], "none")
        self.assertEqual((self.out / "F1_forecasts.csv").read_bytes(), self.approved_bytes)


class PromotionDecisionTest(unittest.TestCase):
    """monitoring.decide_promotion: a forced FAIL blocks publication."""

    @classmethod
    def setUpClass(cls):
        import monitoring
        cls.mon = monitoring

    def checks(self, area: str, status: str) -> pd.DataFrame:
        return pd.DataFrame({"Check_id": ["X1"], "Area": [area], "Status": [status]})

    def test_data_fail_blocks(self):
        self.assertFalse(self.mon.decide_promotion(self.checks("Data", "FAIL"), {}))

    def test_settled_performance_fail_blocks(self):
        self.assertFalse(self.mon.decide_promotion(self.checks("Performance", "WARN"),
                                                   {"P1": "FAIL"}))

    def test_warnings_do_not_block(self):
        self.assertTrue(self.mon.decide_promotion(self.checks("Data", "WARN"),
                                                  {"P1": "WARN"}))

    def test_forced_fail_end_to_end(self):
        """A FAIL decision passed to the gate leaves the dashboard file unchanged."""
        tmp = Path(tempfile.mkdtemp())
        try:
            out = tmp / "Outputs"
            (out / "staging").mkdir(parents=True)
            shutil.copyfile(APPROVED, out / "F1_forecasts.csv")
            cand = pd.read_csv(APPROVED)
            cand["Forecast_rate_100k"] = cand["Forecast_rate_100k"] * 2
            cand[["Lower_80", "Upper_80"]] = cand[["Lower_80", "Upper_80"]] * 2
            cand.to_csv(out / "staging" / "F1_forecasts.csv", index=False)
            before = (out / "F1_forecasts.csv").read_bytes()
            promote = self.mon.decide_promotion(self.checks("Data", "FAIL"), {})
            release_gate.publish(promote, out)
            self.assertEqual((out / "F1_forecasts.csv").read_bytes(), before)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class DashboardReadsApprovedFileTest(unittest.TestCase):

    def test_dashboard_points_at_approved_file_not_staging(self):
        src = (REPO / "core_data.py").read_text(encoding="utf-8")
        self.assertIn('OUTPUTS_DIR = BASE_DIR / "Outputs"', src)
        self.assertIn('"forecasts": "F1_forecasts.csv"', src)
        self.assertNotIn("staging", src)


if __name__ == "__main__":
    unittest.main()
