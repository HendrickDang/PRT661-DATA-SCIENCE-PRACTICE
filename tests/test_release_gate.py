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


class GateFixture(unittest.TestCase):
    """Temporary Outputs folder with an approved file and a different candidate."""

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


class ReleaseGateTest(GateFixture):

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


class ExpectedSetTest(GateFixture):
    """The expected regions and horizons are fixed, not read from the candidate."""

    def write(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.out / "staging" / "F1_forecasts.csv", index=False)

    def candidate(self) -> pd.DataFrame:
        return pd.read_csv(self.out / "staging" / "F1_forecasts.csv")

    def assert_rejected(self, text: str) -> None:
        result = release_gate.publish(promote=True, out_dir=self.out)
        self.assertEqual(result["action"], "rejected")
        self.assertIn(text, result["reason"])
        self.assertEqual((self.out / "F1_forecasts.csv").read_bytes(), self.approved_bytes)

    def test_whole_region_missing_is_rejected(self):
        cand = self.candidate()
        self.write(cand[cand["Region"] != "Barkly"])
        self.assert_rejected("missing: Barkly")

    def test_whole_horizon_missing_is_rejected(self):
        cand = self.candidate()
        self.write(cand[cand["Horizon_months"] != 12])
        self.assert_rejected("horizons must be exactly 1, 3, 6, 12")

    def test_unexpected_region_is_rejected(self):
        cand = self.candidate()
        extra = cand[cand["Region"] == "Barkly"].assign(Region="Tiwi Islands")
        self.write(pd.concat([cand, extra], ignore_index=True))
        self.assert_rejected("unexpected: Tiwi Islands")

    def test_wrong_target_month_is_rejected(self):
        cand = self.candidate()
        cand.loc[cand["Horizon_months"] == 3, "Target_month"] = "2026-04"
        self.write(cand)
        self.assert_rejected("target months do not equal origin plus horizon")

    def test_expected_set_matches_project(self):
        import evaluation as ev
        self.assertEqual(tuple(ev.HORIZONS), release_gate.EXPECTED_HORIZONS)
        panel = pd.read_csv(REPO / "dataset" / "processed" / "nt_crime_merged_2015_2025.csv",
                            usecols=["Region"])
        self.assertEqual(set(panel["Region"]), set(release_gate.EXPECTED_REGIONS))


class DashboardStatusTest(GateFixture):
    """The Administrator Portal reports what was published, not only the checks."""

    def summary(self, promote: bool, release: dict) -> dict:
        return {"overall_status": "WARN", "promote_forecasts_to_dashboard": promote,
                "forecast_release": release}

    def test_published_release_is_cleared(self):
        release = release_gate.publish(promote=True, out_dir=self.out)
        status = release_gate.dashboard_status(self.summary(True, release))
        self.assertTrue(status["cleared"])
        self.assertIn("Internal release checks passed", status["verdict"])
        self.assertIn("prototype demonstration only", status["verdict"])

    def test_checks_pass_but_invalid_candidate_is_not_cleared(self):
        """Monitoring allows promotion, the gate rejects a malformed file:
        the dashboard must not present the forecasts as approved."""
        cand = pd.read_csv(self.out / "staging" / "F1_forecasts.csv")
        cand[cand["Region"] != "Top End"].to_csv(self.out / "staging" / "F1_forecasts.csv", index=False)
        release = release_gate.publish(promote=True, out_dir=self.out)
        status = release_gate.dashboard_status(self.summary(True, release))
        self.assertFalse(status["cleared"])
        self.assertEqual(status["level"], "FAIL")
        self.assertNotIn("checks passed and these forecasts were published", status["verdict"])
        self.assertIn("last approved", status["verdict"])
        self.assertEqual((self.out / "F1_forecasts.csv").read_bytes(), self.approved_bytes)

    def test_failed_checks_are_not_cleared(self):
        release = release_gate.publish(promote=False, out_dir=self.out)
        status = release_gate.dashboard_status(self.summary(False, release))
        self.assertFalse(status["cleared"])
        self.assertIn("last approved", status["verdict"])

    def test_record_without_release_is_not_cleared(self):
        status = release_gate.dashboard_status({"overall_status": "PASS",
                                                "promote_forecasts_to_dashboard": True})
        self.assertFalse(status["cleared"])

    def test_admin_portal_uses_release_record(self):
        src = (REPO / "pages" / "1_Administrator_Portal.py").read_text(encoding="utf-8")
        self.assertIn("dashboard_status(mon)", src)
        self.assertNotIn('if mon["promote_forecasts_to_dashboard"]', src)


class DeploymentRuleTest(unittest.TestCase):
    """E12: a method is deployed only if it leads the full backtest and beats
    the seasonal naive value after the selection window."""

    def test_committed_results(self):
        import evaluation as ev
        chosen = ev.deployed_methods(pd.read_csv(REPO / "Outputs" / "E4_backtest_metrics.csv"),
                                     pd.read_csv(REPO / "Outputs" / "E4_backtest_metrics_post_selection.csv"))
        self.assertEqual(chosen, {1: "XGBoost", 3: "Moving average (12m)",
                                  6: "Moving average (12m)", 12: "Seasonal naive (t-12)"})

    def test_full_window_leader_that_loses_after_selection_falls_back(self):
        import evaluation as ev
        full = pd.DataFrame({"Model": ["XGBoost", ev.SEASONAL_NAIVE], "Horizon_months": [1, 1],
                             "RMSE": [90.0, 100.0]})
        post = pd.DataFrame({"Model": ["XGBoost", ev.SEASONAL_NAIVE], "Horizon_months": [1, 1],
                             "RMSE": [110.0, 100.0]})
        self.assertEqual(ev.deployed_methods(full, post), {1: ev.SEASONAL_NAIVE})


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
