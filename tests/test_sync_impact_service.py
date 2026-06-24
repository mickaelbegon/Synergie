import unittest

from types import SimpleNamespace

import pandas as pd

from synergie.services.sync_impact_service import (
    SyncImpact,
    detect_sync_impacts,
    prepare_sync_impact_signal,
    sync_impact_list_labels,
    sync_impact_plot_title,
    sync_impact_review_warning,
    sync_impact_selected_status,
    sync_impact_signal_window,
)


class SyncImpactServiceTests(unittest.TestCase):
    def test_detect_sync_impacts_returns_strong_stable_acceleration_change(self):
        ms = list(range(0, 2100, 100))
        frame = pd.DataFrame(
            {
                "ms": ms,
                "Acc_X": [0 if value < 1200 else 30 for value in ms],
                "Acc_Y": [0 for _value in ms],
                "Acc_Z": [1 for _value in ms],
            }
        )

        impacts = detect_sync_impacts(frame, max_candidates=1)

        self.assertEqual(len(impacts), 1)
        self.assertEqual(impacts[0].ms, 1200)
        self.assertIn(impacts[0].confidence, {"medium", "high"})

    def test_detect_sync_impacts_rejects_candidate_before_one_second(self):
        ms = list(range(0, 1600, 100))
        frame = pd.DataFrame(
            {
                "ms": ms,
                "Acc_X": [0 if value < 500 else 30 for value in ms],
                "Acc_Y": [0 for _value in ms],
                "Acc_Z": [1 for _value in ms],
            }
        )

        self.assertEqual(detect_sync_impacts(frame, max_candidates=1), [])

    def test_detect_sync_impacts_rejects_unstable_context(self):
        ms = list(range(0, 2100, 100))
        acc_x = []
        for value in ms:
            if value < 1200:
                acc_x.append(20 if (value // 100) % 2 else 0)
            else:
                acc_x.append(60 if (value // 100) % 2 else 40)
        frame = pd.DataFrame(
            {
                "ms": ms,
                "Acc_X": acc_x,
                "Acc_Y": [0 for _value in ms],
                "Acc_Z": [1 for _value in ms],
            }
        )

        self.assertEqual(detect_sync_impacts(frame, max_candidates=1), [])

    def test_detect_sync_impacts_ignores_flat_signal(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 10, 20, 30],
                "Acc_X": [1, 1, 1, 1],
                "Acc_Y": [0, 0, 0, 0],
                "Acc_Z": [0, 0, 0, 0],
            }
        )

        self.assertEqual(detect_sync_impacts(frame), [])

    def test_sync_impact_signal_window_returns_view_and_acc_norm(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 1000, 2000, 3000, 4000],
                "Acc_X": [3, 0, 3, 0, 0],
                "Acc_Y": [4, 0, 4, 0, 0],
                "Acc_Z": [0, 1, 0, 1, 2],
            }
        )

        view, acc_norm = sync_impact_signal_window(frame, 2000.0, before_ms=500.0, after_ms=1000.0)

        self.assertEqual(view["ms"].tolist(), [2000, 3000])
        self.assertEqual(acc_norm.tolist(), [5.0, 1.0])

    def test_sync_impact_signal_window_falls_back_when_window_is_empty(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 100, 200],
                "Acc_X": [1, 2, 3],
                "Acc_Y": [0, 0, 0],
                "Acc_Z": [0, 0, 0],
            }
        )

        view, acc_norm = sync_impact_signal_window(frame, 10_000.0, before_ms=10.0, after_ms=10.0, fallback_rows=2)

        self.assertEqual(view["ms"].tolist(), [0, 100])
        self.assertEqual(acc_norm.tolist(), [1.0, 2.0])

    def test_sync_impact_list_labels_are_stable_for_gui(self):
        impacts = [
            SyncImpact(index=10, ms=1200.0, strength=6.234, confidence="medium"),
            SyncImpact(index=20, ms=3925.0, strength=17.0, confidence="high"),
        ]

        self.assertEqual(
            sync_impact_list_labels(impacts),
            [
                "01 | 1200 ms | strength 6.23 | medium",
                "02 | 3925 ms | strength 17.00 | high",
            ],
        )

    def test_sync_impact_review_warning_guides_manual_review(self):
        self.assertIn("No reliable", sync_impact_review_warning(0))
        self.assertIn("One sync impact", sync_impact_review_warning(1))
        self.assertIn("3 sync impact candidates", sync_impact_review_warning(3))

    def test_sync_impact_status_and_title_include_detection_basis(self):
        impact = SyncImpact(index=10, ms=3925.0, strength=17.0, confidence="high")

        self.assertIn("3D acceleration norm", sync_impact_selected_status(impact, 2))
        self.assertEqual(
            sync_impact_plot_title(impact, 2),
            "Sync impact #2: 3925 ms | 3D acceleration-norm change strength 17.00 | high",
        )

    def test_prepare_sync_impact_signal_loads_cleans_and_windows_raw_session(self):
        raw = pd.DataFrame(
            {
                "ms": [0, 1000, 2000],
                "Acc_X": [0.0, 3.0, 0.0],
                "Acc_Y": [0.0, 4.0, 0.0],
                "Acc_Z": [1.0, 0.0, 2.0],
            }
        )

        def fake_training_session(frame):
            return SimpleNamespace(df=frame.copy())

        def fake_clean(frame, acceleration_limit_g):
            cleaned = frame.copy()
            cleaned["clean_limit"] = acceleration_limit_g
            return cleaned, {"replaced": 0}

        def fake_recompute(frame, smoothing_sigma, threshold):
            prepared = frame.copy()
            prepared["smoothing_sigma"] = smoothing_sigma
            prepared["threshold"] = threshold
            return prepared

        result = prepare_sync_impact_signal(
            "raw.csv",
            1000.0,
            acceleration_limit_g=32.0,
            smoothing_sigma=3.0,
            threshold=-0.2,
            read_csv=lambda path, low_memory: raw,
            training_session_factory=fake_training_session,
            clean_frame=fake_clean,
            recompute_derivatives=fake_recompute,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["cleaning_report"], {"replaced": 0})
        self.assertEqual(result["view_df"]["ms"].tolist(), [0, 1000, 2000])
        self.assertEqual(result["acc_norm"].tolist(), [1.0, 5.0, 2.0])
        self.assertEqual(result["dataframe"]["smoothing_sigma"].tolist(), [3.0, 3.0, 3.0])

    def test_prepare_sync_impact_signal_reports_missing_timeline(self):
        result = prepare_sync_impact_signal(
            "raw.csv",
            1000.0,
            acceleration_limit_g=32.0,
            smoothing_sigma=3.0,
            threshold=-0.2,
            read_csv=lambda path, low_memory: pd.DataFrame({"Acc_X": [1.0]}),
            training_session_factory=lambda frame: SimpleNamespace(df=frame),
        )

        self.assertEqual(result["status"], "missing_timeline")

    def test_prepare_sync_impact_signal_reports_missing_acceleration(self):
        result = prepare_sync_impact_signal(
            "raw.csv",
            1000.0,
            acceleration_limit_g=32.0,
            smoothing_sigma=3.0,
            threshold=-0.2,
            read_csv=lambda path, low_memory: pd.DataFrame({"ms": [0, 1000], "Gyr_X": [1.0, 2.0]}),
            training_session_factory=lambda frame: SimpleNamespace(df=frame),
            clean_frame=lambda frame, acceleration_limit_g: (frame, {"replaced": 0}),
            recompute_derivatives=lambda frame, smoothing_sigma, threshold: frame,
        )

        self.assertEqual(result["status"], "missing_acceleration")


if __name__ == "__main__":
    unittest.main()
