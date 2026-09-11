import json
import tempfile
import unittest
from pathlib import Path

import h5py
import pandas as pd

from synergie.services.annotation_finalization_service import finalize_annotation_file
from synergie.services.training_dataset_service import load_training_dataset_state


class AnnotationFinalizationServiceTests(unittest.TestCase):
    def test_finalize_annotation_file_archives_moves_and_merges_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "data" / "annotated" / "total"
            annotated_root = root / "data" / "annotated"
            pending_root = root / "data" / "pending"
            segment_dir = pending_root / "segments" / "session" / "sensor1"
            dataset_root.mkdir(parents=True)
            segment_dir.mkdir(parents=True)
            (dataset_root / "jumplist.csv").write_text("path,type,success,skater\nexisting.csv,0,1,alice\n", encoding="utf-8")
            segment = segment_dir / "jump001.csv"
            segment.write_text("ms,Gyr_X\n0,0\n", encoding="utf-8")
            annotation_file = pending_root / "20250911_085656_for_annotation.csv"
            pd.DataFrame(
                [
                    {
                        "path": segment.as_posix(),
                        "type": 1,
                        "success": 1,
                        "skater": "sensor_1",
                        "athlete_id": "bob",
                        "annotation_status": "annotated",
                        "sensor_id": "1",
                        "session_key": "20250911_085656",
                    }
                ]
            ).to_csv(annotation_file, index=False)

            result = finalize_annotation_file(annotation_file, dataset_path=dataset_root, annotated_root=annotated_root)

            merged = pd.read_csv(dataset_root / "jumplist.csv")
            self.assertEqual(result["rows_added"], 1)
            self.assertEqual(merged.iloc[1]["skater"], "bob")
            self.assertTrue(load_training_dataset_state(dataset_root)["retraining_recommended"])

    def test_finalize_copies_missing_segment_from_csv_archive_without_mutating_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "data" / "annotated" / "total"
            annotated_root = root / "data" / "annotated"
            pending_root = root / "data" / "pending"
            archive_root = root / "data" / "segment_csv_archive"
            dataset_root.mkdir(parents=True)
            pending_root.mkdir(parents=True)
            original_jumplist = b"path,type,success,skater\nexisting.csv,0,1,alice\n"
            (dataset_root / "jumplist.csv").write_bytes(original_jumplist)

            requested_segment = Path("data/pending/segments/session/sensor1/jump001.csv")
            archived_segment = archive_root / requested_segment
            archived_segment.parent.mkdir(parents=True)
            archived_bytes = b"ms,Gyr_X\n0,12.5\n"
            archived_segment.write_bytes(archived_bytes)
            annotation_file = pending_root / "20250911_085656_for_annotation.csv"
            pd.DataFrame(
                [
                    {
                        "path": requested_segment.as_posix(),
                        "type": 1,
                        "success": 1,
                        "athlete_id": "bob",
                        "annotation_status": "annotated",
                        "sensor_id": "1",
                        "session_key": "20250911_085656",
                    }
                ]
            ).to_csv(annotation_file, index=False)
            annotation_bytes = annotation_file.read_bytes()

            result = finalize_annotation_file(
                annotation_file,
                dataset_path=dataset_root,
                annotated_root=annotated_root,
                segment_archive_root=archive_root,
                hdf5_archive_path=None,
            )

            destination = annotated_root / "20250911_085656" / "sensor1" / requested_segment.name
            self.assertEqual(result["archive_segments_copied"], 1)
            self.assertEqual(result["live_segments_moved"], 0)
            self.assertEqual(destination.read_bytes(), archived_bytes)
            self.assertEqual(archived_segment.read_bytes(), archived_bytes)
            self.assertTrue(archived_segment.exists())
            self.assertFalse(requested_segment.exists())
            self.assertFalse(annotation_file.exists())
            self.assertEqual(result["completed_annotation_path"].read_bytes(), annotation_bytes)

    def test_dry_run_resolves_archive_and_makes_no_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "annotated" / "total"
            annotated_root = root / "annotated"
            pending_root = root / "pending"
            archive_root = root / "segment_csv_archive"
            dataset_root.mkdir(parents=True)
            pending_root.mkdir()
            original_jumplist = b"path,type,success,skater\nexisting.csv,0,1,alice\n"
            jumplist = dataset_root / "jumplist.csv"
            jumplist.write_bytes(original_jumplist)
            requested_segment = Path("data/pending/segments/session/sensor2/jump001.csv")
            archived_segment = archive_root / requested_segment
            archived_segment.parent.mkdir(parents=True)
            archived_bytes = b"ms,Gyr_X\n0,7\n"
            archived_segment.write_bytes(archived_bytes)
            annotation_file = pending_root / "session_for_annotation.csv"
            annotation_bytes = _write_annotation(annotation_file, requested_segment, session_key="session", sensor_id="2")

            result = finalize_annotation_file(
                annotation_file,
                dataset_path=dataset_root,
                annotated_root=annotated_root,
                segment_archive_root=archive_root,
                hdf5_archive_path=None,
                dry_run=True,
            )

            destination = annotated_root / "session" / "sensor2" / requested_segment.name
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["rows_added"], 0)
            self.assertEqual(result["rows_to_add"], 1)
            self.assertEqual(result["archive_segments_to_copy"], 1)
            self.assertEqual(result["segment_transfers"][0]["operation"], "copy")
            self.assertEqual(jumplist.read_bytes(), original_jumplist)
            self.assertEqual(annotation_file.read_bytes(), annotation_bytes)
            self.assertEqual(archived_segment.read_bytes(), archived_bytes)
            self.assertFalse(destination.exists())
            self.assertFalse((dataset_root / "archive").exists())

    def test_dry_run_refuses_overlapping_labelled_annotation_files_without_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "annotated" / "total"
            pending_root = root / "pending"
            archive_root = root / "segment_csv_archive"
            dataset_root.mkdir(parents=True)
            pending_root.mkdir()
            jumplist = dataset_root / "jumplist.csv"
            original_jumplist = b"path,type,success,skater\n"
            jumplist.write_bytes(original_jumplist)
            requested_segment = Path("data/pending/segments/session/sensor1/jump001.csv")
            archived_segment = archive_root / requested_segment
            archived_segment.parent.mkdir(parents=True)
            archived_segment.write_text("ms,Gyr_X\n0,1\n", encoding="utf-8")
            target = pending_root / "session_sensor1_for_annotation_2.csv"
            sibling = pending_root / "session_sensor1_for_annotation_3.csv"
            target_bytes = _write_annotation(target, requested_segment)
            sibling_bytes = _write_annotation(sibling, requested_segment)

            with self.assertRaisesRegex(ValueError, "another labelled annotation file overlaps"):
                finalize_annotation_file(
                    target,
                    dataset_path=dataset_root,
                    annotated_root=root / "annotated",
                    segment_archive_root=archive_root,
                    hdf5_archive_path=None,
                    dry_run=True,
                )

            self.assertEqual(target.read_bytes(), target_bytes)
            self.assertEqual(sibling.read_bytes(), sibling_bytes)
            self.assertEqual(jumplist.read_bytes(), original_jumplist)
            self.assertTrue(archived_segment.exists())
            self.assertFalse((dataset_root / "archive").exists())

    def test_dry_run_ignores_pending_prefill_values_in_overlapping_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "annotated" / "total"
            pending_root = root / "pending"
            archive_root = root / "segment_csv_archive"
            dataset_root.mkdir(parents=True)
            pending_root.mkdir()
            jumplist = dataset_root / "jumplist.csv"
            original_jumplist = b"path,type,success,skater\n"
            jumplist.write_bytes(original_jumplist)
            requested_segment = Path("data/pending/segments/session/sensor1/jump001.csv")
            archived_segment = archive_root / requested_segment
            archived_segment.parent.mkdir(parents=True)
            archived_segment.write_text("ms,Gyr_X\n0,1\n", encoding="utf-8")
            target = pending_root / "session_for_annotation.csv"
            sibling = pending_root / "session_sensor1_for_annotation_2.csv"
            target_bytes = _write_annotation(target, requested_segment)
            sibling_bytes = _write_annotation(sibling, requested_segment)
            sibling_rows = pd.read_csv(sibling)
            sibling_rows["annotation_status"] = "pending"
            sibling_rows["prediction_source"] = "batch_model_prefill"
            sibling_rows.to_csv(sibling, index=False)
            sibling_bytes = sibling.read_bytes()

            result = finalize_annotation_file(
                target,
                dataset_path=dataset_root,
                annotated_root=root / "annotated",
                segment_archive_root=archive_root,
                hdf5_archive_path=None,
                dry_run=True,
            )

            self.assertTrue(result["dry_run"])
            self.assertEqual(result["rows_to_add"], 1)
            self.assertEqual(target.read_bytes(), target_bytes)
            self.assertEqual(sibling.read_bytes(), sibling_bytes)
            self.assertEqual(jumplist.read_bytes(), original_jumplist)
            self.assertTrue(archived_segment.exists())
            self.assertFalse((dataset_root / "archive").exists())

    def test_dry_run_reports_pending_before_prefill_sibling_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "annotated" / "total"
            pending_root = root / "pending"
            dataset_root.mkdir(parents=True)
            pending_root.mkdir()
            jumplist = dataset_root / "jumplist.csv"
            original_jumplist = b"path,type,success,skater\n"
            jumplist.write_bytes(original_jumplist)
            requested_segment = Path("data/pending/segments/session/sensor1/jump001.csv")
            target = pending_root / "session_sensor1_for_annotation_2.csv"
            sibling = pending_root / "session_sensor1_for_annotation_3.csv"
            _write_annotation(target, requested_segment)
            _write_annotation(sibling, requested_segment)
            for path in (target, sibling):
                rows = pd.read_csv(path)
                rows["annotation_status"] = "pending"
                rows["prediction_source"] = "batch_model_prefill"
                rows.to_csv(path, index=False)
            target_bytes = target.read_bytes()
            sibling_bytes = sibling.read_bytes()

            with self.assertRaisesRegex(ValueError, "1 annotations are still pending"):
                finalize_annotation_file(
                    target,
                    dataset_path=dataset_root,
                    annotated_root=root / "annotated",
                    segment_archive_root=root / "segment_csv_archive",
                    hdf5_archive_path=None,
                    dry_run=True,
                )

            self.assertEqual(target.read_bytes(), target_bytes)
            self.assertEqual(sibling.read_bytes(), sibling_bytes)
            self.assertEqual(jumplist.read_bytes(), original_jumplist)
            self.assertFalse((dataset_root / "archive").exists())

    def test_dry_run_refuses_repeated_source_path_even_with_different_sensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "annotated" / "total"
            pending_root = root / "pending"
            dataset_root.mkdir(parents=True)
            pending_root.mkdir()
            jumplist = dataset_root / "jumplist.csv"
            original_jumplist = b"path,type,success,skater\n"
            jumplist.write_bytes(original_jumplist)
            requested_segment = Path("data/pending/segments/session/sensor1/jump001.csv")
            annotation_file = pending_root / "session_for_annotation.csv"
            rows = [
                {
                    "path": requested_segment.as_posix(),
                    "type": 1,
                    "success": 1,
                    "athlete_id": "maksim",
                    "annotation_status": "annotated",
                    "sensor_id": sensor_id,
                    "session_key": "session",
                }
                for sensor_id in ("1", "2")
            ]
            pd.DataFrame(rows).to_csv(annotation_file, index=False)
            annotation_bytes = annotation_file.read_bytes()

            with self.assertRaisesRegex(ValueError, "duplicate source segment paths"):
                finalize_annotation_file(
                    annotation_file,
                    dataset_path=dataset_root,
                    annotated_root=root / "annotated",
                    segment_archive_root=root / "segment_csv_archive",
                    hdf5_archive_path=None,
                    dry_run=True,
                )

            self.assertEqual(annotation_file.read_bytes(), annotation_bytes)
            self.assertEqual(jumplist.read_bytes(), original_jumplist)
            self.assertFalse((dataset_root / "archive").exists())

    def test_hdf5_only_segment_is_refused_with_actionable_message_before_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "annotated" / "total"
            pending_root = root / "pending"
            dataset_root.mkdir(parents=True)
            pending_root.mkdir()
            jumplist = dataset_root / "jumplist.csv"
            original_jumplist = b"path,type,success,skater\n"
            jumplist.write_bytes(original_jumplist)
            requested_segment = Path("unavailable/segments/session/sensor1/jump001.csv")
            hdf5_archive = root / "synergie_archive.h5"
            with h5py.File(hdf5_archive, "w") as handle:
                handle.attrs["segment_path_index_json"] = json.dumps({requested_segment.as_posix(): "/segment"})
            annotation_file = pending_root / "session_for_annotation.csv"
            annotation_bytes = _write_annotation(annotation_file, requested_segment)

            with self.assertRaisesRegex(FileNotFoundError, "available only in the HDF5 archive.*Restore or re-export"):
                finalize_annotation_file(
                    annotation_file,
                    dataset_path=dataset_root,
                    annotated_root=root / "annotated",
                    segment_archive_root=root / "segment_csv_archive",
                    hdf5_archive_path=hdf5_archive,
                    dry_run=True,
                )

            self.assertEqual(annotation_file.read_bytes(), annotation_bytes)
            self.assertEqual(jumplist.read_bytes(), original_jumplist)
            self.assertFalse((dataset_root / "archive").exists())

    def test_existing_training_duplicates_are_rejected_before_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "annotated" / "total"
            pending_root = root / "pending"
            archive_root = root / "segment_csv_archive"
            dataset_root.mkdir(parents=True)
            pending_root.mkdir()
            jumplist = dataset_root / "jumplist.csv"
            original_jumplist = b"path,type,success,skater\nexisting.csv,0,1,a\nexisting.csv,0,1,a\n"
            jumplist.write_bytes(original_jumplist)
            requested_segment = Path("data/pending/segments/session/sensor1/jump001.csv")
            archived_segment = archive_root / requested_segment
            archived_segment.parent.mkdir(parents=True)
            archived_segment.write_text("ms,Gyr_X\n0,1\n", encoding="utf-8")
            annotation_file = pending_root / "session_for_annotation.csv"
            annotation_bytes = _write_annotation(annotation_file, requested_segment)

            with self.assertRaisesRegex(ValueError, "existing training jumplist already contains duplicate paths"):
                finalize_annotation_file(
                    annotation_file,
                    dataset_path=dataset_root,
                    annotated_root=root / "annotated",
                    segment_archive_root=archive_root,
                    hdf5_archive_path=None,
                    dry_run=True,
                )

            self.assertEqual(annotation_file.read_bytes(), annotation_bytes)
            self.assertEqual(jumplist.read_bytes(), original_jumplist)
            self.assertTrue(archived_segment.exists())
            self.assertFalse((dataset_root / "archive").exists())


def _write_annotation(path: Path, segment_path: Path, *, session_key: str = "session", sensor_id: str = "1") -> bytes:
    pd.DataFrame(
        [
            {
                "path": segment_path.as_posix(),
                "type": 1,
                "success": 1,
                "athlete_id": "maksim",
                "annotation_status": "annotated",
                "sensor_id": sensor_id,
                "session_key": session_key,
            }
        ]
    ).to_csv(path, index=False)
    return path.read_bytes()
