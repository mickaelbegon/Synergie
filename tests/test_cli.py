import unittest

import constants
from synergie.cli import build_parser, normalize_args


class CliTests(unittest.TestCase):
    def test_sessions_is_a_dictionary(self):
        self.assertIsInstance(constants.sessions, dict)
        self.assertIn("1331", constants.sessions)

    def test_subcommand_train_parsing(self):
        parser = build_parser()
        args = normalize_args(parser.parse_args(["train", "type", "--epochs", "12"]))
        self.assertEqual(args.command, "train")
        self.assertEqual(args.task, "type")
        self.assertEqual(args.epochs, 12)

    def test_train_architecture_parsing(self):
        parser = build_parser()
        args = normalize_args(parser.parse_args(["train", "success", "--architecture", "tcn"]))
        self.assertEqual(args.command, "train")
        self.assertEqual(args.task, "success")
        self.assertEqual(args.architecture, "tcn")

    def test_show_session_parsing(self):
        parser = build_parser()
        args = normalize_args(parser.parse_args(["show-session", "1331"]))
        self.assertEqual(args.command, "show-session")
        self.assertEqual(args.session, "1331")

    def test_list_session_files_parsing(self):
        parser = build_parser()
        args = normalize_args(parser.parse_args(["list-session-files", "1331", "--raw-root", "tmp/raw"]))
        self.assertEqual(args.command, "list-session-files")
        self.assertEqual(args.session, "1331")
        self.assertEqual(args.raw_root, "tmp/raw")

    def test_benchmark_parsing(self):
        parser = build_parser()
        args = normalize_args(
            parser.parse_args(
                ["benchmark", "type", "--dataset", "data/annotated/total", "--model", "summary", "--test-size", "0.3"]
            )
        )
        self.assertEqual(args.command, "benchmark")
        self.assertEqual(args.task, "type")
        self.assertEqual(args.dataset, "data/annotated/total")
        self.assertEqual(args.model, "summary")
        self.assertEqual(args.test_size, 0.3)

    def test_legacy_train_flag_is_preserved(self):
        parser = build_parser()
        args = normalize_args(parser.parse_args(["-t", "success"]))
        self.assertEqual(args.command, "train")
        self.assertEqual(args.task, "success")

    def test_legacy_repredict_flag_is_preserved(self):
        parser = build_parser()
        args = normalize_args(parser.parse_args(["-repredict"]))
        self.assertEqual(args.command, "repredict")
        self.assertEqual(args.raw_root, "data/raw")


if __name__ == "__main__":
    unittest.main()
