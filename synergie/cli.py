from __future__ import annotations

import argparse
from typing import Sequence

from synergie import operations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="CLI de traitement, entrainement et maintenance du projet Synergie.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-sessions", help="Lister les sessions connues dans constants.py.")

    session_parser = subparsers.add_parser("show-session", help="Afficher les metadonnees d'une session.")
    session_parser.add_argument("session")

    files_parser = subparsers.add_parser("list-session-files", help="Lister les CSV d'une session.")
    files_parser.add_argument("session")
    files_parser.add_argument("--raw-root", default="data/raw")

    train_parser = subparsers.add_parser("train", help="Entrainer un modele.")
    train_parser.add_argument("task", choices=["type", "success"])
    train_parser.add_argument("--dataset", default="data/annotated/total")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--architecture")

    benchmark_parser = subparsers.add_parser("benchmark", help="Lancer un benchmark experimental sur le dataset annote.")
    benchmark_parser.add_argument("task", choices=["type", "success"])
    benchmark_parser.add_argument("--dataset", default="data/annotated/total")
    benchmark_parser.add_argument("--model", default="minirocket", choices=["minirocket", "hydra", "summary"])
    benchmark_parser.add_argument("--test-size", type=float, default=0.2)
    benchmark_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = subparsers.add_parser("process-file", help="Predire les sauts pour un fichier CSV.")
    predict_parser.add_argument("csv_path")
    predict_parser.add_argument("--synchro", type=int, default=0)
    predict_parser.add_argument("--output")

    repredict_parser = subparsers.add_parser("repredict", help="Recalculer les predictions des enregistrements bruts.")
    repredict_parser.add_argument("--raw-root", default="data/raw")

    parser.add_argument("-t", "--legacy-train", choices=["type", "success"], help=argparse.SUPPRESS)
    parser.add_argument("-repredict", action="store_true", dest="legacy_repredict", help=argparse.SUPPRESS)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.command is not None:
        return args

    if args.legacy_train:
        args.command = "train"
        args.task = args.legacy_train
        args.dataset = "data/annotated/total"
        args.epochs = None
        return args

    if args.legacy_repredict:
        args.command = "repredict"
        args.raw_root = "data/raw"
        return args

    return args


def run_command(args: argparse.Namespace) -> int:
    if args.command == "list-sessions":
        for session in operations.list_sessions():
            metadata = operations.session_metadata(session)
            print(f"{session}: {metadata['path']} (synchro={metadata['sample_time_fine_synchro']})")
        return 0

    if args.command == "show-session":
        metadata = operations.session_metadata(args.session)
        print(f"session={args.session}")
        print(f"path={metadata['path']}")
        print(f"synchro={metadata['sample_time_fine_synchro']}")
        return 0

    if args.command == "list-session-files":
        for path in operations.list_session_csv_files(args.session, args.raw_root):
            print(path)
        return 0

    if args.command == "train":
        operations.train_model(args.task, args.dataset, args.epochs, args.architecture)
        return 0

    if args.command == "process-file":
        result = operations.process_csv_file(args.csv_path, args.synchro, args.output)
        print(result["path"])
        return 0

    if args.command == "benchmark":
        try:
            from synergie.experiments.benchmark import benchmark_summary, run_hydra_benchmark, run_minirocket_benchmark
        except ModuleNotFoundError as exc:
            missing = exc.name or "scientific dependencies"
            raise RuntimeError(
                f"The benchmark command requires the scientific stack in the project environment. "
                f"Missing module: {missing}. Activate the 'synergie-data' environment first."
            ) from exc

        if args.model == "summary":
            print(benchmark_summary(args.task, args.dataset))
            return 0

        if args.model == "minirocket":
            try:
                result = run_minirocket_benchmark(
                    args.task,
                    args.dataset,
                    test_size=args.test_size,
                    random_state=args.random_state,
                )
            except ModuleNotFoundError as exc:
                missing = exc.name or "benchmark dependency"
                raise RuntimeError(
                    f"The '{args.model}' benchmark is not available in the current environment. "
                    f"Missing module: {missing}. Recreate or update the 'synergie-data' environment first."
                ) from exc
        else:
            try:
                result = run_hydra_benchmark(
                    args.task,
                    args.dataset,
                    test_size=args.test_size,
                    random_state=args.random_state,
                )
            except ModuleNotFoundError as exc:
                missing = exc.name or "benchmark dependency"
                raise RuntimeError(
                    f"The '{args.model}' benchmark is not available in the current environment. "
                    f"Missing module: {missing}. Recreate or update the 'synergie-data' environment first."
                ) from exc
        print(f"model={result.model_name}")
        print(f"task={result.task}")
        print(f"accuracy={result.accuracy:.4f}")
        print(f"train_samples={result.train_samples}")
        print(f"test_samples={result.test_samples}")
        print(result.report)
        return 0

    if args.command == "repredict":
        processed = operations.repredict_raw_trainings(args.raw_root)
        print(f"{processed} training files processed.")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv))
    if args.command is None:
        parser.print_help()
        return 1
    return run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
