"""CLI script to run Phase 10 unseen dataset benchmark."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.loader import UNSEEN_DATASETS
from src.evaluation.unseen_benchmark import run_unseen_benchmark



def main():
    parser = argparse.ArgumentParser(description="Run MetaHPO-C unseen dataset benchmark.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(UNSEEN_DATASETS),
        help="List of unseen datasets to evaluate.",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of trials per method per dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility random seed.")
    parser.add_argument("--beta", type=float, default=0.1, help="MetaHPO-C acquisition exploration weight.")
    args = parser.parse_args()

    print(f"Starting unseen dataset benchmark on: {args.datasets}")
    print(f"Trials per method: {args.trials}, Seed: {args.seed}, Beta: {args.beta}")

    run_unseen_benchmark(
        dataset_names=args.datasets,
        n_trials=args.trials,
        seed=args.seed,
        beta=args.beta,
    )
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
