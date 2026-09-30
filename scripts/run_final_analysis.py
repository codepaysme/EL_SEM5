"""Executable script to run Phase 11 statistical and empirical analysis pipeline."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evaluation.final_analysis import run_full_analysis


def main():
    print("Starting Phase 11 Final Statistical & Results Analysis...")
    results = run_full_analysis()
    print("\nPhase 11 Analysis completed successfully!")
    print(f"Artifacts saved in: results/analysis/")
    print(f"Report figures saved in: results/analysis/figures/ ({len(results['figure_paths'])} figures)")
    print(f"JSON analysis metadata: {results['json_path']}")
    print(f"Markdown written report: {results['markdown_path']}")


if __name__ == "__main__":
    main()
