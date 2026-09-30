"""Unit tests verifying dataset configuration integrity."""

import re
import unittest
from pathlib import Path

EXPECTED_TRAIN_DATASETS = {
    "adult",
    "bank-marketing",
    "credit-g",
    "wdbc",
    "spambase",
    "vehicle",
    "phoneme",
    "blood-transfusion-service-center",
}

EXPECTED_UNSEEN_DATASETS = {
    "madelon",
    "diabetes",
    "steel-plates-fault",
    "ilpd",
}

ALL_EXPECTED_DATASETS = EXPECTED_TRAIN_DATASETS | EXPECTED_UNSEEN_DATASETS


class TestDatasetConfig(unittest.TestCase):
    """Verify config/datasets.yaml structure and contents."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = Path(__file__).resolve().parent.parent / "config" / "datasets.yaml"

    def test_config_file_exists(self) -> None:
        self.assertTrue(self.config_path.is_file(), f"datasets.yaml not found at {self.config_path}")

    def test_datasets_content(self) -> None:
        """Verify the 12 planned datasets and split classifications."""
        try:
            import yaml

            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            datasets = data.get("datasets", [])
            self.assertEqual(len(datasets), 12, "datasets.yaml should contain exactly 12 datasets.")

            train_found = {d["dataset_name"] for d in datasets if d.get("split") == "train_meta"}
            unseen_found = {d["dataset_name"] for d in datasets if d.get("split") == "unseen_test"}

            self.assertEqual(train_found, EXPECTED_TRAIN_DATASETS)
            self.assertEqual(unseen_found, EXPECTED_UNSEEN_DATASETS)

            for d in datasets:
                self.assertEqual(d.get("source"), "openml")

        except ImportError:
            # Fallback regex parser if PyYAML is not yet installed in current environment
            content = self.config_path.read_text(encoding="utf-8")
            names = set(re.findall(r"dataset_name:\s*([a-zA-Z0-9_\-]+)", content))
            self.assertEqual(
                names,
                ALL_EXPECTED_DATASETS,
                f"Missing or mismatched datasets in config: {ALL_EXPECTED_DATASETS - names}",
            )
            self.assertEqual(len(names), 12)


if __name__ == "__main__":
    unittest.main()
