"""Unit tests verifying that all foundation modules and subpackages import cleanly."""

import importlib
import unittest


class TestModuleImports(unittest.TestCase):
    """Verify that every package and module in src can be imported without errors."""

    def test_import_src(self) -> None:
        module = importlib.import_module("src")
        self.assertTrue(hasattr(module, "__version__"))

    def test_import_data_loader(self) -> None:
        module = importlib.import_module("src.data.loader")
        self.assertTrue(callable(getattr(module, "get_project_root")))
        self.assertTrue(callable(getattr(module, "load_dataset_config")))
        self.assertTrue(callable(getattr(module, "load_dataset")))

    def test_import_meta_features(self) -> None:
        module = importlib.import_module("src.features.meta_features")
        self.assertTrue(callable(getattr(module, "extract_meta_features")))

    def test_import_hpo_optimizer(self) -> None:
        module = importlib.import_module("src.hpo.optimizer")
        self.assertTrue(hasattr(module, "HPOptimizer"))

    def test_import_meta_model(self) -> None:
        module = importlib.import_module("src.meta_learning.meta_model")
        self.assertTrue(hasattr(module, "MetaModel"))

    def test_import_confidence_estimator(self) -> None:
        module = importlib.import_module("src.confidence.estimator")
        self.assertTrue(hasattr(module, "ConfidenceEstimator"))

    def test_import_evaluation_metrics(self) -> None:
        module = importlib.import_module("src.evaluation.metrics")
        self.assertTrue(callable(getattr(module, "compute_simple_regret")))
        self.assertTrue(callable(getattr(module, "compute_average_rank")))

    def test_import_logging_utils(self) -> None:
        module = importlib.import_module("src.utils.logging_utils")
        self.assertTrue(callable(getattr(module, "get_logger")))
        logger = module.get_logger("test_logger")
        self.assertIsNotNone(logger)


if __name__ == "__main__":
    unittest.main()
