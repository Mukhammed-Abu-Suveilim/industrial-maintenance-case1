"""Проверки соответствия объектов и различия AP/PR-AUC."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from industrial_maintenance.baseline import build_text_pipeline
from industrial_maintenance.experiment_review import (
    align_predictions,
    load_cpu_predictor,
    review_metrics,
    subgroup_review,
)


class ExperimentReviewTests(unittest.TestCase):
    def test_saved_tfidf_receives_texts_instead_of_column_names(self):
        """Адаптер должен возвращать вероятность для каждой строки."""
        # 1) Сохранить небольшой синтетический текстовый Pipeline.
        features = pd.DataFrame(
            {
                "operator_log": [
                    "stable pump",
                    "stable motor",
                    "leak pump",
                    "leak motor",
                ],
            }
        )
        model = build_text_pipeline({}, {"max_iter": 100})
        model.fit(features["operator_log"], [0, 0, 1, 1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            joblib.dump(model, root / "model.pkl")

            # 2) Сравнить все строки с прямым применением модели.
            predict, classes = load_cpu_predictor(
                root,
                {
                    "family": "logistic",
                    "run_path": ".",
                    "features": ["operator_log"],
                },
                threads=1,
            )
            np.testing.assert_allclose(
                predict(features),
                model.predict_proba(features["operator_log"]),
            )
            self.assertEqual(classes, [0, 1])

    def setUp(self):
        self.data = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "failure_within_30d": [0, 1, 1],
                "equipment": ["pump", "pump", "motor"],
                "severity": ["low", "high", "high"],
            }
        )
        self.predictions = pd.DataFrame(
            {
                "id": ["c", "a", "b"],
                "y_true": [1, 0, 1],
                "y_pred": [0, 1, 1],
                "probability_0": [0.8, 0.2, 0.3],
                "probability_1": [0.2, 0.8, 0.7],
            }
        )

    def test_alignment_rejects_wrong_labels_and_unknown_ids(self):
        aligned = align_predictions(
            self.predictions, self.data, "failure_within_30d", [0, 1]
        )
        self.assertEqual(aligned["id"].tolist(), ["a", "b", "c"])
        for column, value in [("y_true", 0), ("id", "unknown")]:
            changed = self.predictions.copy()
            changed.loc[0, column] = value
            with self.assertRaises(ValueError):
                align_predictions(
                    changed, self.data, "failure_within_30d", [0, 1]
                )

    def test_pr_auc_is_not_silently_renamed_ap(self):
        metrics = review_metrics(
            self.predictions, [0, 1], "failure_within_30d"
        )
        self.assertAlmostEqual(metrics["average_precision"], 7 / 12)
        self.assertAlmostEqual(metrics["pr_auc_trapezoid"], 5 / 12)

    def test_subgroups_preserve_ids_and_undefined_auc(self):
        groups = subgroup_review(
            self.predictions, self.data, [0, 1], "failure_within_30d"
        )
        self.assertEqual(groups["records"].sum(), 6)
        motor = groups.loc[groups["value"].eq("motor")].iloc[0]
        self.assertTrue(pd.isna(motor["pr_auc_trapezoid"]))


if __name__ == "__main__":
    unittest.main()
