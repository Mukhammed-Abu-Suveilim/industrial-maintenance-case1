"""Синтетические проверки моделей и ограничений precision."""

import pickle
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import FixedThresholdClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_maintenance.baseline import (  # noqa: E402
    classification_metrics,
    precision_recall_operating_point,
    prediction_table,
    select_failure_threshold,
)
from industrial_maintenance.ensembles import (  # noqa: E402
    build_catboost_classifier,
    build_forest_pipeline,
)


class EnsembleTests(unittest.TestCase):
    def test_forest_prediction_threads_change_without_refitting(self):
        # 1) Сохранить деревья небольшого синтетического леса.
        frame = pd.DataFrame({"operating_hours": [0, 1, 2, 3]})
        model = build_forest_pipeline(
            ["operating_hours"],
            [],
            {"n_estimators": 8, "random_state": 42, "n_jobs": 1},
        ).fit(frame, [0, 0, 1, 1])
        trees = pickle.dumps(model.named_steps["classifier"].estimators_)

        # 2) Изменить только режим применения без повторного fit.
        with patch.object(model, "fit", side_effect=AssertionError):
            model.set_params(classifier__n_jobs=6)
            model.set_params(classifier__n_jobs=1)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(
            model.predict_proba(frame), restored.predict_proba(frame)
        )
        self.assertEqual(
            trees, pickle.dumps(model.named_steps["classifier"].estimators_)
        )

    def test_forbidden_and_misplaced_fields_are_rejected(self):
        for builder in [build_forest_pipeline, build_catboost_classifier]:
            for field in [
                "id",
                "severity",
                "required_route",
                "operator_log",
                "fault_type",
                "failure_within_30d",
                "unknown",
            ]:
                for numeric, categorical in [
                    ([field], []),
                    (["operating_hours"], [field]),
                ]:
                    with self.subTest(builder=builder, field=field):
                        with self.assertRaises(ValueError):
                            builder(numeric, categorical, {})
            with self.assertRaises(ValueError):
                builder(["operating_hours", "operating_hours"], [], {})
            with self.assertRaises(ValueError):
                builder([], [], {})

    def test_train_only_encoding_unknown_categories_and_serialization(self):
        # 1) Задать маленькие train и validation с новой категорией.
        train = pd.DataFrame(
            {
                "operating_hours": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "equipment": ["a", "b", "c"] * 3,
            }
        )
        validation = pd.DataFrame(
            {
                "operating_hours": [1000, -1000, 2],
                "equipment": ["unseen", "a", "b"],
            }
        )
        ids = ["synthetic_c", "synthetic_a", "synthetic_b"]
        for target, labels, truth in [
            ("failure_within_30d", [0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 0, 1]),
            (
                "fault_type",
                ["normal", "seal_leak", "overheat"] * 3,
                ["seal_leak", "normal", "overheat"],
            ),
        ]:
            models = [
                build_forest_pipeline(
                    ["operating_hours"],
                    ["equipment"],
                    {"n_estimators": 8, "random_state": 42},
                ),
                build_catboost_classifier(
                    ["operating_hours"],
                    ["equipment"],
                    {
                        "iterations": 8,
                        "depth": 2,
                        "random_seed": 42,
                        "thread_count": 1,
                        "allow_writing_files": False,
                        "use_best_model": False,
                        "verbose": False,
                        "loss_function": "Logloss"
                        if target == "failure_within_30d"
                        else "MultiClass",
                    },
                ),
            ]
            for model in models:
                # 2) Обучить на train и сохранить состояние.
                model.fit(train, labels)
                if hasattr(model, "named_steps"):
                    encoder = model.named_steps[
                        "preprocessing"
                    ].named_transformers_["categorical"]
                    self.assertEqual(
                        encoder.categories_[0].tolist(), ["a", "b", "c"]
                    )
                    state_before = pickle.dumps(encoder)
                    preprocessing = model.named_steps["preprocessing"]
                    numeric = preprocessing.transform(validation)[:, 0]
                    np.testing.assert_array_equal(numeric, [1000, -1000, 2])
                else:
                    state_before = pickle.dumps(model)
                probabilities = model.predict_proba(validation)
                restored = pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(
                    restored.predict_proba(validation), probabilities
                )
                if hasattr(model, "named_steps"):
                    self.assertEqual(state_before, pickle.dumps(encoder))
                else:
                    self.assertEqual(state_before, pickle.dumps(model))
                    self.assertEqual(model.get_cat_feature_indices(), [1])

                # 3) Связать вероятности с id и именами классов.
                classes = model.classes_.tolist()
                predictions = prediction_table(
                    ids, truth, probabilities, classes, target, 0.5
                )
                self.assertEqual(predictions["id"].tolist(), ids)
                for index, label in enumerate(classes):
                    np.testing.assert_array_equal(
                        predictions[f"probability_{label}"],
                        probabilities[:, index],
                    )
                if target == "fault_type":
                    np.testing.assert_array_equal(
                        predictions["y_pred"],
                        np.asarray(classes)[probabilities.argmax(axis=1)],
                    )
                    continue

                # 4) Проверить пороговый объект без повторного fit.
                best, _ = select_failure_threshold(
                    truth, predictions["probability_1"]
                )
                with patch.object(restored, "fit", side_effect=AssertionError):
                    wrapper = FixedThresholdClassifier(
                        restored,
                        threshold=best["threshold"],
                        pos_label=1,
                        response_method="predict_proba",
                    )
                    expected = wrapper.predict(validation)
                loaded = pickle.loads(pickle.dumps(wrapper))
                np.testing.assert_array_equal(
                    loaded.predict(validation), expected
                )
                np.testing.assert_array_equal(
                    loaded.predict_proba(validation), probabilities
                )
                tuned = prediction_table(
                    ids,
                    truth,
                    probabilities,
                    classes,
                    target,
                    best["threshold"],
                )
                np.testing.assert_array_equal(tuned["y_pred"], expected)
                old_metrics = classification_metrics(
                    predictions, classes, target
                )
                new_metrics = classification_metrics(tuned, classes, target)
                for metric in ["average_precision", "roc_auc"]:
                    self.assertEqual(old_metrics[metric], new_metrics[metric])


class PrecisionConstraintTests(unittest.TestCase):
    def test_repeated_probabilities_and_boundary_are_kept_together(self):
        # 1) Два отказа имеют одну вероятность на границе.
        _, curve = select_failure_threshold([0, 1, 1, 0], [0.1, 0.4, 0.4, 0.8])
        point = precision_recall_operating_point(curve, 0.6)
        self.assertEqual(point["status"], "attainable")
        self.assertEqual(point["threshold"], 0.4)
        self.assertEqual(point["tp"], 2)
        self.assertEqual(point["recall"], 1)
        self.assertAlmostEqual(point["precision"], 2 / 3)

    def test_unattainable_constraint_returns_empty_metrics(self):
        _, curve = select_failure_threshold([0, 1], [0.8, 0.2])
        point = precision_recall_operating_point(curve, 0.8)
        self.assertEqual(point["status"], "unattainable")
        self.assertIsNone(point["threshold"])
        self.assertIsNone(point["recall"])

    def test_empty_positive_predictions_are_excluded_even_at_zero_floor(self):
        curve = pd.DataFrame(
            {
                "threshold": [1.0],
                "precision": [1.0],
                "recall": [0.0],
                "tp": [0],
                "fp": [0],
            }
        )
        self.assertEqual(
            precision_recall_operating_point(curve, 0)["status"],
            "unattainable",
        )

    def test_ties_prefer_precision_then_higher_threshold(self):
        # 1) При равной полноте сначала выбрать большую precision.
        curve = pd.DataFrame(
            {
                "threshold": [0.1, 0.3, 0.4],
                "precision": [0.7, 0.8, 0.8],
                "recall": [0.9, 0.9 - 1e-14, 0.9 - 1e-14],
                "f1": [0.8] * 3,
                "f2": [0.8] * 3,
                "tp": [9] * 3,
                "fp": [3, 2, 2],
                "tn": [7, 8, 8],
                "fn": [1] * 3,
            }
        )
        before = curve.copy(deep=True)
        point = precision_recall_operating_point(curve, 0.7)
        self.assertEqual(point["threshold"], 0.4)
        pd.testing.assert_frame_equal(curve, before)

    def test_invalid_precision_floor_is_rejected(self):
        for floor in [-0.1, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                precision_recall_operating_point(pd.DataFrame(), floor)


if __name__ == "__main__":
    unittest.main()
