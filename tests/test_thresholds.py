"""Синтетические проверки F2 и сохранённого порогового объекта."""

import pickle
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import FixedThresholdClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_maintenance.baseline import (  # noqa: E402
    build_logistic_pipeline,
    classification_metrics,
    prediction_table,
    select_failure_threshold,
)
from industrial_maintenance.comparison import (  # noqa: E402
    paired_group_bootstrap,
)


class ThresholdTests(unittest.TestCase):
    def test_f2_matches_hand_calculation(self):
        # 1) Задать TP=1, FP=1, FN=2 и проверить ручную формулу.
        table = prediction_table(
            ["a", "b", "c", "d", "e"],
            [1, 1, 1, 0, 0],
            np.array(
                [[0.2, 0.8], [0.8, 0.2], [0.9, 0.1], [0.3, 0.7], [0.9, 0.1]]
            ),
            [0, 1],
            "failure_within_30d",
            0.5,
        )
        metrics = classification_metrics(table, [0, 1], "failure_within_30d")
        self.assertAlmostEqual(metrics["f2"], 5 / 14)

    def test_optimum_outside_half_includes_equal_probability(self):
        # 1) Второй отказ ниже 0,5: его включение повышает F2 до 1.
        best, curve = select_failure_threshold([0, 1, 1], [0.1, 0.3, 0.9])
        self.assertEqual(best["threshold"], 0.3)
        self.assertEqual(best["f2"], 1)
        self.assertEqual(best["tp"], 2)
        self.assertEqual(best["fn"], 0)
        self.assertEqual(curve["selected"].sum(), 1)
        self.assertEqual(
            curve["threshold"].tolist(), [0, 0.1, 0.3, 0.5, 0.9, 1]
        )

    def test_repeated_probabilities_extremes_and_highest_threshold_tie(self):
        # 1) Порог 1 включает обе вероятности 1 и выигрывает у 0,5.
        best, curve = select_failure_threshold([0, 1, 1, 0], [0, 1, 1, 0])
        self.assertEqual(best["threshold"], 1)
        self.assertEqual(best["f2"], 1)
        self.assertEqual(curve["threshold"].tolist(), [0, 0.5, 1])

        # 2) Постоянная вероятность даёт выбор её значения, а не нуля.
        best, _ = select_failure_threshold([0, 1, 0], [0.2, 0.2, 0.2])
        self.assertEqual(best["threshold"], 0.2)
        self.assertEqual(
            (best["tp"], best["fp"], best["tn"], best["fn"]), (1, 2, 0, 0)
        )

    def test_f2_tie_prefers_precision(self):
        # 1) При P=2 оба решения дают F2=5/9, но разную precision.
        truth = [1] + [0] * 8 + [1]
        probabilities = [0.9] + [0.2] * 8 + [0.1]
        best, curve = select_failure_threshold(truth, probabilities)
        low = curve.loc[curve["threshold"].eq(0.1)].iloc[0]
        self.assertAlmostEqual(low["f2"], best["f2"])
        self.assertEqual(best["threshold"], 0.9)
        self.assertEqual(best["precision"], 1)

    def test_invalid_probabilities_and_shapes_are_rejected(self):
        for probabilities in [
            [np.nan, 0.2],
            [np.inf, 0.2],
            [-0.1, 0.2],
            [1.1, 0.2],
            [0.2],
            [[0.2], [0.3]],
        ]:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    select_failure_threshold([0, 1], probabilities)

    def test_missing_or_invalid_classes_are_rejected(self):
        for truth in [[0, 0], [1, 1], [], [0, 2]]:
            with self.subTest(truth=truth):
                with self.assertRaisesRegex(ValueError, "оба класса"):
                    select_failure_threshold(truth, [0.3] * len(truth))

    def test_saved_wrapper_reuses_fit_and_preserves_probabilities_and_ids(
        self,
    ):
        # 1) Обучить единственный Pipeline на синтетических показаниях.
        features = pd.DataFrame({"operating_hours": [0, 1, 2, 3, 4, 5]})
        truth = np.array([0, 0, 1, 0, 1, 1])
        ids = ["s5", "s2", "s4", "s1", "s3", "s0"]
        pipeline = build_logistic_pipeline(
            ["operating_hours"], [], {"random_state": 42, "max_iter": 1000}
        ).fit(features, truth)
        probabilities = pipeline.predict_proba(features)
        best, _ = select_failure_threshold(truth, probabilities[:, 1])

        # 2) Запретить fit; сериализовать обёртку уже обученной модели.
        with patch.object(pipeline, "fit", side_effect=AssertionError):
            wrapper = FixedThresholdClassifier(
                pipeline,
                threshold=best["threshold"],
                pos_label=1,
                response_method="predict_proba",
            )
            np.testing.assert_array_equal(
                wrapper.predict(features),
                (probabilities[:, 1] >= best["threshold"]).astype(int),
            )
        restored = pickle.loads(pickle.dumps(wrapper))
        np.testing.assert_allclose(
            restored.predict_proba(features), probabilities
        )
        np.testing.assert_array_equal(
            restored.predict(features), wrapper.predict(features)
        )

        # 3) Проверить id и неизменность метрик вероятностей.
        default = prediction_table(
            ids, truth, probabilities, [0, 1], "failure_within_30d", 0.5
        )
        tuned = prediction_table(
            ids,
            truth,
            restored.predict_proba(features),
            [0, 1],
            "failure_within_30d",
            best["threshold"],
        )
        self.assertEqual(tuned["id"].tolist(), ids)
        np.testing.assert_array_equal(
            tuned["y_pred"], restored.predict(features)
        )
        old_metrics = classification_metrics(
            default, [0, 1], "failure_within_30d"
        )
        new_metrics = classification_metrics(
            tuned, [0, 1], "failure_within_30d"
        )
        for metric in ["average_precision", "roc_auc"]:
            self.assertEqual(old_metrics[metric], new_metrics[metric])


class ThresholdBootstrapTests(unittest.TestCase):
    def test_f2_uses_fixed_labels_and_whole_paired_groups(self):
        # 1) Сопоставить два фиксированных решения без изменения порога.
        reference = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "y_true": [0, 1, 0, 1],
                "y_pred": [0, 0, 0, 1],
            }
        )
        candidate = reference.assign(y_pred=[0, 1, 1, 1])
        groups = pd.DataFrame(
            {"id": ["a", "b", "c", "d"], "group_id": [1, 1, 2, 2]}
        )
        generator = Mock()
        generator.integers.return_value = np.array([0, 0])
        with (
            patch(
                "industrial_maintenance.comparison.np.random.default_rng",
                return_value=generator,
            ),
            patch(
                "industrial_maintenance.comparison.fbeta_score",
                wraps=fbeta_score,
            ) as metric,
        ):
            result = paired_group_bootstrap(
                reference.iloc[::-1],
                candidate,
                groups,
                [0, 1],
                "failure_within_30d",
                n_bootstrap=1,
                metric="f2",
            )
        self.assertAlmostEqual(result["difference"], 10 / 11 - 5 / 9)
        np.testing.assert_array_equal(
            metric.call_args_list[2].args[0], [0, 1, 0, 1]
        )
        np.testing.assert_array_equal(
            metric.call_args_list[2].args[1], [0, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            metric.call_args_list[3].args[1], [0, 1, 0, 1]
        )
        self.assertEqual(result["lower_95"], 1)
        self.assertEqual(result["valid_repeats"], 1)


if __name__ == "__main__":
    unittest.main()
