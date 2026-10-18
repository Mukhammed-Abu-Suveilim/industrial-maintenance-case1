"""Синтетические проверки предобработки и метрик базового решения."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# src ещё не устанавливается как пакет; путь ограничен этим проектом.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_maintenance.baseline import (  # noqa: E402
    build_logistic_pipeline,
    classification_metrics,
    prediction_table,
)


class PreprocessingTests(unittest.TestCase):
    def test_validation_does_not_change_scaler_or_categories(self):
        # 1) Обучить преобразования на малом синтетическом train.
        train = pd.DataFrame(
            {
                "operating_hours": [0.0, 2.0, 4.0, 6.0],
                "equipment": ["a", "b", "a", "b"],
            }
        )
        pipeline = build_logistic_pipeline(
            ["operating_hours"], ["equipment"], {"max_iter": 1000}
        )
        pipeline.fit(train, [0, 0, 1, 1])
        preprocessing = pipeline.named_steps["preprocessing"]
        scaler = preprocessing.named_transformers_["numeric"]
        encoder = preprocessing.named_transformers_["categorical"]

        # 2) Применить к сильно иному значению и неизвестной категории.
        validation = pd.DataFrame(
            {"operating_hours": [1000.0], "equipment": ["unseen"]}
        )
        probabilities = pipeline.predict_proba(validation)

        # 3) Убедиться, что статистика train не изменилась.
        np.testing.assert_allclose(scaler.mean_, [3.0])
        self.assertEqual(encoder.categories_[0].tolist(), ["a", "b"])
        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0])

    def test_targets_and_identifiers_are_rejected_as_features(self):
        for field in ["id", "fault_type", "failure_within_30d"]:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Цели"):
                    build_logistic_pipeline([field], [], {})

    def test_excluded_fields_cannot_enter_either_feature_group(self):
        for field in ["severity", "required_route"]:
            for numeric, categorical in [
                ([field], []),
                (["operating_hours"], [field]),
            ]:
                with self.subTest(field=field, categorical=categorical):
                    with self.assertRaisesRegex(ValueError, field):
                        build_logistic_pipeline(numeric, categorical, {})


class PredictionTests(unittest.TestCase):
    def test_binary_metrics_use_class_one_even_in_reversed_class_order(self):
        # 1) Явно передать порядок классов 1, 0 и нестандартные индексы.
        ids = pd.Series(["synthetic_b", "synthetic_a"], index=[20, 10])
        truth = pd.Series([1, 0], index=[20, 10])
        predictions = prediction_table(
            ids,
            truth,
            np.array([[0.8, 0.2], [0.1, 0.9]]),
            [1, 0],
            "failure_within_30d",
            0.5,
        )

        # 2) Проверить позиционную связь id, меток и вероятностей.
        self.assertEqual(predictions["id"].tolist(), ids.tolist())
        self.assertEqual(predictions["y_pred"].tolist(), [1, 0])
        np.testing.assert_allclose(predictions["probability_1"], [0.8, 0.1])

        # 3) Сопоставить метрики с точным ручным результатом.
        metrics = classification_metrics(
            predictions, [1, 0], "failure_within_30d"
        )
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["log_loss"], -np.log([0.8, 0.9]).mean())

    def test_multiclass_probabilities_keep_their_class_names(self):
        # 1) Предсказать три синтетических класса в заданном порядке.
        classes = ["c", "a", "b"]
        probabilities = np.array(
            [
                [0.1, 0.8, 0.1],
                [0.7, 0.1, 0.2],
                [0.05, 0.05, 0.9],
            ]
        )
        predictions = prediction_table(
            ["1", "2", "3"],
            ["a", "c", "b"],
            probabilities,
            classes,
            "fault_type",
            0.5,
        )

        # 2) Проверить метки и согласование классов при расчёте потерь.
        self.assertEqual(predictions["y_pred"].tolist(), ["a", "c", "b"])
        metrics = classification_metrics(predictions, classes, "fault_type")
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(
            metrics["log_loss"], -np.log([0.8, 0.7, 0.9]).mean()
        )

    def test_invalid_probabilities_and_duplicate_ids_are_rejected(self):
        # 1) Отвергнуть строку вероятностей с неверной суммой.
        with self.assertRaisesRegex(ValueError, "сумму 1"):
            prediction_table(
                ["1"],
                [0],
                np.array([[0.5, 0.6]]),
                [0, 1],
                "failure_within_30d",
                0.5,
            )
        # 2) Отвергнуть повторение идентификатора в исходных прогнозах.
        with self.assertRaisesRegex(ValueError, "id повторяются"):
            prediction_table(
                ["1", "1"],
                [0, 1],
                np.array([[0.8, 0.2], [0.1, 0.9]]),
                [0, 1],
                "failure_within_30d",
                0.5,
            )

    def test_single_negative_class_has_no_roc_auc_or_average_precision(self):
        # 1) Подготовить подгруппу без положительных примеров.
        predictions = prediction_table(
            ["1", "2"],
            [0, 0],
            np.array([[0.8, 0.2], [0.9, 0.1]]),
            [0, 1],
            "failure_within_30d",
            0.5,
        )
        # 2) Рассчитать метрики с явным отсутствием определённой оценки.
        metrics = classification_metrics(
            predictions, [0, 1], "failure_within_30d"
        )
        # 3) Сохранить неопределённость вместо выдуманного нулевого AUC.
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["average_precision"])


if __name__ == "__main__":
    unittest.main()
