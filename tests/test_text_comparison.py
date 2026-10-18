"""Синтетические проверки текстовой предобработки и парных сравнений."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_maintenance.baseline import (  # noqa: E402
    build_text_pipeline,
    prediction_table,
)
from industrial_maintenance.comparison import (  # noqa: E402
    paired_group_bootstrap,
)


class TextTests(unittest.TestCase):
    def test_vocabulary_and_idf_use_only_training_text(self):
        # 1) Обучить модель на двух русских синтетических классах.
        texts = pd.Series(
            ["насос стабилен", "насос стабилен", "насос шумит", "насос шумит"]
        )
        model = build_text_pipeline(
            {"min_df": 2, "sublinear_tf": True}, {"max_iter": 1000}
        )
        model.fit(texts, [0, 0, 1, 1])
        vectorizer = model.named_steps["tfidf"]
        vocabulary = dict(vectorizer.vocabulary_)
        idf = vectorizer.idf_.copy()
        self.assertEqual(set(vocabulary), {"насос", "стабилен", "шумит"})
        self.assertAlmostEqual(idf[vocabulary["шумит"]], np.log(5 / 3) + 1)

        # 2) Применить к неизвестным словам, сохранив id и вероятности.
        probabilities = model.predict_proba(pd.Series(["неизвестно", "насос"]))
        result = prediction_table(
            ["synthetic_b", "synthetic_a"],
            [1, 0],
            probabilities,
            model.classes_,
            "failure_within_30d",
            0.5,
        )
        self.assertEqual(result.id.tolist(), ["synthetic_b", "synthetic_a"])
        np.testing.assert_allclose(result.probability_1, probabilities[:, 1])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)

        # 3) Проверить отсутствие дообучения на validation.
        self.assertEqual(vectorizer.vocabulary_, vocabulary)
        np.testing.assert_array_equal(vectorizer.idf_, idf)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        # 1) Задать парные прогнозы двух синтетических моделей.
        self.reference = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "y_true": [0, 1, 0, 1],
                "y_pred": [1, 0, 1, 0],
                "probability_1": [0.9, 0.1, 0.8, 0.2],
            }
        )
        self.candidate = self.reference.assign(
            y_pred=[0, 1, 0, 1], probability_1=[0.1, 0.9, 0.2, 0.8]
        )

        # 2) Объединить связанные записи в две группы.
        self.groups = pd.DataFrame(
            {"id": ["a", "b", "c", "d"], "group_id": ["g1", "g1", "g2", "g2"]}
        )

    def test_pairing_is_independent_of_row_order(self):
        # 1) Рассчитать разницу при исходном порядке строк.
        expected = paired_group_bootstrap(
            self.reference,
            self.candidate,
            self.groups,
            [0, 1],
            "failure_within_30d",
            n_bootstrap=30,
        )

        # 2) Независимо переставить все таблицы и сопоставить результат.
        actual = paired_group_bootstrap(
            self.reference.iloc[::-1],
            self.candidate.iloc[[2, 0, 3, 1]],
            self.groups.iloc[[1, 3, 0, 2]],
            [0, 1],
            "failure_within_30d",
            n_bootstrap=30,
        )
        self.assertEqual(expected, actual)
        self.assertGreater(actual["difference"], 0)
        self.assertEqual(actual["valid_repeats"], 30)

    def test_sampling_keeps_whole_groups_and_pairs(self):
        # 1) Выбрать первую группу дважды в единственном повторе.
        generator = Mock()
        generator.integers.return_value = np.array([0, 0])
        with (
            patch(
                "industrial_maintenance.comparison.np.random.default_rng",
                return_value=generator,
            ),
            patch(
                "industrial_maintenance.comparison.average_precision_score",
                wraps=average_precision_score,
            ) as metric,
        ):
            paired_group_bootstrap(
                self.reference,
                self.candidate,
                self.groups,
                [0, 1],
                "failure_within_30d",
                n_bootstrap=1,
            )

        # 2) Проверить обе строки группы и одну пару выборок моделей.
        np.testing.assert_array_equal(
            metric.call_args_list[2].args[0], [0, 1, 0, 1]
        )
        np.testing.assert_array_equal(
            metric.call_args_list[2].args[1], [0.9, 0.1, 0.9, 0.1]
        )
        np.testing.assert_array_equal(
            metric.call_args_list[3].args[1], [0.1, 0.9, 0.1, 0.9]
        )

    def test_missing_ids_or_conflicting_truth_are_rejected(self):
        # 1) Не допускать сравнение на пересечении вместо полных пар.
        with self.assertRaisesRegex(ValueError, "Наборы id"):
            paired_group_bootstrap(
                self.reference,
                self.candidate.iloc[:-1],
                self.groups,
                [0, 1],
                "failure_within_30d",
                n_bootstrap=2,
            )

        # 2) Не скрывать несовпадение эталонных меток.
        with self.assertRaisesRegex(ValueError, "Эталонные метки"):
            paired_group_bootstrap(
                self.reference,
                self.candidate.assign(y_true=[1, 1, 0, 1]),
                self.groups,
                [0, 1],
                "failure_within_30d",
                n_bootstrap=2,
            )

    def test_undefined_binary_repeats_are_counted(self):
        # 1) Задать группы одного класса и повтор без положительных.
        groups = self.groups.assign(
            group_id=["negative", "positive", "negative", "positive"]
        )
        generator = Mock()
        generator.integers.side_effect = [np.array([0, 0]), np.array([0, 1])]
        with patch(
            "industrial_maintenance.comparison.np.random.default_rng",
            return_value=generator,
        ):
            result = paired_group_bootstrap(
                self.reference,
                self.candidate,
                groups,
                [0, 1],
                "failure_within_30d",
                n_bootstrap=2,
            )

        # 2) Убедиться, что неопределённый повтор не заменён нулём.
        self.assertEqual(result["valid_repeats"], 1)
        self.assertEqual(result["skipped_repeats"], 1)

    def test_multiclass_identical_models_have_zero_interval(self):
        # 1) Использовать строковые классы и одинаковые прогнозы.
        predictions = self.reference.replace(
            {
                "y_true": {0: "normal", 1: "fault"},
                "y_pred": {0: "normal", 1: "fault"},
            }
        )
        result = paired_group_bootstrap(
            predictions,
            predictions,
            self.groups,
            ["normal", "fault"],
            "fault_type",
            n_bootstrap=20,
        )

        # 2) Проверить точную разницу и обе границы интервала.
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["lower_95"], 0)
        self.assertEqual(result["upper_95"], 0)


if __name__ == "__main__":
    unittest.main()
