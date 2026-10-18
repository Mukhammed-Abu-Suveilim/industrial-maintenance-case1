"""Синтетические проверки происхождения текстов и прогнозов без fit."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_maintenance.wording_checks import (  # noqa: E402
    validate_wording_pairs,
    wording_prediction_table,
)


class WordingTests(unittest.TestCase):
    def setUp(self):
        # 1) Задать явно синтетический источник из validation.
        self.validation = pd.DataFrame(
            {
                "id": ["source"],
                "operator_log": ["Нет шума. Нагрузка 50%."],
                "fault_type": ["normal"],
            }
        )
        self.payload = {
            "dataset_kind": "synthetic_wording_check",
            "source_split": "validation",
            "training_allowed": False,
            "failure_labels_available": False,
            "pairs": [
                {
                    "source_id": "source",
                    "source_operator_log": "Нет шума. Нагрузка 50%.",
                    "fault_type": "normal",
                    "paraphrases": [
                        {
                            "id": "rewrite",
                            "style": "order",
                            "operator_log": "Нагрузка 50%, шум отсутствует.",
                        }
                    ],
                }
            ],
        }

    def test_pairs_preserve_ids_and_have_no_failure_labels(self):
        table = validate_wording_pairs(
            self.payload, self.validation, self.validation.operator_log
        )
        self.assertEqual(table.source_id.tolist(), ["source"])
        self.assertTrue(table.synthetic.all())
        self.assertNotIn("failure_within_30d", table)

    def test_sources_labels_numbers_and_duplicates_are_checked(self):
        # 1) Не допускать произвольные источники и изменённые диагнозы.
        for name, value in [
            ("source_id", "test_id"),
            ("fault_type", "seal_leak"),
            ("source_operator_log", "Другой текст"),
        ]:
            payload = copy.deepcopy(self.payload)
            payload["pairs"][0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_wording_pairs(payload, self.validation, [])

        # 2) Отклонить другую нагрузку, знакомый текст и повтор id.
        for text in ["Нагрузка 80%, шума нет.", "Нет шума. Нагрузка 50%."]:
            payload = copy.deepcopy(self.payload)
            payload["pairs"][0]["paraphrases"][0]["operator_log"] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_wording_pairs(
                    payload, self.validation, self.validation.operator_log
                )
        self.payload["pairs"][0]["paraphrases"] *= 2
        with self.assertRaises(ValueError):
            validate_wording_pairs(self.payload, self.validation, [])

    def test_probabilities_follow_ids_classes_and_fixed_threshold(self):
        binary = wording_prediction_table(
            ["b", "a"],
            [[0.3, 0.7], [0.9, 0.1]],
            [1, 0],
            "failure_within_30d",
            0.3,
        )
        self.assertEqual(binary.id.tolist(), ["b", "a"])
        self.assertEqual(binary.y_pred.tolist(), [1, 1])
        self.assertNotIn("y_true", binary)
        diagnosis = wording_prediction_table(
            ["a"], [[0.8, 0.2]], ["normal", "seal_leak"], "fault_type", 0.5
        )
        self.assertEqual(diagnosis.y_pred.tolist(), ["normal"])
        with self.assertRaises(ValueError):
            wording_prediction_table(
                ["a"], [[np.nan, 0.2]], [0, 1], "failure_within_30d", 0.5
            )


if __name__ == "__main__":
    unittest.main()
