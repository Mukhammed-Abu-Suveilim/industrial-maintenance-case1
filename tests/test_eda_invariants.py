"""Малые синтетические примеры; не исследовательские данные."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_maintenance.data import get_or_create_splits, validate_splits
from industrial_maintenance.eda import export_tables, feature_ledger


def synthetic_frame():
    rows = []
    for group in range(30):
        for replica in range(3):
            rows.append({
                "id": str(group * 3 + replica), "equipment": "synthetic",
                "operating_hours": group, "temperature_c": group + replica,
                "vibration_mm_s": 1.0, "pressure_bar": 2.0, "load_ratio": 0.5,
                "operator_log": f"synthetic log {group}", "manual_id": "synthetic",
                "fault_type": "synthetic_a" if group % 2 else "synthetic_b",
                "severity": "synthetic", "required_route": "synthetic", "horizon_days": 10,
                "failure_within_30d": group % 2,
            })
    return pd.DataFrame(rows).astype({"id": "string"})


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()
        self.config = {"sha256": "synthetic", "seed": 42, "split_version": "synthetic",
                       "n_splits": 5, "test_fold": 0, "validation_fold": 1,
                       "split_group": "normalized_exact_operator_log"}

    def test_repeated_logs_stay_together_and_reload_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _ = get_or_create_splits(root, self.frame, self.config)
            second, _ = get_or_create_splits(root, self.frame, self.config)
            self.assertEqual(first.astype(str).to_dict("list"), second.astype(str).to_dict("list"))
            validate_splits(self.frame, first)

    def test_cross_split_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            membership, _ = get_or_create_splits(Path(tmp), self.frame, self.config)
            corrupt = membership.copy()
            corrupt.loc[0, "split"] = "validation" if corrupt.loc[1, "split"] != "validation" else "test"
            with self.assertRaisesRegex(ValueError, "operator_log"):
                validate_splits(self.frame, corrupt)

    def test_changed_protocol_cannot_overwrite_reserved_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            get_or_create_splits(root, self.frame, self.config)
            changed = {**self.config, "seed": 43}
            with self.assertRaisesRegex(ValueError, "молча"):
                get_or_create_splits(root, self.frame, changed)

    def test_duplicate_ids_cannot_replace_missing_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            membership, _ = get_or_create_splits(Path(tmp), self.frame, self.config)
            membership.loc[0, "id"] = membership.loc[1, "id"]
            with self.assertRaises(ValueError):
                validate_splits(self.frame, membership)


class EvidenceTests(unittest.TestCase):
    def test_availability_assumption_preserves_target_and_id_roles(self):
        # 1) Задать допущение для всех нецелевых полей примера.
        frame = synthetic_frame()
        targets = ["failure_within_30d", "fault_type"]
        config = {
            "targets": targets,
            "availability": {
                field: "available_by_assumption"
                for field in frame.columns if field not in targets
            },
            "availability_basis": "Допущение пользователя от 2026-10-16",
        }
        associations = pd.DataFrame(
            columns=["Поле", "Цель", "Метод", "Эффект"]
        )

        # 2) Проверить применение допущения к ранее спорным полям.
        ledger = feature_ledger(frame, associations, config).set_index("Поле")
        fields = ["severity", "required_route", "horizon_days"]
        self.assertTrue(ledger.loc[
            fields, "Статус"
        ].eq("Доступность принята как допущение").all())
        self.assertTrue(ledger.loc[
            fields, "Основание доступности"
        ].eq(config["availability_basis"]).all())

        # 3) Сохранить роли целей и идентификатора записи.
        self.assertEqual(ledger.loc["id", "Статус"], "Не вход модели")
        self.assertTrue(ledger.loc[
            targets, "Статус"
        ].eq("Не вход ни одной из двух моделей").all())

    def test_every_field_is_accounted_for_and_targets_are_not_inputs(self):
        frame = synthetic_frame()
        config = {"targets": ["failure_within_30d", "fault_type"], "availability": {}}
        associations = pd.DataFrame(columns=["Поле", "Цель", "Метод", "Эффект"])
        ledger = feature_ledger(frame, associations, config)
        self.assertEqual(set(ledger["Поле"]), set(frame.columns))
        self.assertTrue((ledger.loc[ledger["Поле"].isin(config["targets"]), "Статус"] == "Не вход ни одной из двух моделей").all())

    def test_export_does_not_erase_human_feature_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manual = folder / "feature_decisions.csv"
            manual.write_text("human decision", encoding="utf-8")
            export_tables(folder, {"feature_decisions": pd.DataFrame({"a": [1]})}, {})
            self.assertEqual(manual.read_text(encoding="utf-8"), "human decision")
            self.assertTrue((folder / "feature_evidence_current.csv").exists())


if __name__ == "__main__":
    unittest.main()
