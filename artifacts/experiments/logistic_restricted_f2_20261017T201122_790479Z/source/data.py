"""Загрузка неизменяемого набора данных и зафиксированное разбиение."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

EXPECTED_COLUMNS = [
    "id", "equipment", "operating_hours", "temperature_c", "vibration_mm_s",
    "pressure_bar", "load_ratio", "operator_log", "manual_id", "fault_type",
    "severity", "required_route", "horizon_days", "failure_within_30d",
]


def load_config(root: Path) -> dict:
    return json.loads((root / "configs/eda.json").read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(root: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    path = root / config["dataset"]
    checksum = file_sha256(path)
    if checksum != config["sha256"]:
        raise ValueError("Контрольная сумма CSV изменилась: сначала обновите описание источника.")
    frame = pd.read_csv(path, encoding="utf-8", dtype={"id": "string"})
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Схема CSV не совпадает с ожидаемыми именами и порядком полей.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("id должен быть непустым и уникальным.")
    for target in config["targets"]:
        if frame[target].isna().any():
            raise ValueError(f"Пустая целевая переменная: {target}")
    if not set(frame["failure_within_30d"].unique()).issubset({0, 1}):
        raise ValueError("failure_within_30d должна содержать только 0 и 1.")
    for column in config["numeric_columns"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    manifest = {
        "dataset": config["dataset"], "sha256": checksum,
        "rows": len(frame), "columns": list(frame.columns),
        "bytes": path.stat().st_size,
    }
    return frame, manifest


def normalize_log(value: object, mask_numbers: bool = False) -> str:
    text = "<missing>" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if mask_numbers:
        text = re.sub(r"\d+(?:[.,]\d+)?", "<number>", text)
    return text


def log_groups(frame: pd.DataFrame) -> pd.Series:
    return frame["operator_log"].map(normalize_log).map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


def duplicate_audit(frame: pd.DataFrame) -> pd.Series:
    observed = ["equipment", "operating_hours", "temperature_c", "vibration_mm_s",
                "pressure_bar", "load_ratio", "operator_log", "manual_id"]
    normalized = frame["operator_log"].map(normalize_log)
    templates = frame["operator_log"].map(lambda value: normalize_log(value, True))
    return pd.Series({
        "Повторы всех полей без id": int(frame.drop(columns="id").duplicated().sum()),
        "Повторы наблюдений по восьми полям": int(frame[observed].duplicated().sum()),
        "Различные operator_log": int(normalized.nunique()),
        "Повторные operator_log": int(normalized.duplicated().sum()),
        "Различные тексты после маскирования чисел": int(templates.nunique()),
    }, name="Число")


def validate_splits(frame: pd.DataFrame, membership: pd.DataFrame) -> None:
    required = {"id", "split", "group_id"}
    if set(membership.columns) != required:
        raise ValueError("Некорректная схема файла разбиения.")
    if membership.isna().any().any() or membership["id"].duplicated().any():
        raise ValueError("Пропуски или повторные id в разбиении.")
    if set(membership["id"]) != set(frame["id"]):
        raise ValueError("Разбиение не покрывает исходные id ровно один раз.")
    if set(membership["split"]) != {"train", "validation", "test"}:
        raise ValueError("Ожидаются три непустые выборки.")
    if membership.groupby("group_id")["split"].nunique().max() != 1:
        raise ValueError("Одинаковые operator_log попали в разные выборки.")
    expected = pd.Series(log_groups(frame).to_numpy(), index=frame["id"])
    actual = membership.set_index("id")["group_id"].reindex(expected.index)
    if actual.tolist() != expected.tolist():
        raise ValueError("Группы не соответствуют исходным operator_log.")
    observed = ["equipment", "operating_hours", "temperature_c", "vibration_mm_s",
                "pressure_bar", "load_ratio", "operator_log", "manual_id"]
    joined = frame.merge(membership[["id", "split"]], on="id", validate="one_to_one")
    if joined.groupby(observed, dropna=False)["split"].nunique().max() > 1:
        raise ValueError("Повторные наблюдения пересекают выборки.")


def get_or_create_splits(root: Path, frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    folder = root / "Data/splits" / config["split_version"]
    path, meta_path = folder / "membership.csv", folder / "manifest.json"
    protocol = {key: config[key] for key in (
        "sha256", "seed", "n_splits", "test_fold", "validation_fold", "split_group"
    )}
    if path.exists() != meta_path.exists():
        raise ValueError("Разбиение сохранено не полностью; восстановите пару файлов.")
    if path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["protocol"] != protocol:
            raise ValueError("Протокол изменён. Нельзя молча пересоздать выборки этой версии.")
        if metadata["membership_sha256"] != file_sha256(path):
            raise ValueError("Сохранённая принадлежность к выборкам была изменена.")
        membership = pd.read_csv(path, dtype={"id": "string", "group_id": "string"})
    else:
        if config["split_group"] != "normalized_exact_operator_log":
            raise ValueError("Неизвестное правило группировки.")
        groups = log_groups(frame)
        n_splits = config["n_splits"]
        if not (0 <= config["test_fold"] < n_splits and 0 <= config["validation_fold"] < n_splits) or config["test_fold"] == config["validation_fold"]:
            raise ValueError("Нужны два разных допустимых номера отложенных частей.")
        if groups.nunique() < n_splits:
            raise ValueError("Недостаточно групп для указанного разбиения.")
        joint_target = frame["fault_type"].astype(str) + "|" + frame["failure_within_30d"].astype(str)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config["seed"])
        membership = pd.DataFrame({"id": frame["id"], "split": "train", "group_id": groups})
        for fold, (_, held_indices) in enumerate(splitter.split(frame, joint_target, groups)):
            if fold == config["test_fold"]:
                membership.loc[held_indices, "split"] = "test"
            elif fold == config["validation_fold"]:
                membership.loc[held_indices, "split"] = "validation"
        validate_splits(frame, membership)
        metadata = {
            "protocol": protocol,
            "description": "Одинаковые operator_log после приведения регистра/пробелов остаются вместе; стратификация по паре целевых меток приблизительна.",
            "limitation": "Общие тексты после маскирования чисел могут пересекать выборки. Это не временная проверка и не доказательство качества на неизвестных шаблонах или машинах.",
            "sizes": membership["split"].value_counts().to_dict(),
        }
        folder.mkdir(parents=True, exist_ok=True)
        membership.to_csv(path, index=False, encoding="utf-8")
        metadata["membership_sha256"] = file_sha256(path)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_splits(frame, membership)
    return membership, metadata


def split_overlap_audit(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    joined = frame.merge(membership, on="id", validate="one_to_one")
    joined["masked_text"] = joined["operator_log"].map(lambda value: normalize_log(value, True))
    records = []
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        a, b = joined[joined["split"] == left], joined[joined["split"] == right]
        records.append({
            "Пара выборок": f"{left} / {right}",
            "Общие id": len(set(a["id"]) & set(b["id"])),
            "Общие точные тексты": len(set(a["group_id"]) & set(b["group_id"])),
            "Общие тексты без чисел": len(set(a["masked_text"]) & set(b["masked_text"])),
        })
    return pd.DataFrame(records)
