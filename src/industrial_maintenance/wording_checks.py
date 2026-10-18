"""Проверки синтетических формулировок без обучения и меток отказа."""

import re

import numpy as np
import pandas as pd


def validate_wording_pairs(payload, validation, known_texts):
    """Проверить происхождение, числа и уникальность перефразировок.

    Проверка программы не заменяет проверку смысла человеком.
    Исходные строки должны принадлежать только validation.
    """
    # 1) Не разрешать подмену обучающих данных или меток отказа.
    if (
        payload["dataset_kind"] != "synthetic_wording_check"
        or payload["source_split"] != "validation"
        or payload["training_allowed"] is not False
        or payload["failure_labels_available"] is not False
    ):
        raise ValueError("Нужен отдельный синтетический набор без обучения.")
    sources = validation.set_index("id")
    if not sources.index.is_unique:
        raise ValueError("Нужны однозначные исходные id.")
    known = {" ".join(text.lower().split()) for text in known_texts}
    records = []

    # 2) Сопоставить текст и диагноз с исходной строкой validation.
    for pair in payload["pairs"]:
        source_id = pair["source_id"]
        if source_id not in sources.index:
            raise ValueError("Источник отсутствует в validation.")
        source = sources.loc[source_id]
        if (
            pair["source_operator_log"] != source["operator_log"]
            or pair["fault_type"] != source["fault_type"]
        ):
            raise ValueError("Изменён исходный текст или его диагноз.")
        original_numbers = re.findall(r"\d+(?:[.,]\d+)?", source.operator_log)
        for paraphrase in pair["paraphrases"]:
            text = paraphrase["operator_log"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Нужна непустая переформулировка.")
            if re.findall(r"\d+(?:[.,]\d+)?", text) != original_numbers:
                raise ValueError("Числа исходного наблюдения изменены.")
            if " ".join(text.lower().split()) in known:
                raise ValueError("Переформулировка уже встречалась в данных.")
            records.append(
                {
                    "id": paraphrase["id"],
                    "source_id": source_id,
                    "source_operator_log": source.operator_log,
                    "operator_log": text,
                    "fault_type": source.fault_type,
                    "style": paraphrase["style"],
                    "synthetic": True,
                }
            )

    # 3) Сохранить связь четырёх вариантов с одним исходным наблюдением.
    table = pd.DataFrame(records)
    if table.empty or table.id.isna().any() or table.id.duplicated().any():
        raise ValueError("Нужны непустые уникальные id переформулировок.")
    if table.operator_log.duplicated().any():
        raise ValueError("Повторяющиеся переформулировки не допускаются.")
    return table


def wording_prediction_table(ids, probabilities, classes, target, threshold):
    """Сохранить прогнозы без выдумывания эталонных меток отказа."""
    # 1) Проверить id, порядок классов и вероятности.
    values = np.asarray(probabilities, dtype=float)
    if len(set(ids)) != len(ids) or pd.isna(ids).any():
        raise ValueError("Нужны уникальные непустые id.")
    if len(set(classes)) != len(classes) or not len(classes):
        raise ValueError("Нужны различные классы.")
    if values.shape != (len(ids), len(classes)):
        raise ValueError("Вероятности не согласованы с id и классами.")
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1)
    ):
        raise ValueError("Некорректные вероятности.")

    # 2) Применить сохранённое правило, не подбирать новый порог.
    if target == "failure_within_30d":
        if set(classes) != {0, 1} or not 0 <= threshold <= 1:
            raise ValueError("Нужны классы 0/1 и корректный порог.")
        positive = values[:, list(classes).index(1)]
        predicted = (positive >= threshold).astype(int)
    elif target == "fault_type":
        predicted = np.asarray(classes)[values.argmax(axis=1)]
    else:
        raise ValueError("Неизвестная цель.")
    result = pd.DataFrame({"id": ids, "y_pred": predicted})
    for position, label in enumerate(classes):
        result[f"probability_{label}"] = values[:, position]
    return result
