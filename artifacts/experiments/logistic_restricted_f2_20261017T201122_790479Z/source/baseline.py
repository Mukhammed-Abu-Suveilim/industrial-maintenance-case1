"""Предобработка и оценивание табличной логистической регрессии."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_text_pipeline(
    tfidf_parameters: dict, model_parameters: dict
) -> Pipeline:
    """Создать необученный Pipeline для Series исходных operator_log."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_parameters)),
            ("classifier", LogisticRegression(**model_parameters)),
        ]
    )


def build_logistic_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_parameters: dict,
) -> Pipeline:
    """Создать необученный Pipeline для одного табличного опыта."""
    # 1) Защитить входы от целей, id, текста и исключённых полей.
    columns = numeric_columns + categorical_columns
    excluded = {
        "id",
        "fault_type",
        "failure_within_30d",
        "operator_log",
        "severity",
        "required_route",
    }
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Нужны непустые, неповторяющиеся входные поля.")
    if excluded.intersection(columns):
        raise ValueError(
            "Цели, id, текст, severity и required_route не входят в опыт."
        )

    # 2) Определить преобразования; обучение произойдёт только при fit.
    transformations = [("numeric", StandardScaler(), numeric_columns)]
    if categorical_columns:
        transformations.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        )
    preprocessing = ColumnTransformer(transformations, remainder="drop")

    # 3) Вернуть стандартный Pipeline без отдельного уровня управления.
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(**model_parameters)),
        ]
    )


def prediction_table(
    record_ids,
    true_labels,
    probabilities: np.ndarray,
    classes,
    target: str,
    threshold: float,
) -> pd.DataFrame:
    """Связать вероятности с id в порядке переданных строк и классов."""
    # 1) Проверить соответствие строк и вероятностный смысл значений.
    ids = np.asarray(record_ids)
    truth = np.asarray(true_labels)
    classes = np.asarray(classes)
    probabilities = np.asarray(probabilities, dtype=float)
    if target not in {"failure_within_30d", "fault_type"}:
        raise ValueError("Неизвестная целевая переменная.")
    if probabilities.shape != (len(ids), len(classes)):
        raise ValueError("Размер вероятностей не соответствует id/классам.")
    if len(truth) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Метки не согласованы с id или id повторяются.")
    if pd.isna(ids).any() or not np.isin(truth, classes).all():
        raise ValueError("Пустые id или метки вне обученного набора классов.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Вероятности должны быть конечными числами.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Вероятности должны принадлежать [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Вероятности классов должны давать сумму 1.")

    # 2) Применить бинарный порог или выбрать наиболее вероятный класс.
    if target == "failure_within_30d":
        if set(classes) != {0, 1} or not 0 <= threshold <= 1:
            raise ValueError("Ожидаются классы 0/1 и порог в [0, 1].")
        positive_position = classes.tolist().index(1)
        predicted = (probabilities[:, positive_position] >= threshold).astype(
            int
        )
    else:
        predicted = classes[probabilities.argmax(axis=1)]

    # 3) Сохранить исходные вероятности с явным именем каждого класса.
    table = pd.DataFrame({"id": ids, "y_true": truth, "y_pred": predicted})
    for position, label in enumerate(classes):
        table[f"probability_{label}"] = probabilities[:, position]
    return table


def classification_metrics(
    predictions: pd.DataFrame, classes, target: str
) -> dict:
    """Рассчитать метрики, вернув None для неопределённых ROC-AUC/AP."""
    # 1) Восстановить порядок классов для согласованного расчёта потерь.
    ordered_classes = sorted(np.asarray(classes).tolist())
    probability_columns = [f"probability_{label}" for label in ordered_classes]
    truth = predictions["y_true"].to_numpy()
    predicted = predictions["y_pred"].to_numpy()
    metrics = {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(
            truth,
            predicted,
            labels=ordered_classes,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            truth, predictions[probability_columns], labels=ordered_classes
        ),
    }

    # 2) Добавить бинарные метрики с положительным классом отказа 1.
    if target == "failure_within_30d":
        positive_probabilities = predictions["probability_1"].to_numpy()
        metrics.update(
            {
                "precision": precision_score(
                    truth, predicted, zero_division=0
                ),
                "recall": recall_score(truth, predicted, zero_division=0),
                "f1": f1_score(truth, predicted, zero_division=0),
                "f2": fbeta_score(truth, predicted, beta=2, zero_division=0),
                "roc_auc": (
                    roc_auc_score(truth, positive_probabilities)
                    if len(np.unique(truth)) == 2
                    else None
                ),
                "average_precision": (
                    average_precision_score(truth, positive_probabilities)
                    if np.any(truth == 1)
                    else None
                ),
            }
        )

    # 3) Привести числовые типы к сериализуемым значениям Python.
    return {
        name: float(value) if value is not None else None
        for name, value in metrics.items()
    }


def select_failure_threshold(
    true_labels,
    positive_probabilities,
    selection_decimals: int = 12,
) -> tuple[dict, pd.DataFrame]:
    """Выбрать порог по F2 и вернуть результат и полный перебор.

    Входы — одномерные согласованные массивы validation. Решение
    положительно при p >= порогу. После округления F2 равенство
    разрешается по precision и большему порогу. Входы не меняются.
    """
    # 1) Проверить метки и вероятности до формирования кандидатов.
    truth = np.asarray(true_labels)
    probabilities = np.asarray(positive_probabilities, dtype=float)
    if truth.ndim != 1 or probabilities.shape != truth.shape:
        raise ValueError("Нужны одномерные согласованные метки и вероятности.")
    if set(truth.tolist()) != {0, 1}:
        raise ValueError("Validation должна содержать оба класса: 0 и 1.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Вероятности должны быть конечными числами.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Вероятности должны принадлежать [0, 1].")
    if not isinstance(selection_decimals, int) or selection_decimals < 0:
        raise ValueError("Точность сравнения должна быть целым числом >= 0.")

    # 2) Перебрать уникальные вероятности и обязательные пороги.
    thresholds = np.unique(np.concatenate([probabilities, [0, 0.5, 1]]))
    positive = truth == 1
    records = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = int(np.count_nonzero(positive & predicted))
        fp = int(np.count_nonzero(~positive & predicted))
        tn = int(np.count_nonzero(~positive & ~predicted))
        fn = int(np.count_nonzero(positive & ~predicted))
        records.append(
            {
                "threshold": float(threshold),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn),
                "f1": 2 * tp / (2 * tp + fp + fn),
                "f2": 5 * tp / (5 * tp + fp + 4 * fn),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    curve = pd.DataFrame(records)

    # 3) Разрешить равенство по precision, затем по большему порогу.
    ranked = curve.assign(
        rounded_f2=curve["f2"].round(selection_decimals),
    ).sort_values(
        ["rounded_f2", "precision", "threshold"],
        ascending=False,
        kind="stable",
    )
    best_index = ranked.index[0]
    best = records[best_index].copy()
    curve["selected"] = curve.index == best_index
    return best, curve
