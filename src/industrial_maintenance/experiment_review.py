"""Проверка сохранённых прогнозов и измерение CPU-применения без fit."""

from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import auc, precision_recall_curve

from industrial_maintenance.baseline import classification_metrics


def align_predictions(predictions, validation, target, classes):
    """Проверить id, метки и вероятности, вернуть порядок validation."""
    # 1) Отклонить неоднозначные или посторонние объекты.
    predictions = predictions.copy()
    predictions["id"] = predictions["id"].astype("string")
    ids = validation["id"].astype("string")
    if ids.duplicated().any() or predictions["id"].duplicated().any():
        raise ValueError("id должны быть уникальными.")
    if set(predictions["id"]) != set(ids):
        raise ValueError("Прогнозы должны покрывать ровно validation.")
    aligned = predictions.set_index("id").loc[ids].reset_index()
    if not np.array_equal(aligned["y_true"], validation[target]):
        raise ValueError("Эталонные метки не соответствуют id.")

    # 2) Проверить явный порядок классов и распределения вероятностей.
    if not aligned["y_pred"].isin(classes).all():
        raise ValueError("Неизвестный предсказанный класс.")
    values = aligned[[f"probability_{label}" for label in classes]]
    if (
        not np.isfinite(values).all().all()
        or ((values < 0) | (values > 1)).any().any()
        or not np.allclose(values.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Некорректные вероятности.")
    return aligned


def review_metrics(predictions, classes, target):
    """Добавить трапецеидальную PR-AUC, не переименовывая AP."""
    # 1) Повторно использовать принятые определения основных метрик.
    metrics = classification_metrics(predictions, classes, target)

    # 2) Рассчитать отдельную PR-AUC при наличии обоих классов.
    if target == "failure_within_30d":
        metrics["pr_auc_trapezoid"] = None
        if predictions["y_true"].nunique() == 2:
            precision, recall, _ = precision_recall_curve(
                predictions["y_true"], predictions["probability_1"]
            )
            metrics["pr_auc_trapezoid"] = float(auc(recall, precision))
    return metrics


def subgroup_review(predictions, validation, classes, target):
    """Рассчитать метрики для equipment/severity с числом объектов."""
    # 1) Зафиксировать соответствие строк по id, а не их положению.
    predictions = align_predictions(predictions, validation, target, classes)
    joined = predictions.merge(
        validation[["id", "equipment", "severity"]],
        on="id",
        validate="one_to_one",
    )

    # 2) Сохранить размеры и метрики; малые группы не отбрасывать.
    rows = []
    for field in ["equipment", "severity"]:
        for value, group in joined.groupby(field, sort=True):
            rows.append(
                {
                    "grouping": field,
                    "value": value,
                    "records": len(group),
                    "positives": (
                        int(group["y_true"].eq(1).sum())
                        if target == "failure_within_30d"
                        else None
                    ),
                    **review_metrics(group, classes, target),
                }
            )
    return pd.DataFrame(rows)


def time_predictions(predict, features, batch_size, repeats):
    """Измерить тёплое применение: весь набор по равным пакетам."""
    # 1) Проверить протокол и прогреть тот же путь вычислений.
    if features.empty or batch_size < 1 or repeats < 1:
        raise ValueError("Нужны объекты, положительные batch и repeats.")
    predict(features.iloc[:batch_size])

    # 2) Включить предобработку и вероятности в каждое измерение.
    durations = []
    probabilities = None
    for _ in range(repeats):
        outputs = []
        started = perf_counter()
        for offset in range(0, len(features), batch_size):
            outputs.append(
                predict(features.iloc[offset : offset + batch_size])
            )
        probabilities = np.concatenate(outputs)
        durations.append(perf_counter() - started)

    # 3) Вернуть все замеры; на объект — амортизированное время.
    return probabilities, {
        "batch_size": batch_size,
        "records": len(features),
        "repeats": repeats,
        "seconds": durations,
        "median_seconds": float(np.median(durations)),
        "amortized_ms_per_record": (
            float(np.median(durations)) * 1000 / len(features)
        ),
    }


def load_cpu_predictor(root, specification, threads):
    """Загрузить модель и вернуть функцию вероятностей и классы."""
    # 1) Для классических моделей сохранить готовую предобработку.
    directory = root / specification["run_path"]
    if specification["family"] != "rubert":
        model = joblib.load(directory / "model.pkl")
        if specification["features"] == ["operator_log"]:

            def predict_text(features):
                return model.predict_proba(features["operator_log"])

            return predict_text, model.classes_.tolist()
        return model.predict_proba, model.classes_.tolist()

    # 2) Загрузить сохранённые веса и токенизатор без сети и Trainer.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        directory / "model",
        local_files_only=True,
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            directory / "model",
            local_files_only=True,
            attn_implementation="eager",
        )
        .cpu()
        .eval()
    )
    classes = specification["classes"]
    model_classes = [
        model.config.id2label[index] for index in range(len(classes))
    ]
    if model_classes != [str(label) for label in classes]:
        raise ValueError("Классы checkpoint не совпадают с конфигурацией.")

    # 3) Включать токенизацию в измеряемый путь каждого пакета.
    def predict(features):
        texts = features["operator_log"].tolist()
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=specification["max_length"],
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
        return softmax(logits.numpy().astype(float), axis=1)

    return predict, classes
