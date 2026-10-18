"""Парное сравнение прогнозов с учётом групп зависимых записей."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, fbeta_score


def paired_group_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    membership: pd.DataFrame,
    classes: list,
    target: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
    metric: str | None = None,
) -> dict:
    """Оценить candidate − reference по AP, macro-F1 или F2 на общих id.

    membership содержит ровно оцениваемые id и group_id. Группы
    выбираются с возвращением, обе модели получают одни строки.
    Интервал описывает фиксированные прогнозы после выбора моделей;
    повторное обучение и неопределённость выбора в него не входят.
    """
    # 1) Проверить однозначность и полноту пар до выравнивания строк.
    if target not in {"failure_within_30d", "fault_type"}:
        raise ValueError("Неизвестная целевая переменная.")
    if n_bootstrap < 1:
        raise ValueError("Число повторов должно быть положительным.")
    if metric is None:
        metric = (
            "average_precision"
            if target == "failure_within_30d"
            else "macro_f1"
        )
    allowed_metrics = (
        {"average_precision", "f2"}
        if target == "failure_within_30d"
        else {"macro_f1"}
    )
    if metric not in allowed_metrics:
        raise ValueError("Метрика не соответствует целевой переменной.")
    tables = []
    for table in [reference, candidate, membership]:
        copy = table.copy()
        if copy["id"].isna().any() or copy["id"].duplicated().any():
            raise ValueError("Нужны уникальные непустые id.")
        copy["id"] = copy["id"].astype("string")
        if copy["id"].duplicated().any():
            raise ValueError("id неоднозначны после приведения к строкам.")
        tables.append(copy.set_index("id").sort_index())
    reference, candidate, groups = tables
    if reference.empty or not reference.index.equals(candidate.index):
        raise ValueError("Наборы id прогнозов не совпадают или пусты.")
    if not reference.index.equals(groups.index):
        raise ValueError("Группы должны покрывать ровно оцениваемые id.")
    if groups["group_id"].isna().any():
        raise ValueError("Для каждого id нужна непустая группа.")
    if not reference["y_true"].equals(candidate["y_true"]):
        raise ValueError("Эталонные метки моделей не совпадают.")
    if not reference["y_true"].isin(classes).all():
        raise ValueError("Эталонные метки отсутствуют среди classes.")

    # 2) Подготовить массивы; одна группа всегда переносится целиком.
    truth = reference["y_true"].to_numpy()
    if metric == "average_precision":
        if set(classes) != {0, 1}:
            raise ValueError("Для отказа нужны классы 0 и 1.")
        reference_values = reference["probability_1"].to_numpy()
        candidate_values = candidate["probability_1"].to_numpy()
        if not np.isfinite([reference_values, candidate_values]).all():
            raise ValueError("Вероятности должны быть конечными.")
        if not np.any(truth == 1):
            raise ValueError("AP не определена без положительных меток.")
        reference_score = average_precision_score(truth, reference_values)
        candidate_score = average_precision_score(truth, candidate_values)
    else:
        reference_values = reference["y_pred"].to_numpy()
        candidate_values = candidate["y_pred"].to_numpy()
        if not all(
            np.isin(values, classes).all()
            for values in [reference_values, candidate_values]
        ):
            raise ValueError("Предсказанные метки отсутствуют среди classes.")
        if metric == "f2":
            if set(classes) != {0, 1} or set(truth) != {0, 1}:
                raise ValueError("Для сравнения F2 нужны оба класса 0 и 1.")
            score = fbeta_score
            score_parameters = {"beta": 2, "zero_division": 0}
        else:
            score = f1_score
            score_parameters = {
                "labels": classes,
                "average": "macro",
                "zero_division": 0,
            }
        reference_score = score(truth, reference_values, **score_parameters)
        candidate_score = score(truth, candidate_values, **score_parameters)
    group_positions = list(
        groups.groupby("group_id", sort=True).indices.values()
    )
    generator = np.random.default_rng(seed)

    # 3) Повторять парное оценивание с возвращением целых групп.
    differences = []
    for _ in range(n_bootstrap):
        draw = generator.integers(
            len(group_positions), size=len(group_positions)
        )
        positions = np.concatenate([group_positions[index] for index in draw])
        sampled_truth = truth[positions]
        if metric == "average_precision":
            if not np.any(sampled_truth == 1):
                continue
            reference_sample = average_precision_score(
                sampled_truth, reference_values[positions]
            )
            candidate_sample = average_precision_score(
                sampled_truth, candidate_values[positions]
            )
        else:
            reference_sample = score(
                sampled_truth,
                reference_values[positions],
                **score_parameters,
            )
            candidate_sample = score(
                sampled_truth,
                candidate_values[positions],
                **score_parameters,
            )
        differences.append(float(candidate_sample - reference_sample))

    # 4) Вернуть процентильный интервал и число неопределённых повторов.
    lower, upper = (None, None)
    if differences:
        lower, upper = np.quantile(differences, [0.025, 0.975]).tolist()
    return {
        "metric": metric,
        "reference_score": float(reference_score),
        "candidate_score": float(candidate_score),
        "difference": float(candidate_score - reference_score),
        "lower_95": lower,
        "upper_95": upper,
        "groups": len(group_positions),
        "requested_repeats": n_bootstrap,
        "valid_repeats": len(differences),
        "skipped_repeats": n_bootstrap - len(differences),
        "seed": seed,
        "direction": "candidate - reference",
    }
