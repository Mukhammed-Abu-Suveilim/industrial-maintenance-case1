"""Проверки и описательные связи; результаты не выполняют отбор признаков."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer


def quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for field in frame.columns:
        series = frame[field]
        numeric = pd.api.types.is_numeric_dtype(series)
        records.append({
            "Поле": field, "Тип": str(series.dtype), "Пропуски": int(series.isna().sum()),
            "Доля пропусков": float(series.isna().mean()), "Различные значения": int(series.nunique()),
            "Пустые строки": int(series.map(lambda x: isinstance(x, str) and not x.strip()).sum()),
            "Бесконечные значения": int(np.isinf(series.dropna()).sum()) if numeric else 0,
        })
    return pd.DataFrame(records).set_index("Поле")


def class_balance(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    counts = frame[field].value_counts(dropna=False).rename("Число")
    return pd.concat([counts, (counts / len(frame)).rename("Доля")], axis=1)


def numeric_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T


def outlier_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    records = []
    for field in columns:
        x = frame[field].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        flags = (x < lower) | (x > upper)
        records.append({"Поле": field, "Нижняя граница IQR": lower,
                        "Верхняя граница IQR": upper, "Число за границами": int(flags.sum()),
                        "Доля за границами": float(flags.mean()), "IQR": spread})
    return pd.DataFrame(records).set_index("Поле")


def numeric_associations(frame: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    records = []
    for field in columns:
        valid = frame[[field, target]].replace([np.inf, -np.inf], np.nan).dropna()
        levels = sorted(valid[target].unique(), key=str)
        samples = [valid.loc[valid[target] == level, field].to_numpy() for level in levels]
        record = {"Поле": field, "Цель": target, "Метод": None, "Эффект": np.nan,
                  "p-value": np.nan, "n": len(valid), "Примечание": "Описательное сравнение; независимость событий не подтверждена."}
        if len(samples) < 2 or valid[field].nunique() < 2:
            record["Примечание"] = "Недостаточно классов или изменчивости для сравнения."
        elif len(samples) == 2:
            statistic, pvalue = stats.mannwhitneyu(samples[1], samples[0], alternative="two-sided", method="asymptotic")
            record.update({"Метод": "Mann–Whitney U / rank-biserial", "Эффект": 2 * statistic / (len(samples[1]) * len(samples[0])) - 1,
                           "p-value": pvalue})
        else:
            statistic, pvalue = stats.kruskal(*samples)
            n, k = len(valid), len(samples)
            record.update({"Метод": "Kruskal–Wallis / epsilon-squared", "Эффект": max(0, (statistic - k + 1) / (n - k)) if n > k else np.nan,
                           "p-value": pvalue})
        records.append(record)
    return pd.DataFrame(records)


def categorical_associations(frame: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    records = []
    for field in columns:
        table = pd.crosstab(frame[field].fillna("<missing>"), frame[target])
        n = int(table.to_numpy().sum())
        record = {"Поле": field, "Цель": target, "Метод": "Chi-square / corrected Cramér's V",
                  "Эффект": np.nan, "p-value": np.nan, "n": n,
                  "Примечание": "Описательное сравнение; ассоциация не доказывает допустимость входа."}
        if min(table.shape) >= 2 and n > 1:
            statistic, pvalue, _, expected = stats.chi2_contingency(table, correction=False)
            r, k = table.shape
            phi2 = max(0, statistic / n - ((k - 1) * (r - 1)) / (n - 1))
            denominator = min(k - (k - 1) ** 2 / (n - 1) - 1, r - (r - 1) ** 2 / (n - 1) - 1)
            effect = np.sqrt(phi2 / denominator) if denominator > 0 else np.nan
            record.update({"Эффект": effect, "p-value": pvalue})
            if (expected < 5).any():
                record["Примечание"] = f"В {float((expected < 5).mean()):.1%} ячеек ожидаемое число < 5: приближение p-value может быть ненадёжным."
        else:
            record["Примечание"] = "Недостаточно категорий для сравнения."
        records.append(record)
    return pd.DataFrame(records)


def adjust_pvalues(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["q-value (BH)"] = np.nan
    for _, positions in result.groupby("Цель").groups.items():
        valid = [i for i in positions if pd.notna(result.loc[i, "p-value"])]
        if valid:
            result.loc[valid, "q-value (BH)"] = stats.false_discovery_control(result.loc[valid, "p-value"].to_numpy(), method="bh")
    return result


def association_table(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    derived = frame.assign(operator_log_length=frame["operator_log"].fillna("").str.len())
    records = []
    for target in config["targets"]:
        records.append(numeric_associations(derived, target, config["numeric_columns"] + ["operator_log_length"]))
        records.append(categorical_associations(derived, target, config["categorical_columns"]))
    table = pd.concat(records, ignore_index=True)
    return adjust_pvalues(table).sort_values(["Цель", "q-value (BH)", "Поле"])


def field_relationships(frame: pd.DataFrame, left: str, right: str) -> dict:
    table = pd.crosstab(frame[left], frame[right])
    if table.empty:
        return {"table": table, "summary": "Нет непустых пар значений."}
    left_cardinality = (table > 0).sum(axis=1)
    right_cardinality = (table > 0).sum(axis=0)
    return {
        "table": table,
        "summary": {
            "Поле слева": left, "Поле справа": right,
            "Каждое левое значение связано ровно с одним правым": bool((left_cardinality == 1).all()),
            "Каждое правое значение связано ровно с одним левым": bool((right_cardinality == 1).all()),
            "Это причинная или временная связь": "Не установлено по таблице частот.",
        },
    }


def text_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    texts = frame["operator_log"].fillna("")
    summary = pd.DataFrame({"operator_log_length": texts.str.len(),
                            "operator_log_words": texts.str.split().str.len()}).describe().T
    extracted = pd.to_numeric(texts.str.extract(r"(?i)нагрузка\s+(\d+(?:[.,]\d+)?)\s*%", expand=False).str.replace(",", ".", regex=False), errors="coerce")
    check = pd.DataFrame({"id": frame["id"], "Нагрузка в тексте, %": extracted,
                          "load_ratio × 100": frame["load_ratio"] * 100})
    check["Разница, процентные пункты"] = check["Нагрузка в тексте, %"] - check["load_ratio × 100"]
    return summary, check


def token_associations(frame: pd.DataFrame, target: str, min_documents: int, top_n: int = 10) -> pd.DataFrame:
    vectorizer = CountVectorizer(binary=True, min_df=min_documents, token_pattern=r"(?u)\b[^\W\d_]{3,}\b", lowercase=True)
    try:
        matrix = vectorizer.fit_transform(frame["operator_log"].fillna(""))
    except ValueError:
        return pd.DataFrame(columns=["Класс", "Слово", "Документов класса со словом", "Доля в классе", "Доля вне класса", "Отношение долей со сглаживанием"])
    words = vectorizer.get_feature_names_out()
    records = []
    for level in sorted(frame[target].unique(), key=str):
        mask = (frame[target] == level).to_numpy()
        n_inside, n_outside = int(mask.sum()), int((~mask).sum())
        inside = np.asarray(matrix[mask].sum(axis=0)).ravel()
        outside = np.asarray(matrix[~mask].sum(axis=0)).ravel()
        ratios = ((inside + 1) / (n_inside + 2)) / ((outside + 1) / (n_outside + 2))
        for position in np.argsort(-ratios)[:top_n]:
            records.append({"Класс": level, "Слово": words[position],
                            "Документов класса со словом": int(inside[position]),
                            "Доля в классе": inside[position] / n_inside,
                            "Доля вне класса": outside[position] / n_outside if n_outside else np.nan,
                            "Отношение долей со сглаживанием": ratios[position]})
    return pd.DataFrame(records)


def feature_ledger(
    frame: pd.DataFrame, associations: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Собрать факты, роли и основание доступности каждого поля."""
    # 1) Задать роли кандидатов без автоматического выбора входов.
    rows = []
    sensors = {
        "operating_hours", "temperature_c", "vibration_mm_s",
        "pressure_bar", "load_ratio",
    }

    # 2) Сопоставить наблюдаемые факты и принятую доступность полей.
    for field in frame.columns:
        facts = (
            f"Различных значений: {frame[field].nunique()}; "
            f"пропусков: {int(frame[field].isna().sum())}."
        )
        association_field = (
            "operator_log_length" if field == "operator_log" else field
        )
        evidence = associations[associations["Поле"] == association_field]
        if not evidence.empty:
            effect_descriptions = []
            for _, row in evidence.iterrows():
                effect_descriptions.append(
                    f"{row['Цель']}: {row['Метод']}, "
                    f"эффект={row['Эффект']:.3g}"
                )
            facts += " " + "; ".join(effect_descriptions)

        availability = config["availability"].get(field, "не применимо")
        assumed_available = availability == "available_by_assumption"
        availability_source = (
            config.get("availability_basis", "Допущение пользователя")
            if assumed_available else ""
        )
        role, status = "Кандидат во входы", "Решение не принято"
        reason = (
            "Нужны смысл поля и сравнение вариантов на валидационной выборке."
        )
        if field == "id":
            role, status = "Идентификатор записи", "Не вход модели"
            reason = (
                "Используется для связывания результатов; "
                "семантика времени/машины не подтверждена. "
                "Доступность не превращает идентификатор в измерение."
            )
        elif field in config["targets"]:
            role = "Целевая переменная"
            status = "Не вход ни одной из двух моделей"
            reason = (
                "Метки прогнозируются и используются отдельно "
                "для обучения и оценивания; во входы не передаются."
            )
        elif field in {"severity", "required_route", "horizon_days"}:
            if assumed_available:
                status = "Доступность принята как допущение"
                reason = (
                    "По допущению пользователя поле доступно до прогноза. "
                    "Его можно рассматривать как вход; смысл поля и "
                    "наблюдаемые связи с целями учитывать в сравнении."
                )
            else:
                status = "Ожидается описание данных"
                reason = (
                    "Доступность до прогноза не установлена. "
                    "Статистическая связь не устраняет риск утечки данных."
                )
        elif field in sensors:
            reason = (
                "Кандидат числового базового решения A по заданию; "
                "корреляция и p-value не определяют состав входов."
            )
        elif field == "operator_log":
            reason = (
                "Сохраняется для текстового контроля, B и C. "
                "Не входит в числовой A по определению эксперимента. "
                "Учитывать повторяемость диагностических формулировок."
            )
        elif field == "manual_id":
            reason = (
                "Проверить связь с equipment и целями, "
                "дублирование информации и пользу на validation; "
                "не исключать только из-за вида идентификатора."
            )
        elif field == "equipment":
            reason = (
                "Установить, обозначает тип или машину; "
                "сравнить числовой A с вариантом A + equipment."
            )
        rows.append({
            "Поле": field,
            "Роль": role,
            "Наблюдаемые факты": facts,
            "Доступность до прогноза": availability,
            "Основание доступности": availability_source,
            "Статус": status,
            "Обоснование / следующая проверка": reason,
            "Решение команды": "",
            "Источник решения": "",
        })

    # 3) Вернуть таблицу; сохранением и решениями управляет тетрадь.
    return pd.DataFrame(rows)


def export_tables(folder: Path, tables: dict[str, pd.DataFrame], manifest: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name == "feature_decisions" and (folder / f"{name}.csv").exists():
            # Сохраняем пользовательские решения отдельно от повторно вычисляемой статистики.
            destination = folder / "feature_evidence_current.csv"
        else:
            destination = folder / f"{name}.csv"
        table.to_csv(destination, index=True, encoding="utf-8-sig")
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
