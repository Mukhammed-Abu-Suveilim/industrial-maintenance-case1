"""Графики для исследовательской тетради; подписи сохраняют имена полей."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_style() -> None:
    sns.set_theme(style="whitegrid", palette="colorblind", font="DejaVu Sans")
    plt.rcParams.update({"figure.dpi": 110, "axes.unicode_minus": False})


def numerical_distributions(frame: pd.DataFrame, columns: list[str]):
    figure, axes = plt.subplots((len(columns) + 2) // 3, 3, figsize=(15, 8), squeeze=False)
    for field, axis in zip(columns, axes.flat):
        sns.histplot(data=frame, x=field, bins=30, ax=axis)
        axis.set_title(field)
        axis.set_ylabel("Число записей")
    for axis in list(axes.flat)[len(columns):]:
        axis.set_visible(False)
    figure.suptitle("Распределения числовых полей: только обучающая выборка")
    figure.tight_layout()
    return figure


def numeric_by_target(frame: pd.DataFrame, columns: list[str], target: str):
    figure, axes = plt.subplots((len(columns) + 2) // 3, 3, figsize=(16, 9), squeeze=False)
    for field, axis in zip(columns, axes.flat):
        sns.boxplot(data=frame, x=target, y=field, showfliers=False, ax=axis)
        axis.tick_params(axis="x", labelrotation=35)
        axis.set_title(field)
    for axis in list(axes.flat)[len(columns):]:
        axis.set_visible(False)
    figure.suptitle(f"Числовые поля по {target}: точки за границами скрыты только на графике")
    figure.tight_layout()
    return figure


def correlation_plot(frame: pd.DataFrame, columns: list[str]):
    correlation = frame[columns].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag", ax=axis)
    axis.set_title("Spearman correlation: числовые поля, обучающая выборка")
    figure.tight_layout()
    return figure


def categorical_panels(frame: pd.DataFrame, columns: list[str]):
    figure, axes = plt.subplots((len(columns) + 1) // 2, 2, figsize=(15, 9), squeeze=False)
    for field, axis in zip(columns, axes.flat):
        counts = frame[field].fillna("<missing>").value_counts()
        sns.barplot(x=counts.to_numpy(), y=counts.index.astype(str), ax=axis, color="#39769d")
        axis.set_title(field)
        axis.set_xlabel("Число записей")
        axis.set_ylabel("")
    for axis in list(axes.flat)[len(columns):]:
        axis.set_visible(False)
    figure.suptitle("Категориальные поля: только обучающая выборка")
    figure.tight_layout()
    return figure


def failure_by_category(frame: pd.DataFrame, field: str):
    table = frame.groupby(field, dropna=False)["failure_within_30d"].agg(["size", "mean"])
    table.columns = ["Число записей", "Доля failure_within_30d = 1"]
    figure, axis = plt.subplots(figsize=(9, max(3.5, len(table) * 0.45)))
    table["Доля failure_within_30d = 1"].plot.barh(ax=axis, color="#39769d")
    axis.set_xlim(0, 1)
    axis.set_title(f"Наблюдаемая доля отказов по {field}; обучающая выборка")
    axis.set_xlabel("Доля, без причинной интерпретации")
    for i, (_, row) in enumerate(table.iterrows()):
        axis.text(min(row.iloc[1] + 0.01, 0.85), i, f"n={int(row.iloc[0])}", va="center")
    figure.tight_layout()
    return figure, table
