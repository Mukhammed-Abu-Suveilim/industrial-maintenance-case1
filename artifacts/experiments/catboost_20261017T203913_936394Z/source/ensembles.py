"""Создание двух табличных моделей с проверкой разрешённых полей."""

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def validate_ensemble_features(numeric_columns, categorical_columns):
    """Проверить непустой состав входов обеих табличных моделей."""
    # 1) Разрешить только согласованные поля табличного опыта.
    allowed_numeric = {
        "operating_hours",
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "load_ratio",
        "horizon_days",
    }
    allowed_categorical = {"equipment", "manual_id"}
    if not set(numeric_columns).issubset(allowed_numeric):
        raise ValueError("Числовая группа содержит запрещённые входы.")
    if not set(categorical_columns).issubset(allowed_categorical):
        raise ValueError("Категориальная группа содержит запрещённые входы.")

    # 2) Не допускать пустой список и неоднозначный порядок полей.
    columns = list(numeric_columns) + list(categorical_columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Нужны непустые неповторяющиеся входы.")


def build_forest_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    parameters: dict,
) -> Pipeline:
    """Создать Pipeline без масштабирования числовых показаний."""
    # 1) Проверить входы до создания предобработки.
    validate_ensemble_features(numeric_columns, categorical_columns)
    transformations = [("numeric", "passthrough", numeric_columns)]
    if categorical_columns:
        transformations.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )

    # 2) Вернуть необученные преобразования и классификатор.
    return Pipeline(
        [
            (
                "preprocessing",
                ColumnTransformer(transformations, remainder="drop"),
            ),
            ("classifier", RandomForestClassifier(**parameters)),
        ]
    )


def build_catboost_classifier(
    numeric_columns: list[str],
    categorical_columns: list[str],
    parameters: dict,
) -> CatBoostClassifier:
    """Создать CatBoost для DataFrame со строковыми категориями."""
    # 1) Проверить входы и единый источник определения категорий.
    validate_ensemble_features(numeric_columns, categorical_columns)
    if "cat_features" in parameters:
        raise ValueError("Категории задаются только списком входных полей.")

    # 2) Обучение категориальных преобразований произойдёт внутри fit.
    return CatBoostClassifier(cat_features=categorical_columns, **parameters)
