import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
PROJECT_ROOT = Path("../..").resolve()
if not (PROJECT_ROOT / "configs/04_rubert_tiny2.json").is_file():
    raise RuntimeError("Рабочий каталог должен быть notebooks/experiments.")
if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
    raise RuntimeError("Выберите ядро .venv проекта.")
if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Нужен Python 3.11.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from industrial_maintenance import baseline, data, text_transformer
from industrial_maintenance.comparison import paired_group_bootstrap

DEVICE = "cpu"  # На сервере замените на "cuda" перед Run All.

CONFIG = json.loads(
    (PROJECT_ROOT / "configs/04_rubert_tiny2.json").read_text(encoding="utf-8")
)
CONFIG["execution"]["device"] = DEVICE
hardware = text_transformer.configure_execution(
    DEVICE, CONFIG["execution"]["cpu_threads"]
)
display(hardware)
assert CONFIG["features"] == ["operator_log"]
assert CONFIG["expected_training_runs"] == len(CONFIG["targets"]) == 2
versions = {
    name: version(name)
    for name in [
        "torch", "transformers", "accelerate", "numpy", "pandas",
        "scikit-learn", "mlflow", "scipy",
    ]
}
display(CONFIG)
display(pd.Series(versions, name="Версия"))

dataset_config = json.loads(
    (PROJECT_ROOT / CONFIG["dataset_config"]).read_text(encoding="utf-8")
)
frame, source_manifest = data.load_dataset(PROJECT_ROOT, dataset_config)
membership_path = PROJECT_ROOT / CONFIG["membership_file"]
assert data.file_sha256(membership_path) == CONFIG["membership_sha256"]
split_manifest = json.loads(
    (PROJECT_ROOT / CONFIG["split_manifest"]).read_text(encoding="utf-8")
)
assert split_manifest["protocol"]["sha256"] == source_manifest["sha256"]
assert split_manifest["membership_sha256"] == CONFIG["membership_sha256"]
assert split_manifest["sizes"] == CONFIG["expected_sizes"]
membership = pd.read_csv(membership_path, dtype={"id": "string"})
data.validate_splits(frame, membership)
parts = {}
for split in ["train", "validation"]:
    ids = membership.loc[membership["split"].eq(split), "id"]
    parts[split] = frame.loc[frame["id"].isin(ids)].copy()
    assert len(parts[split]) == CONFIG["expected_sizes"][split]
del frame
train, validation = parts["train"], parts["validation"]
for target, expected_classes in CONFIG["expected_classes"].items():
    assert sorted(train[target].unique().tolist()) == expected_classes
    assert set(validation[target]).issubset(set(expected_classes))
    display(pd.DataFrame({
        "train": train[target].value_counts(),
        "validation": validation[target].value_counts(),
    }))
display(pd.Series(split_manifest["sizes"], name="Размер выборки"))

references = {}
reference_checksums = {}
for reference in CONFIG["references"]:
    batch = PROJECT_ROOT / reference["batch"]
    reference_manifest = json.loads(
        (batch / "manifest.json").read_text(encoding="utf-8")
    )
    assert reference_manifest["status"] == "completed"
    assert not reference_manifest["test_evaluated"]
    assert reference_manifest["source"]["sha256"] == source_manifest["sha256"]
    assert (
        reference_manifest["config"]["membership_sha256"]
        == CONFIG["membership_sha256"]
    )
    assert reference_manifest["split"]["sizes"] == split_manifest["sizes"]
    selected = pd.read_csv(batch / reference["table"])
    selected = selected.loc[selected["variant"].eq(reference["variant"])]
    assert len(selected) == 2
    paths = [batch / "manifest.json", batch / reference["table"]]
    for row in selected.to_dict("records"):
        target = row["target"]
        directory = PROJECT_ROOT / row["run_path"]
        trial = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        assert sorted(trial["classes"]) == CONFIG["expected_classes"][target]
        assert not set(CONFIG["excluded_features"]).intersection(
            trial["features"]
        )
        predictions = pd.read_csv(
            directory / "validation_predictions.csv",
            dtype={"id": "string"}, float_precision="round_trip",
        )
        truth = validation.set_index("id")[target]
        assert predictions.id.is_unique
        assert set(predictions.id) == set(truth.index)
        np.testing.assert_array_equal(
            predictions.set_index("id").loc[truth.index, "y_true"], truth
        )
        references[(reference["name"], target)] = {
            "predictions": predictions, "directory": directory,
        }
        paths.extend([
            directory / "manifest.json",
            directory / "validation_predictions.csv",
        ])
        if target == "failure_within_30d":
            paths.append(directory / "validation_predictions_f2.csv")
    for path in paths:
        reference_checksums[path.relative_to(PROJECT_ROOT).as_posix()] = (
            data.file_sha256(path)
        )
display(pd.DataFrame([
    {"control": name, "target": target, **baseline.classification_metrics(
        item["predictions"], CONFIG["expected_classes"][target], target
    )}
    for (name, target), item in references.items()
]))

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_id"], revision=CONFIG["revision"], trust_remote_code=False
)
length_records = []
for split, part in parts.items():
    texts = text_transformer.text_inputs(part[CONFIG["features"]])
    tokens = tokenizer(texts, truncation=False, padding=False)
    lengths = np.array([len(sequence) for sequence in tokens["input_ids"]])
    length_records.append({
        "split": split, "rows": len(lengths),
        "max_tokens": int(lengths.max()),
        "mean_tokens": float(lengths.mean()),
        "truncated_rows": int((lengths > CONFIG["max_length"]).sum()),
    })
    if split == "train" and (lengths > CONFIG["max_length"]).any():
        raise ValueError(
            "max_length обрезает train; сначала уточните настройку."
        )
token_lengths = pd.DataFrame(length_records)
display(token_lengths)

started = datetime.now(timezone.utc)
batch_id = CONFIG["batch_prefix"] + started.strftime("%Y%m%dT%H%M%S_%fZ")
batch_directory = PROJECT_ROOT / CONFIG["output_root"] / batch_id
batch_directory.mkdir(parents=True, exist_ok=False)


def write_json(path, value):
    """Сохранить JSON в UTF-8 без нечисловых значений NaN."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


logger = logging.getLogger(CONFIG["experiment_name"])
for handler in list(logger.handlers):
    handler.close()
    logger.removeHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
for handler in [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(batch_directory / "run.log", encoding="utf-8"),
]:
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
hardware.update({"platform": platform.platform(), "python": sys.version})
source_directory = batch_directory / "source"
source_directory.mkdir()
source_checksums = {}
for relative in [*CONFIG["source_files"], "pyproject.toml", "uv.lock"]:
    path = PROJECT_ROOT / relative
    destination = source_directory / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    source_checksums[relative] = data.file_sha256(path)
notebook = json.loads(
    (PROJECT_ROOT / CONFIG["notebook"]).read_text(encoding="utf-8")
)
notebook_code = "\n\n".join(
    "".join(cell["source"])
    for cell in notebook["cells"] if cell["cell_type"] == "code"
)
(source_directory / "notebook_code.py").write_text(
    notebook_code, encoding="utf-8"
)
manifest = {
    "status": "running", "batch_id": batch_id, "config": CONFIG,
    "source": source_manifest, "split": split_manifest,
    "libraries": versions, "hardware": hardware,
    "source_checksums": source_checksums,
    "notebook_code_sha256": hashlib.sha256(
        notebook_code.encode("utf-8")
    ).hexdigest(),
    "reference_checksums": reference_checksums,
    "started_at_utc": started.isoformat(),
    "test_evaluated": False,
    "completed_training_runs": 0,
}
write_json(batch_directory / "manifest.json", manifest)
write_json(batch_directory / "config.json", CONFIG)
token_lengths.to_csv(batch_directory / "token_lengths.csv", index=False)
tracking_database = PROJECT_ROOT / CONFIG["tracking_database"]
tracking_database.parent.mkdir(parents=True, exist_ok=True)
tracking_artifacts = PROJECT_ROOT / CONFIG["tracking_artifacts"]
tracking_artifacts.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("sqlite:///" + tracking_database.as_posix())
experiment = mlflow.get_experiment_by_name(CONFIG["experiment_name"])
experiment_id = (
    experiment.experiment_id if experiment else mlflow.create_experiment(
        CONFIG["experiment_name"],
        artifact_location=tracking_artifacts.as_uri(),
    )
)
logger.info("Начало %s; два обучения; %s", batch_id, hardware["device"])

results = []
prediction_tables = {}
reloaded_predictions = {}
for target, primary_metric in CONFIG["targets"].items():
    classes = CONFIG["expected_classes"][target]
    set_seed(CONFIG["seed"])
    datasets = {
        split: text_transformer.TextDataset(
            part[CONFIG["features"]], part[target], classes,
            tokenizer, CONFIG["max_length"],
        )
        for split, part in parts.items()
    }

    def evaluate_epoch(output):
        """Рассчитать метрики эпохи на неизменной validation."""
        # 1) Сопоставить выходы и исходные метки в порядке validation.
        predictions = text_transformer.predictions_from_logits(
            validation.id, validation[target], output.predictions,
            classes, target, CONFIG["binary_threshold"],
        )
        # 2) Использовать те же определения метрик, что и у контролей.
        return baseline.classification_metrics(predictions, classes, target)

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=batch_id + "_" + target,
        tags={"batch_id": batch_id, "target": target, "system": "B"},
    ) as active_run:
        run_id = active_run.info.run_id
        directory = batch_directory / "runs" / run_id
        directory.mkdir(parents=True)
        trial = {
            **manifest, "status": "running", "target": target,
            "classes": classes, "features": CONFIG["features"],
            "run_id": run_id,
        }
        write_json(directory / "manifest.json", trial)
        write_json(directory / "config.json", CONFIG)
        clock = perf_counter()
        try:
            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                model = text_transformer.build_text_classifier(
                    CONFIG["model_id"], CONFIG["revision"], classes
                )
                assert all(
                    parameter.requires_grad for parameter in model.parameters()
                )
                trainable_parameters = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                arguments = TrainingArguments(
                    output_dir=str(directory / "checkpoints"),
                    seed=CONFIG["seed"], data_seed=CONFIG["seed"],
                    disable_tqdm=True, use_cpu=(DEVICE == "cpu"),
                    **CONFIG["training"],
                )
                trainer = Trainer(
                    model=model, args=arguments,
                    train_dataset=datasets["train"],
                    eval_dataset=datasets["validation"],
                    processing_class=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer),
                    compute_metrics=evaluate_epoch,
                )
                logger.info("Обучение %s, MLflow %s", target, run_id)
                clock = perf_counter()
                trainer.train()
                fit_seconds = perf_counter() - clock
                history = trainer.state.log_history
                write_json(directory / "training_summary.json", {
                    "fit_seconds": fit_seconds, "history": history,
                })
                best = text_transformer.select_epoch(
                    history, primary_metric, CONFIG["selection_decimals"]
                )
                best_checkpoint = directory / "checkpoints" / (
                    "checkpoint-" + str(best["step"])
                )
                del trainer, model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                model = AutoModelForSequenceClassification.from_pretrained(
                    best_checkpoint, use_safetensors=True,
                    attn_implementation="eager",
                )
                model.save_pretrained(directory / "model")
                tokenizer.save_pretrained(directory / "model")
                trainer = text_transformer.build_prediction_trainer(
                    model, arguments, tokenizer
                )
                metrics = {}
                for split, part in parts.items():
                    inference_clock = perf_counter()
                    logits = trainer.predict(datasets[split]).predictions
                    predictions = text_transformer.predictions_from_logits(
                        part.id, part[target], logits, classes, target,
                        CONFIG["binary_threshold"],
                    )
                    assert predictions.id.tolist() == part.id.tolist()
                    prediction_tables[(target, split)] = predictions
                    predictions.to_csv(
                        directory / f"{split}_predictions.csv", index=False
                    )
                    metrics[split] = baseline.classification_metrics(
                        predictions, classes, target
                    )
                    metrics[split]["prediction_seconds"] = (
                        perf_counter() - inference_clock
                    )
                restored = AutoModelForSequenceClassification.from_pretrained(
                    directory / "model", attn_implementation="eager"
                )
                del trainer, model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                restored_trainer = text_transformer.build_prediction_trainer(
                    restored, arguments, tokenizer
                )
                reloaded = text_transformer.predictions_from_logits(
                    validation.id, validation[target],
                    restored_trainer.predict(datasets["validation"]).predictions,
                    classes, target, CONFIG["binary_threshold"],
                )
                columns = [f"probability_{label}" for label in classes]
                np.testing.assert_allclose(
                    reloaded[columns],
                    prediction_tables[(target, "validation")][columns],
                    rtol=1e-6, atol=1e-7,
                )
                np.testing.assert_array_equal(
                    reloaded.y_pred,
                    prediction_tables[(target, "validation")].y_pred,
                )
                reloaded_predictions[target] = reloaded
                del restored_trainer, restored
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            warning_text = "\n".join(
                str(item.message) for item in recorded_warnings
            )
            (directory / "warnings.log").write_text(
                warning_text, encoding="utf-8"
            )
            write_json(directory / "history.json", history)
            write_json(directory / "metrics.json", metrics)
            trial.update({
                "status": "completed", "best_epoch": best,
                "fit_seconds": fit_seconds, "reload_verified": True,
                "trainable_parameters": trainable_parameters,
            })
            write_json(directory / "manifest.json", trial)
            mlflow.log_params({
                "model_id": CONFIG["model_id"], "revision": CONFIG["revision"],
                "target": target, "features": "operator_log",
                "seed": CONFIG["seed"], **CONFIG["execution"],
                **CONFIG["training"],
            })
            mlflow.log_metrics({
                f"{split}_{name}": value
                for split, values in metrics.items()
                for name, value in values.items() if value is not None
            })
            mlflow.log_metric("fit_seconds", fit_seconds)
            for row in history:
                mlflow.log_metrics({
                    key: float(value) for key, value in row.items()
                    if isinstance(value, (int, float)) and key != "step"
                }, step=int(row["step"]))
            for path in directory.glob("*.*"):
                mlflow.log_artifact(str(path))
            mlflow.log_artifacts(str(directory / "model"), "model")
            results.append({
                "target": target, "run_id": run_id,
                "run_path": directory.relative_to(PROJECT_ROOT).as_posix(),
                "best_epoch": best["epoch"], "fit_seconds": fit_seconds,
                **metrics["validation"],
            })
            manifest["completed_training_runs"] = len(results)
            write_json(batch_directory / "manifest.json", manifest)
            logger.info("Завершено %s: эпоха %s", target, best["epoch"])
        except Exception as error:
            trial.update({"status": "failed", "error": repr(error)})
            manifest.update({"status": "failed", "error": repr(error)})
            write_json(directory / "manifest.json", trial)
            write_json(batch_directory / "manifest.json", manifest)
            logger.exception("Опыт остановлен без повторного обучения")
            raise
selected_models = pd.DataFrame(results)
selected_models.to_csv(batch_directory / "selected_models.csv", index=False)
display(selected_models)

for row in results:
    target = row["target"]
    classes = CONFIG["expected_classes"][target]
    directory = PROJECT_ROOT / row["run_path"]
    predictions = prediction_tables[(target, "validation")]
    history = json.loads((directory / "history.json").read_text())
    history_frame = pd.DataFrame(history)
    metric = "eval_" + CONFIG["targets"][target]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    history_frame.dropna(subset=["loss"]).plot(
        x="epoch", y="loss", ax=axes[0], label="Train loss"
    )
    history_frame.dropna(subset=["eval_loss"]).plot(
        x="epoch", y="eval_loss", ax=axes[0], label="Validation loss"
    )
    history_frame.dropna(subset=[metric]).plot(
        x="epoch", y=metric, ax=axes[1], label=metric
    )
    figure.tight_layout()
    figure.savefig(directory / "learning_curves.png")
    plt.show()
    plt.close(figure)
    report = pd.DataFrame(classification_report(
        predictions.y_true, predictions.y_pred,
        labels=classes, output_dict=True, zero_division=0,
    )).T
    report.to_csv(directory / "classification_report.csv")
    display(Markdown(f"### {target}"))
    display(report)
    matrix = confusion_matrix(
        predictions.y_true, predictions.y_pred, labels=classes
    )
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        directory / "confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(
        ax=axis, colorbar=False, xticks_rotation=35
    )
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix.png")
    plt.show()
    plt.close(figure)
    diagnosis = predictions.merge(
        validation[["id", "operator_log", "equipment", "severity"]],
        on="id", validate="one_to_one",
    )
    errors = diagnosis.loc[diagnosis.y_true.ne(diagnosis.y_pred)]
    errors.head(CONFIG["error_examples"]).to_csv(
        directory / "error_examples.csv", index=False
    )
    display(Markdown(f"Ошибок validation: **{len(errors)}**."))
    display(errors.head(CONFIG["error_examples"]))
    subgroups = []
    for grouping in ["equipment", "severity"]:
        for value, group in diagnosis.groupby(grouping):
            subgroups.append({
                "grouping": grouping, "value": value, "rows": len(group),
                **baseline.classification_metrics(group, classes, target),
            })
    pd.DataFrame(subgroups).to_csv(directory / "subgroups.csv", index=False)
    if target == "failure_within_30d":
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        PrecisionRecallDisplay.from_predictions(
            predictions.y_true, predictions.probability_1, ax=axes[0]
        )
        RocCurveDisplay.from_predictions(
            predictions.y_true, predictions.probability_1, ax=axes[1]
        )
        figure.tight_layout()
        figure.savefig(directory / "pr_roc.png")
        plt.show()
        plt.close(figure)

binary_target = "failure_within_30d"
binary_row = next(row for row in results if row["target"] == binary_target)
binary_directory = PROJECT_ROOT / binary_row["run_path"]
original = prediction_tables[(binary_target, "validation")]
best_threshold, curve = baseline.select_failure_threshold(
    original.y_true, original.probability_1, CONFIG["selection_decimals"]
)
np.testing.assert_array_equal(
    original.probability_1.ge(best_threshold["threshold"]),
    reloaded_predictions[binary_target].probability_1.ge(
        best_threshold["threshold"]
    ),
)
curve.to_csv(binary_directory / "threshold_curve.csv", index=False)
write_json(binary_directory / "model/decision_rule.json", {
    "positive_class": 1, "threshold": best_threshold["threshold"],
    "rule": "probability_1 >= threshold", "selection_split": "validation",
    "selection_metric": "f2", "refit": False,
})
threshold_metrics = []
for split in ["train", "validation"]:
    predictions = prediction_tables[(binary_target, split)].copy()
    predictions["y_pred"] = predictions.probability_1.ge(
        best_threshold["threshold"]
    ).astype(int)
    predictions.to_csv(
        binary_directory / f"{split}_predictions_f2.csv", index=False
    )
    before = baseline.classification_metrics(
        prediction_tables[(binary_target, split)], [0, 1], binary_target
    )
    after = baseline.classification_metrics(predictions, [0, 1], binary_target)
    assert before["average_precision"] == after["average_precision"]
    assert before["roc_auc"] == after["roc_auc"]
    for name, values in [("default_0.5", before), ("selected_f2", after)]:
        threshold_metrics.append({"split": split, "rule": name, **values})
    prediction_tables[(binary_target, split + "_f2")] = predictions
    if split == "validation":
        pd.DataFrame(confusion_matrix(
            predictions.y_true, predictions.y_pred, labels=[0, 1]
        ), index=[0, 1], columns=[0, 1]).to_csv(
            binary_directory / "confusion_matrix_f2.csv"
        )
pd.DataFrame(threshold_metrics).to_csv(
    binary_directory / "threshold_metrics.csv", index=False
)
operating_points = [
    baseline.precision_recall_operating_point(
        curve, floor, CONFIG["selection_decimals"]
    ) for floor in CONFIG["precision_floors"]
]
pd.DataFrame(operating_points).to_csv(
    binary_directory / "precision_operating_points.csv", index=False
)
display(pd.DataFrame(threshold_metrics))
display(pd.DataFrame(operating_points))
with mlflow.start_run(run_id=binary_row["run_id"]):
    mlflow.log_param("f2_threshold", best_threshold["threshold"])
    mlflow.log_metrics({
        "validation_f2_threshold_" + key: value
        for key, value in baseline.classification_metrics(
            prediction_tables[(binary_target, "validation_f2")],
            [0, 1], binary_target,
        ).items() if value is not None
    })

validation_groups = membership.loc[
    membership["split"].eq("validation"), ["id", "group_id"]
]
comparisons = []
for (name, target), reference in references.items():
    comparisons.append({
        "target": target, "reference": name,
        **paired_group_bootstrap(
            reference["predictions"],
            prediction_tables[(target, "validation")], validation_groups,
            CONFIG["expected_classes"][target], target, **CONFIG["bootstrap"],
        ),
    })
f2_references = {"default_0.5": original}
for name in ["tfidf", "catboost_numeric"]:
    f2_references[name] = pd.read_csv(
        references[(name, binary_target)]["directory"]
        / "validation_predictions_f2.csv",
        dtype={"id": "string"}, float_precision="round_trip",
    )
for name, predictions in f2_references.items():
    comparisons.append({
        "target": binary_target, "reference": name,
        **paired_group_bootstrap(
            predictions, prediction_tables[(binary_target, "validation_f2")],
            validation_groups, [0, 1], binary_target, metric="f2",
            **CONFIG["bootstrap"],
        ),
    })
assert len(comparisons) == 9
comparison_results = pd.DataFrame(comparisons)
comparison_results.to_csv(
    batch_directory / "paired_bootstrap.csv", index=False
)
display(comparison_results)

assert len(results) == CONFIG["expected_training_runs"]
assert data.file_sha256(membership_path) == CONFIG["membership_sha256"]
assert data.file_sha256(
    PROJECT_ROOT / dataset_config["dataset"]
) == dataset_config["sha256"]
for relative, checksum in reference_checksums.items():
    assert data.file_sha256(PROJECT_ROOT / relative) == checksum
for row in results:
    with mlflow.start_run(run_id=row["run_id"]):
        directory = PROJECT_ROOT / row["run_path"]
        for path in directory.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path))
        if row["target"] == binary_target:
            mlflow.log_artifact(str(directory / "model/decision_rule.json"))
    assert mlflow.get_run(row["run_id"]).info.status == "FINISHED"
manifest.update({
    "status": "completed",
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "selected_models": results, "bootstrap_comparisons": len(comparisons),
    "threshold": best_threshold, "test_evaluated": False,
})
write_json(batch_directory / "manifest.json", manifest)
summary = [
    "# RuBERT-tiny2: выполненный опыт B", "",
    "Два независимых обучения, один seed, пять эпох; вход operator_log.",
    "Test не оценивался. Число сравнений: 9.", "",
]
for row in results:
    metric = CONFIG["targets"][row["target"]]
    summary.append(
        f"- {row['target']}: {metric}={row[metric]:.6f}; "
        f"эпоха={row['best_epoch']}; fit={row['fit_seconds']:.2f} с."
    )
summary.extend([
    "", f"F2-порог: {best_threshold['threshold']!r}.", "",
    "Выбор эпохи и порога выполнен на одной validation; оценки условны.",
    "Шесть шаблонов ограничивают перенос на неизвестные формулировки.",
    "Бутстрэп не оценивает вариативность повторного обучения.",
])
(batch_directory / "summary.md").write_text(
    "\n".join(summary), encoding="utf-8"
)
logger.info("Завершено: %s", batch_directory)
display(Markdown("\n".join(summary)))