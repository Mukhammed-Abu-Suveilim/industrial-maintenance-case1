"""Собрать переносимый архив трансформера без отправки файлов."""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def package_transformer(root: Path, destination: Path) -> Path:
    """Скопировать явно необходимые файлы в новый ZIP с SHA-256."""
    # 1) Собрать код, протокол, данные и инструкции по явному списку.
    config_path = root / "configs/04_rubert_tiny2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = json.loads(
        (root / config["dataset_config"]).read_text(encoding="utf-8")
    )
    relative_paths = {
        "pyproject.toml",
        "uv.lock",
        "CODING_STANDARDS.md",
        "PROJECT.md",
        "TERMINOLOGY.md",
        "TRANSFORMER_COLAB.md",
        "COLAB_MCP.md",
        "TRANSFORMER_SERVER.md",
        "scripts/start_jupyter.py",
        "scripts/check_transformer_environment.py",
        "artifacts/preparation/rubert_tiny2_20261018/verification.json",
        "artifacts/preparation/rubert_tiny2_cpu_20261018/verification.json",
        "configs/04_rubert_tiny2.json",
        config["notebook"],
        config["dataset_config"],
        dataset["dataset"],
        config["membership_file"],
        config["split_manifest"],
        "scripts/check_eda.py",
        "scripts/jupyter_environment.py",
        "src/industrial_maintenance/__init__.py",
        *config["source_files"],
    }
    relative_paths.update(
        path.relative_to(root).as_posix()
        for path in (root / "tests").glob("test_*.py")
    )
    relative_paths.update(
        path.relative_to(root).as_posix()
        for path in (root / "src/industrial_maintenance").glob("*.py")
    )

    # 2) Добавить сохранённые прогнозы контролей без старых весов.
    for reference in config["references"]:
        batch = root / reference["batch"]
        table = pd.read_csv(batch / reference["table"])
        selected = table.loc[table["variant"].eq(reference["variant"])]
        if set(selected["target"]) != set(config["targets"]):
            raise ValueError("Не найдены обе цели контроля для архива.")
        relative_paths.add(reference["batch"] + "/manifest.json")
        relative_paths.add(reference["batch"] + "/" + reference["table"])
        for row in selected.to_dict("records"):
            for name in ["manifest.json", "validation_predictions.csv"]:
                relative_paths.add(row["run_path"] + "/" + name)
            if row["target"] == "failure_within_30d":
                relative_paths.add(
                    row["run_path"] + "/validation_predictions_f2.csv"
                )

    # 3) Для проверок формулировок нужны сами текстовые классификаторы.
    if "wording_check" in config:
        relative_paths.update(
            {
                config["wording_check"]["file"],
                config["wording_check"]["review_file"],
                "notebooks/experiments/WORDING_REVIEW.md",
            }
        )
        for relative in [
            "notebooks/experiments/RUBERT_CHECKS_RESULTS.md",
            "artifacts/preparation/seed_checks_20261018/verification.json",
            "artifacts/preparation/seed_checks_20261018/"
            "04_rubert_tiny2_seed42_executed.ipynb",
        ]:
            if (root / relative).is_file():
                relative_paths.add(relative)
        reference = next(
            item for item in config["references"] if item["name"] == "tfidf"
        )
        selected = pd.read_csv(root / reference["batch"] / reference["table"])
        selected = selected.loc[selected["variant"].eq("T0_tfidf")]
        for row in selected.to_dict("records"):
            relative_paths.add(row["run_path"] + "/model.pkl")
            if row["target"] == "failure_within_30d":
                relative_paths.add(
                    row["run_path"] + "/threshold_selection/manifest.json"
                )
        checks = config["stability_checks"]
        batches = [checks["reference_batch"]]
        if checks["reuse_fitted_batch"]:
            batches.append(checks["reuse_fitted_batch"])
        for relative in batches:
            batch = root / relative
            relative_paths.update(
                {
                    relative + "/manifest.json",
                    relative + "/selected_models.csv",
                }
            )
            selected = pd.read_csv(batch / "selected_models.csv")
            for row in selected.to_dict("records"):
                directory = root / row["run_path"]
                for name in [
                    "manifest.json",
                    "metrics.json",
                    "validation_predictions.csv",
                ]:
                    relative_paths.add(row["run_path"] + "/" + name)
                if row["target"] == "failure_within_30d":
                    relative_paths.add(
                        row["run_path"] + "/threshold_metrics.csv"
                    )
                relative_paths.update(
                    path.relative_to(root).as_posix()
                    for path in (directory / "model").glob("*")
                    if path.is_file()
                )

    # 4) Проверить пути и исключить перезапись готового архива.
    paths = {}
    for relative in sorted(relative_paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Некорректный файл архива: {relative}")
        paths[relative] = path
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksums = {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in paths.items()
    }
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for relative, path in paths.items():
            archive.write(path, relative)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(checksums, ensure_ascii=False, indent=2),
        )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = arguments.output or (
        project_root / "artifacts/cloud" / f"rubert_tiny2_{timestamp}.zip"
    )
    print(package_transformer(project_root, output))
