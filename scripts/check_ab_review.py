"""Проверить передачу A/B и сохранность исходных опытов без fit."""

import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    """Вычислить SHA-256 файла без изменения содержимого."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    # 1) Проверить исходные данные, разбиение и прежние результаты.
    config = json.loads((ROOT / "configs/ab_review.json").read_text("utf-8"))
    destination = ROOT / config["output_root"]
    data_config = json.loads(
        (ROOT / config["dataset_config"]).read_text("utf-8")
    )
    assert file_sha256(ROOT / data_config["dataset"]) == data_config["sha256"]
    assert (
        file_sha256(ROOT / config["membership"]) == config["membership_sha256"]
    )
    protected = json.loads(
        (destination / "protected_source_files.json").read_text("utf-8")
    )
    changed = [
        path
        for path, digest in protected.items()
        if file_sha256(ROOT / path) != digest
    ]
    assert not changed, changed
    original_cells = {}
    for path in (destination / "source_notebooks").glob("*.ipynb"):
        original = nbformat.read(path, as_version=4)
        current = nbformat.read(
            ROOT / "notebooks/experiments" / path.name, as_version=4
        )
        nbformat.validate(current)
        assert current.cells[: len(original.cells)] == original.cells
        original_cells[path.name] = len(original.cells)
        for cell in current.cells:
            if (
                set(cell.metadata.get("tags", []))
                & {"ab-review", "ab-summary"}
                and cell.cell_type == "code"
            ):
                assert cell.execution_count is not None
                assert not any(
                    output.get("output_type") == "error"
                    for output in cell.get("outputs", [])
                )

    # 2) Проверить изоляцию данных и отдельное хранение целевых меток.
    membership = pd.read_csv(
        ROOT / config["membership"], dtype={"id": "string"}
    )
    handoff = destination / "handoff"
    forbidden = {
        "required_route",
        "severity",
        "fault_type",
        "failure_within_30d",
    }
    for split in ["train", "validation"]:
        expected = set(membership.loc[membership["split"].eq(split), "id"])
        features = pd.read_csv(
            handoff / (split + "_features.csv"), dtype={"id": "string"}
        )
        labels = pd.read_csv(
            handoff / (split + "_labels.csv"), dtype={"id": "string"}
        )
        assert features["id"].is_unique and set(features["id"]) == expected
        assert labels["id"].is_unique and set(labels["id"]) == expected
        assert not forbidden.intersection(features.columns)
    for path in handoff.glob("models/*/*predictions_without_labels.csv"):
        table = pd.read_csv(path, dtype={"id": "string"})
        assert "y_true" not in table and "fault_type" not in table
        assert "failure_within_30d" not in table
        split = path.name.split("_", 1)[0]
        expected = set(membership.loc[membership["split"].eq(split), "id"])
        assert table["id"].is_unique and set(table["id"]) == expected
    errors = pd.read_csv(
        destination / "summary/reviewed_errors_annotated.csv",
        dtype={"id": "string"},
    )
    assert len(errors) == 30 and errors["id"].is_unique
    assert errors["review_status"].eq("agent_content_checked").all()
    assert not errors["human_reviewed"].any()

    # 3) Проверить байты копий моделей и записи локальной проверки.
    manifest = json.loads((handoff / "manifest.json").read_text("utf-8"))
    assert len(manifest["models"]) == 6
    for model in manifest["models"]:
        for path in (handoff / model["bundle_path"]).rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(handoff / model["bundle_path"])
            source = ROOT / model["run_path"] / relative
            if relative.parts[0] == "model" or relative.name in {
                "model.pkl",
                "model_f2.pkl",
            }:
                assert file_sha256(path) == file_sha256(source)
    bundle_check = json.loads(
        (destination / "summary/handoff_verification.json").read_text("utf-8")
    )
    assert len(bundle_check["models"]) == 6
    assert all(
        row["probabilities_match"] and row["labels_match"]
        for row in bundle_check["models"]
    )
    assert (
        file_sha256(destination / "ab_handoff.zip")
        == bundle_check["archive_sha256"]
    )
    assert not list(destination.glob("**/test_predictions*.csv"))

    # 4) Сохранить фактически подтверждённое состояние этой проверки.
    verification = {
        "status": "passed",
        "research_fit_count": 0,
        "test_evaluated": False,
        "protected_files": len(protected),
        "protected_files_changed": changed,
        "original_notebook_cells_unchanged": original_cells,
        "reviewed_error_ids": 30,
        "handoff_models_verified": 6,
        "handoff_prediction_files": len(
            list(handoff.glob("models/*/*predictions_without_labels.csv"))
        ),
        "selected_models": len(config["models"]),
        "source_dataset_and_split_sha256_match": True,
        "archive_sha256": bundle_check["archive_sha256"],
        "archive_bytes": bundle_check["archive_bytes"],
        "external_machine_tested": False,
    }
    (destination / "verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, ensure_ascii=False))
