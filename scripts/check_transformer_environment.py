"""Проверить окружение и настоящий кодировщик без обучения на CSV."""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_maintenance import data, text_transformer  # noqa: E402

if __name__ == "__main__":
    # 1) Проверить изолированное окружение и явный выбор устройства.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    arguments = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Нужно окружение .venv: используйте uv run.")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Нужен Python 3.11.")
    config = json.loads(
        (ROOT / "configs/04_rubert_tiny2.json").read_text(encoding="utf-8")
    )
    hardware = text_transformer.configure_execution(
        arguments.device, config["execution"]["cpu_threads"]
    )

    # 2) Проверить исходные данные и разбиение без оценивания test.
    dataset = json.loads(
        (ROOT / config["dataset_config"]).read_text(encoding="utf-8")
    )
    for relative, expected in [
        (dataset["dataset"], dataset["sha256"]),
        (config["membership_file"], config["membership_sha256"]),
    ]:
        if data.file_sha256(ROOT / relative) != expected:
            raise ValueError(f"Не совпадает SHA-256: {relative}")

    # 3) Проверить настоящие веса на явно синтетическом тексте.
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"],
        trust_remote_code=False,
    )
    checks = []
    for target, classes in config["expected_classes"].items():
        set_seed(config["seed"])
        model = text_transformer.build_text_classifier(
            config["model_id"], config["revision"], classes
        ).to(arguments.device)
        encoded = tokenizer(
            "Синтетический насос работает", return_tensors="pt"
        ).to(arguments.device)
        model.eval()
        with torch.no_grad():
            logits = model(**encoded).logits.cpu().numpy()
        if logits.shape != (1, len(classes)) or not np.isfinite(logits).all():
            raise RuntimeError(f"Неверные выходы: {target}")
        checks.append({"target": target, "outputs": len(classes)})
        del model
        if arguments.device == "cuda":
            torch.cuda.empty_cache()

    # 4) Сохранить проверку отдельно от исследовательских результатов.
    report = {
        "status": "passed", "research_training_runs": 0,
        "test_evaluated": False, "hardware": hardware,
        "platform": platform.platform(), "python": sys.version,
        "versions": {
            name: version(name)
            for name in ["torch", "transformers", "accelerate"]
        },
        "synthetic_forward_checks": checks,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = ROOT / "artifacts/preparation" / (
        "transformer_environment_" + timestamp + ".json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(destination.relative_to(ROOT))
