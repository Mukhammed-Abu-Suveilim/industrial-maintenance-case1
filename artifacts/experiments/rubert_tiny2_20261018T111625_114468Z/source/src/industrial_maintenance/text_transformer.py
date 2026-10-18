"""Входы, метки и прогнозы текстового трансформера без обучения."""

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)

from industrial_maintenance.baseline import prediction_table


def configure_execution(device: str, cpu_threads: int) -> dict:
    """Выбрать CPU или один CUDA GPU без скрытого переключения."""
    # 1) Отклонить неподдерживаемый режим до начала исследования.
    if device not in {"cpu", "cuda"}:
        raise ValueError("Устройство должно быть cpu или cuda.")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads должен быть положительным целым.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Нужен один процесс, без распределённого запуска.")
    torch.set_num_threads(cpu_threads)
    hardware = {
        "device": device, "cpu_threads": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(), "cuda_build": torch.version.cuda,
        "gpu": None, "gpu_bytes": None,
    }

    # 2) При явном выборе CUDA проверить GPU и реальную операцию.
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Выбран cuda, но CUDA недоступна. Проверьте драйвер, "
                "сборку PyTorch и выделенный GPU; либо выберите cpu."
            )
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Нужен один видимый GPU. Задайте CUDA_VISIBLE_DEVICES "
                "до запуска ядра."
            )
        probe = torch.ones(1, device="cuda") + 1
        torch.cuda.synchronize()
        if probe.item() != 2:
            raise RuntimeError("Проверочная операция CUDA не прошла.")
        hardware.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        })
    return hardware


def text_inputs(features: pd.DataFrame) -> list[str]:
    """Разрешить только исходный непустой operator_log."""
    # 1) Не допустить цели, id или добавленные табличные признаки.
    if list(features.columns) != ["operator_log"] or features.empty:
        raise ValueError("Нужен непустой DataFrame только с operator_log.")
    texts = features["operator_log"].tolist()

    # 2) Отклонить пропуски без молчаливого преобразования текста.
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("operator_log должен содержать непустые строки.")
    return texts


def encode_labels(labels, classes: list) -> np.ndarray:
    """Привести известные метки к позициям выходов модели."""
    # 1) Проверить однозначность порядка классов.
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Нужен непустой список различных классов.")
    positions = {label: index for index, label in enumerate(classes)}
    values = np.asarray(labels)
    if values.ndim != 1 or not np.isin(values, classes).all():
        raise ValueError("Метки должны принадлежать заданным классам.")

    # 2) Сохранить исходный порядок объектов.
    return np.array([positions[label] for label in values], dtype=np.int64)


class TextDataset(Dataset):
    """Токены и отдельные метки; id никогда не передаётся модели."""

    def __init__(self, features, labels, classes, tokenizer, max_length):
        # 1) Проверить состав входов и согласованность меток.
        texts = text_inputs(features)
        self.labels = encode_labels(labels, classes)
        if len(self.labels) != len(texts):
            raise ValueError("Число текстов и меток не совпадает.")
        if not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length должен быть целым числом >= 2.")

        # 2) Применить готовый токенизатор без обучения словаря.
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length, padding=False
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            **{name: values[index] for name, values in self.encodings.items()},
            "labels": int(self.labels[index]),
        }


def build_text_classifier(model_id: str, revision: str, classes: list):
    """Создать кодировщик с новой классификационной частью."""
    # 1) Проверить классы до загрузки весов и закрепить их порядок.
    encode_labels([], classes)
    id_to_label = {index: str(label) for index, label in enumerate(classes)}

    # 2) Загрузить точную версию без исполнения кода из репозитория.
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=len(classes),
        id2label=id_to_label,
        label2id={label: index for index, label in id_to_label.items()},
        problem_type="single_label_classification",
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="eager",
    )


def build_prediction_trainer(model, arguments, tokenizer):
    """Создать исполнитель прогнозов без обучения модели."""
    # 1) Не переносить расписание обучения в отдельный расчёт прогнозов.
    prediction_arguments = replace(
        arguments, eval_strategy="no", save_strategy="no",
        load_best_model_at_end=False,
    )

    # 2) Сохранить batch, устройство и остальные настройки вычислений.
    return Trainer(
        model=model, args=prediction_arguments, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )


def predictions_from_logits(ids, labels, logits, classes, target, threshold):
    """Связать softmax-вероятности с исходными id и порядком классов."""
    # 1) Проверить выходы до преобразования в вероятности.
    values = np.asarray(logits, dtype=float)
    if values.shape != (len(ids), len(classes)):
        raise ValueError("Размер выходов не соответствует id и классам.")
    if not np.isfinite(values).all():
        raise ValueError("Выходы должны быть конечными числами.")

    # 2) Повторно использовать общие правила порога и диагностики.
    return prediction_table(
        ids, labels, softmax(values, axis=1), classes, target, threshold
    )


def select_epoch(history: list[dict], metric: str, decimals: int = 12) -> dict:
    """Выбрать лучшую оценку, при равенстве более раннюю эпоху."""
    # 1) Взять только завершённые оценки эпох, исключив итоговый журнал.
    key = "eval_" + metric
    evaluations = [record for record in history if key in record]
    if not evaluations or any(
        not np.isfinite(record[key]) or not np.isfinite(record["epoch"])
        for record in evaluations
    ):
        raise ValueError("Нет пригодных оценок эпох по основной метрике.")

    # 2) Сравнить округлённые оценки, сохранив исходное значение.
    return min(
        evaluations,
        key=lambda row: (-round(row[key], decimals), row["epoch"]),
    ).copy()
