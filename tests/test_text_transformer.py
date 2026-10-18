"""Синтетические проверки трансформера без скачивания весов."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    from industrial_maintenance.baseline import classification_metrics
    from industrial_maintenance.text_transformer import (
        TextDataset,
        build_prediction_trainer,
        configure_execution,
        encode_labels,
        predictions_from_logits,
        select_epoch,
        text_inputs,
    )
else:
    torch = None


@unittest.skipIf(torch is None, "Нужен uv sync --extra transformer.")
class TransformerTests(unittest.TestCase):
    def setUp(self):
        # 1) Создать отдельный синтетический словарь русского текста.
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        vocabulary = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "насос", "шумит", "стабилен",
        ]
        vocabulary_path = self.root / "vocab.txt"
        vocabulary_path.write_text("\n".join(vocabulary), encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocabulary_path))
        self.features = pd.DataFrame(
            {"operator_log": ["насос стабилен", "насос шумит"]}
        )

        # 2) Ограничить размер кодировщика и число потоков.
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = BertForSequenceClassification(
            BertConfig(
                vocab_size=len(vocabulary), hidden_size=8,
                num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=16, num_labels=2,
                hidden_dropout_prob=0, attention_probs_dropout_prob=0,
                attn_implementation="eager",
            )
        )

    def test_only_original_text_is_allowed(self):
        # 1) Не менять исходные русские строки и не расширять входы.
        self.assertEqual(
            text_inputs(self.features), self.features.iloc[:, 0].tolist()
        )
        for column in ["severity", "required_route", "id", "fault_type"]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                text_inputs(self.features.assign(**{column: "forbidden"}))

        # 2) Не преобразовывать пропуски в текстовые значения.
        for value in [None, "", " ", 12]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                text_inputs(pd.DataFrame({"operator_log": [value]}))

    def test_cpu_is_explicit_even_when_cuda_is_present(self):
        # 1) CPU не должен обращаться к свойствам доступного GPU.
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_name",
                         side_effect=AssertionError),
        ):
            hardware = configure_execution("cpu", 1)
        self.assertEqual(hardware["device"], "cpu")
        self.assertIsNone(hardware["gpu"])
        self.assertEqual(hardware["cpu_threads"], 1)

        # 2) Неверный режим или число потоков отклоняются явно.
        for device, threads in [("auto", 1), ("cpu", 0), ("cpu", True)]:
            with self.subTest(device=device), self.assertRaises(ValueError):
                configure_execution(device, threads)

    def test_cuda_does_not_silently_fall_back_to_cpu(self):
        # 1) Недоступный GPU не должен подменяться CPU.
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "CUDA недоступна"),
        ):
            configure_execution("cuda", 1)

        # 2) Несколько GPU требуют явного выбора до запуска ядра.
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            self.assertRaisesRegex(RuntimeError, "один видимый GPU"),
        ):
            configure_execution("cuda", 1)

    def test_tokenizer_is_fixed_and_unknown_words_work(self):
        # 1) Применить готовый словарь к явно переданным входам.
        before = self.tokenizer.get_vocab().copy()
        train = TextDataset(
            self.features, [0, 1], [0, 1], self.tokenizer, 128
        )
        unknown = TextDataset(
            pd.DataFrame({"operator_log": ["насос неизвестно"]}),
            [1], [0, 1], self.tokenizer, 128,
        )

        # 2) Проверить русские токены, неизвестное слово и входы.
        self.assertIn(before["насос"], train[0]["input_ids"])
        self.assertIn(before["[UNK]"], unknown[0]["input_ids"])
        self.assertNotIn("id", train[0])
        self.assertEqual(before, self.tokenizer.get_vocab())

    def test_probabilities_follow_class_order_and_ids(self):
        # 1) Проверить нестандартный порядок классов и исходные id.
        table = predictions_from_logits(
            ["b", "a"], [1, 0], [[3, 0], [0, 3]], [1, 0],
            "failure_within_30d", 0.5,
        )
        self.assertEqual(table.id.tolist(), ["b", "a"])
        self.assertEqual(table.y_pred.tolist(), [1, 0])
        np.testing.assert_allclose(table.iloc[:, 3:].sum(axis=1), 1)
        diagnosis = predictions_from_logits(
            ["c"], ["normal"], [[0, 1, 3]],
            ["seal_leak", "bearing_wear", "normal"], "fault_type", 0.5,
        )
        self.assertEqual(diagnosis.y_pred.tolist(), ["normal"])

        # 2) Отклонить нечисловые выходы и неизвестные эталонные метки.
        with self.assertRaises(ValueError):
            predictions_from_logits(
                ["a"], [0], [[np.nan, 1]], [0, 1],
                "failure_within_30d", 0.5,
            )
        with self.assertRaises(ValueError):
            encode_labels(["unknown"], ["normal"])

    def test_epoch_selection_rounds_and_prefers_earlier_epoch(self):
        # 1) Сравнить оценки, отличающиеся за пределами точности выбора.
        history = [
            {"epoch": 1, "eval_macro_f1": 0.8},
            {"epoch": 2, "eval_macro_f1": 0.8000000000001},
            {"epoch": 3, "train_loss": 0.1},
        ]
        self.assertEqual(select_epoch(history, "macro_f1")["epoch"], 1)

        # 2) Явно отклонить отсутствие пригодных оценок.
        with self.assertRaises(ValueError):
            select_epoch([], "macro_f1")

    def test_full_encoder_receives_gradients_and_reload_matches(self):
        # 1) Рассчитать синтетическую потерю и градиенты кодировщика.
        dataset = TextDataset(
            self.features, [0, 1], [0, 1], self.tokenizer, 128
        )
        collator = DataCollatorWithPadding(self.tokenizer)
        batch = collator([dataset[0], dataset[1]])
        self.model(**batch).loss.backward()
        gradient = self.model.bert.embeddings.word_embeddings.weight.grad
        self.assertGreater(float(gradient.abs().sum()), 0)

        # 2) Проверить Trainer и сохранение без исследовательского fit.
        arguments = TrainingArguments(
            output_dir=str(self.root / "output"), use_cpu=True,
            report_to="none", per_device_eval_batch_size=2,
        )
        trainer = Trainer(
            model=self.model, args=arguments,
            processing_class=self.tokenizer, data_collator=collator,
        )
        before = trainer.predict(dataset).predictions
        self.model.save_pretrained(self.root / "model")
        restored = BertForSequenceClassification.from_pretrained(
            self.root / "model", attn_implementation="eager"
        )
        restored_trainer = Trainer(
            model=restored, args=arguments,
            processing_class=self.tokenizer, data_collator=collator,
        )
        after = restored_trainer.predict(dataset).predictions
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(before.argmax(1), after.argmax(1))

    def test_prediction_after_epoch_training_does_not_require_evaluation(self):
        # 1) Воспроизвести настройки завершённого обучения по эпохам.
        arguments = TrainingArguments(
            output_dir=str(self.root / "prediction"), use_cpu=True,
            eval_strategy="epoch", save_strategy="epoch",
            per_device_eval_batch_size=2, report_to="none",
            disable_tqdm=True,
        )
        dataset = TextDataset(
            self.features, [0, 1], [0, 1], self.tokenizer, 128
        )

        # 2) Прогнозировать без eval_dataset и без повторного fit.
        with patch.object(Trainer, "train", side_effect=AssertionError):
            predictor = build_prediction_trainer(
                self.model, arguments, self.tokenizer
            )
            before = predictor.predict(dataset).predictions
            self.model.save_pretrained(self.root / "prediction_model")
            restored = BertForSequenceClassification.from_pretrained(
                self.root / "prediction_model", attn_implementation="eager"
            )
            restored_predictor = build_prediction_trainer(
                restored, arguments, self.tokenizer
            )
            after = restored_predictor.predict(dataset).predictions

        # 3) Сохранить исходные настройки и совпадение прогнозов.
        self.assertEqual(arguments.eval_strategy, "epoch")
        self.assertEqual(predictor.args.eval_strategy, "no")
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(before.argmax(1), after.argmax(1))

    def test_synthetic_training_saves_a_selectable_checkpoint(self):
        # 1) Использовать тот же протокол, сократив синтетический опыт.
        config = json.loads(
            (Path(__file__).resolve().parents[1]
             / "configs/04_rubert_tiny2.json").read_text(encoding="utf-8")
        )
        parameters = {
            **config["training"], "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
        }
        dataset = TextDataset(
            self.features, [0, 1], [0, 1], self.tokenizer, 128
        )
        arguments = TrainingArguments(
            output_dir=str(self.root / "training"), use_cpu=True,
            disable_tqdm=True, **parameters,
        )

        # 2) Выполнить один шаг только на синтетических данных.
        def evaluate(output):
            # 1) Сопоставить выходы с синтетическими id.
            predictions = predictions_from_logits(
                ["synthetic_a", "synthetic_b"], [0, 1],
                output.predictions, [0, 1], "failure_within_30d", 0.5,
            )
            # 2) Рассчитать фактические метрики для проверки Trainer.
            return classification_metrics(
                predictions, [0, 1], "failure_within_30d"
            )

        trainer = Trainer(
            model=self.model, args=arguments, train_dataset=dataset,
            eval_dataset=dataset, processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=evaluate,
        )
        trainer.train()
        selected = select_epoch(trainer.state.log_history, "macro_f1")
        checkpoint = self.root / "training" / (
            "checkpoint-" + str(selected["step"])
        )
        self.assertTrue((checkpoint / "model.safetensors").is_file())
        self.assertEqual(selected["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
