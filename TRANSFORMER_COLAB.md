# RuBERT-tiny2: подготовка и выполнение в Colab

**Исторический сценарий.** Пользователь выбрал локальную проверку CPU
и последующий перенос на A100. Актуальная инструкция:
[TRANSFORMER_SERVER.md](TRANSFORMER_SERVER.md). Текущая тетрадь по
умолчанию использует CPU; приведённый ниже старый исполнитель Colab
не является инструкцией запуска актуальной версии на GPU.

Альтернативное соединение для Codex в VS Code: [Colab MCP](COLAB_MCP.md).
Сервер подключён; выполнение ячеек и GPU Tesla T4 подтверждены.

На 2026-10-18 пользователь разрешил загрузку подготовленного
архива в подключённый Colab и сохранение результатов на Google Drive.
Архив вручную загружен пользователем; размер, SHA-256 и все 50 файлов
проверены. Drive подключён. .venv Python 3.11.16 синхронизировано
через uv; пройдены 43 облачных теста и Ruff изменённых файлов.
В scripts/check_eda.py исправлен конфликт системного профиля Colab:
явно заданы обычный IPythonKernel и пустой список внешних расширений.
Первый процесс остановился до обучения; после проверки свежего ядра
выполнение начато заново. Модель отказа прошла все пять эпох, затем
выполнение остановилось при создании отдельного Trainer для прогнозов:
расписание eval_strategy=epoch требовало отсутствующий eval_dataset.
Checkpoints сохранены на Drive; диагноз не обучался, полный опыт
и сравнения не завершены. Ошибка исправлена в локальной версии через
отдельные настройки прогнозирования, без повторного fit. Новый запуск
в Colab не выполнялся; текущих процессов обучения там нет.
Локальный упаковщик сам ничего не отправляет в сеть.

## Состав опыта

- Тетрадь: [04_rubert_tiny2.ipynb](notebooks/experiments/04_rubert_tiny2.ipynb).
- Настройки: [04_rubert_tiny2.json](configs/04_rubert_tiny2.json).
- Кодировщик: `cointegrated/rubert-tiny2`, revision
  `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`.
- Только исходный `operator_log`, без дополнительных полей.
- Две независимые модели: `failure_within_30d` и `fault_type`.
- Полное обучение кодировщика и классификационной части; cross-entropy
  без весов классов. AdamW, learning_rate=0.00005, weight_decay=0.01,
  линейное расписание и 10% warm-up, ограничение нормы градиента 1.0.
- Пять эпох без ранней остановки, train batch=32, evaluation batch=64,
  max_length=128, динамическое дополнение последовательностей, FP32,
  seed=42. Готовый токенизатор не переобучается.
- Выбор эпохи по validation AP / macro-F1, при равенстве до 12 знаков
  более ранняя эпоха. Поиска гиперпараметров нет: **два обучения**.
- Порог отказа по validation F2 после выбора эпохи; диагноз — argmax,
  включая normal. Также диагностика precision ≥ 70%/80%.
- Сравнения: TF-IDF, числовой CatBoost, prior из закреплённых запусков;
  шесть AP/macro-F1 и три F2, по 2 000 парных групповых повторов.
- Ни переобучения на train+validation, ни оценки test нет.

Первый опыт использует один seed. Бутстрэп характеризует выборочную
неопределённость фиксированных прогнозов, а не разброс повторных обучений.
Выбор эпохи и порога на одной validation может завышать оценки.
Шесть общих шаблонов ограничивают выводы о новых формулировках.

## Локальная подготовка

Зависимости PyTorch 2.9.1, Transformers 4.57.6 и Accelerate 1.12.0
объявлены в optional dependency group `transformer`. Основные зависимости
классических моделей остаются отдельно. После обычного `uv sync` группа
не устанавливается; для работы с трансформером нужна команда с extra:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.cache/uv'
uv sync --extra transformer
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/package_transformer.py
```

Команда упаковки создаёт новый `artifacts/cloud/rubert_tiny2_<UTC>.zip`.
Архив содержит исходный CSV, eda_v1, код, настройки, uv.lock, тетрадь,
малые тесты и сохранённые прогнозы выбранных контролей. Старые модели,
база MLflow, .venv, секреты и остальной рабочий каталог не включаются.
`bundle_manifest.json` перечисляет SHA-256 всех включённых файлов.
Архив предназначен для личной тетради Colab.
Для текущего запуска пользователь вручную загрузил архив через панель
файлов Colab. Colab MCP подтвердил размер, SHA-256 и manifest файлов.

## Запуск в Colab

Следующая последовательность подготовлена для Colab; фактическое
выполнение в облаке ещё не проверено. Нужны аккаунт Google, доступный
GPU и место на Drive для checkpoints. Доступность GPU не гарантирована:
[официальные ограничения Colab](https://research.google.com/colaboratory/faq.html).

1. Создать личную тетрадь Colab и выбрать GPU через Runtime → Change
   runtime type. Тетрадь Colab служит исполнителем; исследование остаётся
   целиком в единственной проектной тетради `04_rubert_tiny2.ipynb`.
2. Выполнить приведённую ниже ячейку, выбрать ровно один подготовленный
   ZIP и подтвердить подключение своего Google Drive.
3. Дождаться завершения. Не выбирать другой Python из предустановленного
   окружения Colab: настоящий исследовательский kernel создаётся из .venv.

```python
import hashlib
import io
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive, files

# 1) Загрузить подготовленный архив в новую рабочую копию.
uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Выберите ровно один ZIP проекта.")
root = Path("/content/industrial-maintenance")
root.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
    for entry in archive.infolist():
        if not (root / entry.filename).resolve().is_relative_to(root):
            raise ValueError("Недопустимый путь в ZIP.")
    archive.extractall(root)
checksums = json.loads((root / "bundle_manifest.json").read_text())
for relative, expected in checksums.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Изменён файл архива: {relative}")

# 2) Хранить результаты и checkpoints на Drive до завершения сессии.
drive.mount("/content/drive")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
persistent = Path("/content/drive/MyDrive/LLMs-transformer") / stamp
persistent.mkdir(parents=True, exist_ok=False)
shutil.copytree(root / "artifacts", persistent / "artifacts")
(root / "artifacts").rename(root / "reference_artifacts_copy")
(root / "artifacts").symlink_to(persistent / "artifacts", target_is_directory=True)

# 3) Использовать uv и отдельное .venv Python 3.11 с GPU-зависимостями.
uv = shutil.which("uv")
if uv is None:
    installer = Path("/content/install-uv.sh")
    urllib.request.urlretrieve("https://astral.sh/uv/install.sh", installer)
    subprocess.run(["sh", str(installer)], check=True)
    uv = str(Path.home() / ".local/bin/uv")
environment = os.environ.copy()
environment["UV_CACHE_DIR"] = "/content/uv-cache"
environment["HF_HOME"] = "/content/hf-cache"
environment["CUDA_VISIBLE_DEVICES"] = "0"
subprocess.run(
    [uv, "sync", "--frozen", "--extra", "transformer", "--python", "3.11"],
    cwd=root, env=environment, check=True,
)

# 4) Выполнить настоящую тетрадь в свежем ядре .venv и сохранить её.
try:
    subprocess.run(
        [uv, "run", "--no-sync", "python", "scripts/check_eda.py",
         "notebooks/experiments/04_rubert_tiny2.ipynb", "--timeout", "3600"],
        cwd=root, env=environment, check=True,
    )
finally:
    shutil.copy2(
        root / "notebooks/experiments/04_rubert_tiny2.ipynb",
        persistent / "04_rubert_tiny2.ipynb",
    )
print("Результаты и тетрадь:", persistent)
```

Эта ячейка не устанавливает пакеты проекта в глобальный Python Colab.
При прерывании полные завершённые эпохи
сохраняются на Drive; автоматического повторного обучения нет. Не запускайте
Run All повторно для «продолжения»: это создаст новый опыт и два обучения.
Возобновление прерванного опыта следует оформить отдельно и явно.

## Возврат результатов

Сохранить с Drive выполненную тетрадь и новый каталог
`artifacts/experiments/rubert_tiny2_<UTC>/`, не заменяя старые запуски.
В нём находятся конфигурация, manifest, сводка, исходный код, checkpoints,
модели, токенизатор, train/validation-прогнозы, кривые и сравнения.

Порог находится в `runs/<run_id>/model/decision_rule.json`. Загрузка
через from_pretrained не применяет его автоматически: сначала softmax,
затем `probability_1 >= threshold`. Порядок выходных классов сохранён
в model/config.json и manifest. Для диагноза применяется argmax.

MLflow использует отдельную SQLite-базу
`artifacts/tracking/transformer/mlflow.db` и отдельный каталог файлов.
Они сохраняются на Drive. Абсолютные адреса файлов MLflow относятся
к облачной рабочей копии; простое копирование SQLite не переносит эти
адреса в Windows. JSON/CSV и каталоги моделей остаются самостоятельными
результатами. Автоматического объединения с прежней базой MLflow нет.

После возврата проверить сохранение/загрузку, число выполненных обучений,
согласованность id и SHA-256. Только после этого фиксировать метрики
в PROJECT.md. Подготовленная тетрадь сама по себе не является результатом
обучения. Числовой CatBoost и текстовый трансформер используют разные
входы, поэтому их разность нельзя целиком приписать архитектуре.
