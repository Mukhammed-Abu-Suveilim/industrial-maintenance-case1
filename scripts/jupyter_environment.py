"""Настройка Jupyter внутри каталога проекта и его .venv."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def prepare_environment() -> Path:
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise RuntimeError("Запустите скрипт из корня проекта через uv run.")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Проект использует Python 3.11.")
    os.chdir(root)
    locations = {
        "JUPYTER_CONFIG_DIR": root / ".cache/jupyter/config",
        "JUPYTER_DATA_DIR": root / ".cache/jupyter/data",
        "JUPYTER_RUNTIME_DIR": root / "artifacts/jupyter/runtime",
        "IPYTHONDIR": root / ".cache/ipython",
        "MPLCONFIGDIR": root / ".cache/matplotlib",
    }
    for variable, directory in locations.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    subprocess.run([
        sys.executable, "-m", "ipykernel", "install", "--sys-prefix",
        "--name", "industrial-maintenance", "--display-name", "Python 3.11 (.venv, кейс 1)",
    ], check=True, stdout=subprocess.DEVNULL)
    return root
