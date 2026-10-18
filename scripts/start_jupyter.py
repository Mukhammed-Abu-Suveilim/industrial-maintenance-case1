"""Запуск JupyterLab с ядром .venv и открытием тетради EDA."""

import subprocess
import sys

from jupyter_environment import prepare_environment

if __name__ == "__main__":
    # 1) Подготовить ядро и служебные каталоги внутри проекта.
    root = prepare_environment()
    default_url = "/lab/tree/notebooks/eda/EDA-16-10.ipynb"

    # 2) Открыть объединённую тетрадь с параметрами пользователя.
    arguments = [
        sys.executable,
        "-m",
        "jupyterlab",
        "--ip=127.0.0.1",
        f"--ServerApp.root_dir={root}",
        f"--ServerApp.default_url={default_url}",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(arguments))
