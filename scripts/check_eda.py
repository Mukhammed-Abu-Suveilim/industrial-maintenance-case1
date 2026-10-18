"""Выполнить тетрадь в новом ядре и сохранить результаты."""

import argparse

import nbformat
from jupyter_environment import prepare_environment
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

if __name__ == "__main__":
    # 1) Выбрать объединённый EDA или явно указанную тетрадь.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook", nargs="?", default="notebooks/eda/EDA-16-10.ipynb"
    )
    parser.add_argument(
        "--tag",
        help="Выполнить только самодостаточные ячейки с указанным тегом.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Максимальное время одной ячейки в секундах (по умолчанию 180).",
    )
    arguments = parser.parse_args()
    if arguments.timeout < 1:
        parser.error("--timeout должен быть положительным числом.")
    root = prepare_environment()
    path = (root / arguments.notebook).resolve()
    if not path.is_file() or path.suffix != ".ipynb":
        raise ValueError(f"Не найдена исследовательская тетрадь: {path}")

    # 2) Проверить структуру и запустить чистое ядро рядом с тетрадью.
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    original = notebook
    selected = None
    if arguments.tag:
        selected = [
            index
            for index, cell in enumerate(original.cells)
            if arguments.tag in cell.metadata.get("tags", [])
        ]
        if not selected:
            raise ValueError("Не найдены ячейки с указанным тегом.")
        notebook = nbformat.v4.new_notebook(
            cells=[original.cells[index] for index in selected],
            metadata=original.metadata,
        )
    client = NotebookClient(
        notebook,
        timeout=arguments.timeout,
        kernel_name="industrial-maintenance",
        # Colab задаёт класс и расширение в системном профиле IPython.
        extra_arguments=[
            "--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel",
            "--InteractiveShellApp.extensions=[]",
        ],
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError):
        if selected is not None:
            for index, cell in zip(selected, notebook.cells, strict=True):
                original.cells[index] = cell
            original.metadata.setdefault("project", {})[
                f"section_{arguments.tag}"
            ] = "failed"
        else:
            original.metadata.setdefault("project", {})["status"] = "failed"
        nbformat.write(original, path)
        raise

    # 3) Сохранить фактические результаты выбранной тетради.
    if selected is not None:
        for index, cell in zip(selected, notebook.cells, strict=True):
            original.cells[index] = cell
        original.metadata.setdefault("project", {})[
            f"section_{arguments.tag}"
        ] = "executed"
    else:
        original.metadata.setdefault("project", {})["status"] = "executed"
    nbformat.write(original, path)
    print(f"Тетрадь выполнена: {path.relative_to(root)}")
