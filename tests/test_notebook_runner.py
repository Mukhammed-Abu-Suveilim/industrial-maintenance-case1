"""Проверить изоляцию проектного ядра от внешней настройки IPython."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class NotebookRunnerTests(unittest.TestCase):
    def test_tag_runs_only_self_contained_section(self):
        """Выбранный раздел не должен запускать старые ячейки fit."""
        # 1) Создать запрещённую ячейку и самостоятельный раздел.
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            path = Path(temporary) / "section_probe.ipynb"
            untouched = nbformat.v4.new_code_cell(
                'raise RuntimeError("training_must_not_run")'
            )
            section = nbformat.v4.new_code_cell(
                'print("read_only_section")',
                metadata={"tags": ["review"]},
            )
            nbformat.write(
                nbformat.v4.new_notebook(cells=[untouched, section]), path
            )

            # 2) Проверить изоляцию и сохранность исходной ячейки.
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_eda.py",
                    str(path),
                    "--tag",
                    "review",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            executed = nbformat.read(path, as_version=4)
            self.assertIsNone(executed.cells[0].execution_count)
            self.assertEqual(executed.cells[0].source, untouched.source)
            self.assertEqual(
                executed.cells[1].outputs[0].text, "read_only_section\n"
            )
            self.assertEqual(
                executed.metadata.project.section_review, "executed"
            )

    def test_external_kernel_class_does_not_replace_project_kernel(self):
        """Исполнитель должен использовать обычное ядро своего .venv."""
        # 1) Временно воспроизвести недоступный класс ядра среды Colab.
        profile = ROOT / ".cache/ipython/profile_default"
        profile.mkdir(parents=True, exist_ok=True)
        configuration = profile / "ipython_kernel_config.py"
        original = (
            configuration.read_bytes() if configuration.exists() else None
        )
        configuration.write_text(
            "c = get_config()\n"
            "c.IPKernelApp.kernel_class = "
            '"colab_only_missing_kernel.Kernel"\n'
            "c.InteractiveShellApp.extensions = "
            '["colab_only_missing_extension"]\n'
            "c.InteractiveShellApp.reraise_ipython_extension_failures = "
            "True\n",
            encoding="utf-8",
        )

        # 2) Выполнить минимальную тетрадь исполнителем проекта.
        try:
            with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
                path = Path(temporary) / "kernel_probe.ipynb"
                notebook = nbformat.v4.new_notebook(
                    cells=[
                        nbformat.v4.new_code_cell(
                            "import sys\n"
                            f"assert sys.prefix == {sys.prefix!r}\n"
                            "assert type(get_ipython().kernel).__name__ "
                            '== "IPythonKernel"\n'
                            'print("project_kernel_verified")'
                        )
                    ]
                )
                nbformat.write(notebook, path)
                result = subprocess.run(
                    [sys.executable, "scripts/check_eda.py", str(path)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stderr[:500] + result.stderr[-2000:],
                )
                executed = nbformat.read(path, as_version=4)
                self.assertEqual(
                    executed.metadata["project"]["status"], "executed"
                )
                self.assertEqual(
                    executed.cells[0].outputs[0].text,
                    "project_kernel_verified\n",
                )
        finally:
            # 3) Восстановить профиль даже при неудачной проверке.
            if original is None:
                configuration.unlink()
            else:
                configuration.write_bytes(original)
