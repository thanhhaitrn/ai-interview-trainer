"""Regression tests for top-level utility scripts."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ScriptEntrypointTestCase(unittest.TestCase):
    """Exercise non-legacy top-level script entrypoints."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_app_main_prints_workflow(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "app.main",
                "show-workflow",
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("flowchart TD", result.stdout)
        self.assertIn("Start interview", result.stdout)
        self.assertIn("Final report", result.stdout)

    def test_app_main_normalizes_resume_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "app.main",
                    "normalize-resume",
                    str(
                        self.project_root
                        / "data"
                        / "resumes"
                        / "parsed"
                        / "resume1_parsed.json"
                    ),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            output_path = output_dir / "resume1_parsed_llm.json"

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())
            self.assertIn("Saved LLM-ready resume JSON", result.stdout)


if __name__ == "__main__":
    unittest.main()
