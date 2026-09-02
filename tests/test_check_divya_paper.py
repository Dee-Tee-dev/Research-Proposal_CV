from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_divya_paper",
    REPO_ROOT / "scripts" / "check_divya_paper.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_text() -> str:
    rows = "\n".join(
        f"| {label} | 50.0% | 50.0% | 50.0% | 50.0% | 84/168 (50.0%) |"
        for label in MODULE.REQUIRED_RESULT_LABELS
    )
    return f"""## 1. Introduction
## 2. Method
### 2.1 Dataset
The subset has 168 unique images and seven images per category–quartile cell.
### 2.2 Models and tasks
### 2.3 Evaluation
Intervals use 2,000 bootstrap samples, seed 2026.
## 3. Results
{rows}
![Result figure](assets/figure.png)
The 95% intervals do not by themselves show that income caused the errors.
"""


class CheckDivyaPaperTests(unittest.TestCase):
    def test_complete_paper_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paper_dir = Path(directory)
            assets = paper_dir / "assets"
            assets.mkdir()
            (assets / "figure.png").write_bytes(b"figure")
            self.assertEqual(MODULE.validate_paper(valid_text(), paper_dir), [])

    def test_pending_value_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paper_dir = Path(directory)
            issues = MODULE.validate_paper(
                valid_text().replace("84/168 (50.0%)", "pending", 1),
                paper_dir,
            )
        self.assertTrue(any("pending" in issue for issue in issues))

    def test_missing_figure_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            issues = MODULE.validate_paper(valid_text(), Path(directory))
        self.assertTrue(any("Missing referenced figure" in issue for issue in issues))

    def test_unchecked_internal_checklist_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paper_dir = Path(directory)
            assets = paper_dir / "assets"
            assets.mkdir()
            (assets / "figure.png").write_bytes(b"figure")
            issues = MODULE.validate_paper(valid_text() + "\n- [ ] verify values\n", paper_dir)
        self.assertTrue(any("unchecked" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
