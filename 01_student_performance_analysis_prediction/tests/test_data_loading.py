from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_raw_data_file_is_documented_as_downloaded():
    # This test checks the repository layout only; dataset download is a separate step.
    assert (ROOT / "src" / "download_data.py").exists()
    assert (ROOT / "README.md").exists()


def test_target_name_in_example_contract():
    # The project contract uses G3 as the final-grade target.
    assert "G3" in (ROOT / "README.md").read_text(encoding="utf-8")
