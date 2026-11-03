import subprocess
import sys
from pathlib import Path

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]


def test_app_missing_config_exits_2(tmp_path):
    run = subprocess.run(
        [sys.executable, str(AI_ENGINE_DIR / "scripts" / "rsafe_app.py"), "--config", str(tmp_path / "nope.yaml")],
        capture_output=True, text=True, timeout=120,
    )

    assert run.returncode == 2 and "nope.yaml" in run.stderr
