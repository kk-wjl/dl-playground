from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def data_dir(name: str) -> Path:
    return DATA_ROOT / name


def checkpoint_path(experiment: str, filename: str) -> Path:
    path = ARTIFACTS_ROOT / "checkpoints" / experiment / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(experiment: str) -> Path:
    path = ARTIFACTS_ROOT / "runs" / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_dir(experiment: str) -> Path:
    path = ARTIFACTS_ROOT / "figures" / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path
