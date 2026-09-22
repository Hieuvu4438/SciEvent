import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

OFFICIAL_EVALUATOR = WORKSPACE / "vendor/SciEvent/baselines/ONEIE/EM_overlap_eval.py"
DATA_DIR = WORKSPACE / "data" / "official"
BACKBONE = WORKSPACE / "checkpoints" / "modernbert-large"
