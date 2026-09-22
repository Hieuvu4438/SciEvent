#!/usr/bin/env bash
set -e

WORKSPACE_DIR="/home/haipd/SciEvent/method/TARS-SciEvent"
LOG_DIR="${WORKSPACE_DIR}/artifacts/audits"
LOG_FILE="${LOG_DIR}/setup_environment.log"
SESSION_NAME="tars_setup_download"

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================="
echo "Starting TARS-SciEvent Environment Setup & Pretrained Download"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host: $(uname -a)"
echo "Log file: ${LOG_FILE}"
echo "================================================================="

cd "${WORKSPACE_DIR}"

# 1. Initialize Conda
echo -e "\n>>> [1/6] Initializing Conda..."
source /home/haipd/miniconda3/etc/profile.d/conda.sh

# 2. Create conda environment if not exists
ENV_NAME="scievent-tars"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Reusing it."
else
    echo "Creating conda environment '${ENV_NAME}' with python=3.11..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

# 3. Activate environment
echo -e "\n>>> [2/6] Activating environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"
python --version
which python

# 4. Install PyTorch with CUDA 12.4
echo -e "\n>>> [3/6] Installing PyTorch 2.6.0 with CUDA 12.4 support..."
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 5. Install requirements-tars.txt
echo -e "\n>>> [4/6] Installing dependencies from requirements-tars.txt..."
pip install -r "${WORKSPACE_DIR}/requirements-tars.txt"

# Download required NLTK resources
echo "Downloading NLTK punkt and punkt_tab..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 6. Verify environment & GPU
echo -e "\n>>> [5/6] Verifying PyTorch and GPU availability..."
python - <<'EOF'
import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Transformers version: {transformers.__version__}")
EOF

# 7. Download Pretrained Checkpoints & Tokenizer with HF Transfer
echo -e "\n>>> [6/6] Downloading pretrained checkpoints (ModernBERT-large & BART tokenizer)..."
export HF_HUB_ENABLE_HF_TRANSFER=1
python "${WORKSPACE_DIR}/scripts/download_checkpoints.py" \
    --profile all \
    --output "${WORKSPACE_DIR}/checkpoints" \
    --manifest "${WORKSPACE_DIR}/checkpoints/manifest.json"

# Quick load verification
echo -e "\n>>> Verifying model loading from local snapshot..."
python - <<'EOF'
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "/home/haipd/SciEvent/method/TARS-SciEvent/checkpoints/modernbert-large"
tok_path = "/home/haipd/SciEvent/method/TARS-SciEvent/checkpoints/preprocess-bart-tokenizer"

print(f"Testing local ModernBERT load from: {model_path}")
model = AutoModel.from_pretrained(model_path, local_files_only=True)
print(f"Successfully loaded ModernBERT-large: {type(model).__name__} (params: {sum(p.numel() for p in model.parameters()):,})")

print(f"Testing local BART tokenizer load from: {tok_path}")
tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
print(f"Successfully loaded BART tokenizer: vocab size = {len(tok)}")
EOF

echo -e "\n================================================================="
echo "ALL SETUP & DOWNLOAD TASKS COMPLETED SUCCESSFULLY!"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Manifest: ${WORKSPACE_DIR}/checkpoints/manifest.json"
echo "Cleaning up tmux session '${SESSION_NAME}' as requested..."
echo "================================================================="

# Record initial pip freeze audit
mkdir -p "${WORKSPACE_DIR}/artifacts/audits"
pip freeze > "${WORKSPACE_DIR}/artifacts/audits/pip_freeze.initial.txt"

# Terminate this tmux session as requested
tmux kill-session -t "${SESSION_NAME}" || true
