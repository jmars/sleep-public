#!/bin/bash
# Bring-up for a fresh Lambda GPU box (E5). Idempotent; every step echoes its exit code.
#
# Two image-level defects bit the E4 session, both at a distance from their symptom:
#  * the image's system Pillow has no PIL.Image.Resampling, so transformers' lazy module
#    init raises inside image_utils and EVERY name fails to resolve (peft's
#    "cannot import name BloomPreTrainedModel from transformers" is the first casualty);
#  * the image's numpy 2.x breaks the system scipy with a dtype-size ABI error.
# So pillow is upgraded and numpy is pinned BEFORE anything imports transformers.
set -u
cd "$HOME/qlora-sleep" || exit 1

echo "[0] bitsandbytes + pytest (absent from the base image; E4/E5 installed them by hand)"
python3 -c "import bitsandbytes, pytest" 2>/dev/null \
  || pip install -q bitsandbytes pytest
python3 -c "import bitsandbytes, pytest; print('    bnb', bitsandbytes.__version__, 'pytest', pytest.__version__)"
echo "    exit=$?"

echo "[1] pillow + numpy pin"
pip install -q -U pillow; echo "    pillow exit=$?"
python3 -c "import numpy,sys; sys.exit(0 if numpy.__version__.startswith('1.26') else 1)" \
  || pip install -q numpy==1.26.4
python3 -c "import numpy, PIL; print('    numpy', numpy.__version__, 'pillow', PIL.__version__)"

echo "[2] transformers + peft"
python3 -c "import transformers, peft; print('    have', transformers.__version__, peft.__version__)" 2>/dev/null \
  || pip install -q transformers==4.57.6 peft==0.21.2
python3 -c "import transformers, peft
from transformers import OlmoeForCausalLM
print('    transformers', transformers.__version__, 'peft', peft.__version__, 'OlmoeForCausalLM ok')"
echo "    import exit=$?"

echo "[3] CPU gate"
python3 -m pytest -q 2>&1 | tail -3
echo "    pytest exit=${PIPESTATUS[0]}"

echo "[4] weights"
python3 - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("allenai/OLMoE-1B-7B-0125")
print("    snapshot", path)
PY
echo "    download exit=$?"

echo "[5] GPU smoke gate"
python3 scripts/smoke_gpu.py
echo "    smoke exit=$?"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[bringup] done"
