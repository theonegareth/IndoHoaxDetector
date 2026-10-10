#!/bin/bash
# Runs on the pod: install the paper's exact versions, then work through jobs.txt.
cd /workspace/job
# ssh sessions do not inherit the container environment; take the RunPod variables from PID 1
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
{ command -v runpodctl && runpodctl get pod "$RUNPOD_POD_ID"; } > cap_check.log 2>&1
( sleep 9000; runpodctl remove pod "$RUNPOD_POD_ID" ) > /dev/null 2>&1 &      # hard 2.5 h lifetime cap
export HF_HUB_DOWNLOAD_TIMEOUT=60 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# the image's torchvision/torchaudio are built for its own torch and break transformers' imports once torch
# is upgraded; neither is used here, so remove them. READY requires the classes the jobs actually import.
{ pip uninstall -y -q torchvision torchaudio;
  pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 &&
  pip install -q transformers==4.57.1 accelerate==1.15.0 scikit-learn pandas &&
  python -c "import torch,transformers; from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments; assert torch.cuda.is_available(); print('READY',torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))"; } > setup.log 2>&1
grep -q READY setup.log || { echo "SETUP_FAILED" >> run.log; exit 1; }
while read -r tag split seed lr model; do
  [ -z "$tag" ] && continue
  python train_split.py "$tag" "$split" "$seed" "$lr" "$model" >> run.log 2>&1 || echo "FAILED $tag" >> run.log
done < jobs.txt
echo ALL_DONE >> run.log
