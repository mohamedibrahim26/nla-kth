#!/usr/bin/env bash
# run_code_analysis.sh — End-to-end NLA code-model extension pipeline
#
# Prerequisites: the warm-start verbalizer and reconstructor checkpoints must
# already exist in DATA (run the main pipeline first, sections 4.1–4.5).
#
# Runtime estimate on Kaggle 2x T4: ~2–3 hours for 2000 function pairs.
#
# Usage:
#   bash run_code_analysis.sh
#   DATA=data bash run_code_analysis.sh   # explicit data dir

set -euo pipefail
DATA="${DATA:-data}"

echo "================================================================"
echo "  NLA Code Extension — Bug/Fix Activation Analysis"
echo "================================================================"

# Step 1: Harvest activations from Qwen2.5-Coder-0.5B on bug/fix pairs
echo ""
echo "Step 1: Harvesting code activations …"
python src/harvest_code_activations.py \
    --data_dir "$DATA" \
    --num_functions 2000 \
    --layer 16 \
    --max_tokens 256

# Step 2: Generate teacher summaries for each code snippet
echo ""
echo "Step 2: Generating teacher summaries …"
python src/generate_code_teacher_summaries.py \
    --data_dir "$DATA" \
    --batch_size 4 \
    --limit 2000

# Step 3: Train reconstructor on code activations
# Reuse train_reconstructor.py — just point it at the code activations
echo ""
echo "Step 3: Training reconstructor on code activations …"
python src/train_reconstructor.py \
    --data_dir   "$DATA" \
    --act_file   code_activations.pkl \
    --sum_file   code_teacher_summaries.pkl \
    --output_dir "$DATA" \
    --ckpt_prefix code_reconstructor \
    --mode lora \
    --epochs 8

# Step 4: Train verbalizer on code activations (warm-start only)
echo ""
echo "Step 4: Training code verbalizer …"
python src/train_verbalizer.py \
    --data_dir   "$DATA" \
    --act_file   code_activations.pkl \
    --sum_file   code_teacher_summaries.pkl \
    --output_dir "$DATA" \
    --ckpt_prefix code_verbalizer \
    --epochs 3

# Step 5: Run bug description analysis (the research question)
echo ""
echo "Step 5: Analysing bug vs correct descriptions …"
python src/analyze_bug_descriptions.py \
    --data_dir  "$DATA" \
    --out_dir   results \
    --verbalizer_pt    "$DATA/code_verbalizer.pt" \
    --verbalizer_lora  "$DATA/code_verbalizer_lora" \
    --reconstructor_pt "$DATA/code_reconstructor.pt" \
    --reconstructor_lora "$DATA/code_reconstructor_lora"

echo ""
echo "================================================================"
echo "  Done!  Results → results/bug_analysis.json"
echo "================================================================"
