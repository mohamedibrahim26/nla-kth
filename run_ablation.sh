#!/usr/bin/env bash
# run_ablation.sh — layer ablation for the NLA reconstructor.
#
# Trains one reconstructor per layer in {8, 12, 16, 20} and compares
# oracle FVE (upper bound: teacher summary -> reconstructor).
#
# Prerequisite: teacher summaries must already exist (run the main pipeline
# up to generate_teacher_summaries.py first).
#
# Usage:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh          # uses data/ by default
#   DATA=my_data ./run_ablation.sh

DATA=${DATA:-data}

set -euo pipefail

python src/layer_ablation.py \
    --data_dir   "$DATA" \
    --output_dir "$DATA" \
    --layers 8 12 16 20 \
    --n_epochs_recon 5

echo ""
echo "Done. Results: ${DATA}/layer_ablation_summary.json"
echo ""
echo "To evaluate the full NLA pipeline at a specific layer:"
echo "  python src/evaluate_nla.py --data_dir ${DATA}/ablation_layer_16"
