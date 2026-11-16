#!/bin/bash
# Evaluate every submitted snapshot on seeds 0-19 (20 episodes each).
# Run from the ZIP root:  bash LAB5_115550809_Code/eval_all.sh
# (or from the code folder with the .pt files next to it: bash eval_all.sh)
set -e
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
PT_DIR="${1:-.}"   # folder that holds the LAB5_115550809_*.pt files (default: current folder)

echo "=== Task 1 (CartPole-v1) ==="
python "$CODE_DIR/test_model_task1.py" --model_path "$PT_DIR/LAB5_115550809_task1.pt"

echo "=== Task 2 (Pong-v5) ==="
python "$CODE_DIR/test_model.py" --model_path "$PT_DIR/LAB5_115550809_task2.pt" --env-steps 3092538

echo "=== Task 3 (Pong-v5) ==="
for s in 600000 1000000 1500000 2000000 2500000; do
  python "$CODE_DIR/test_model_task3.py" --model_path "$PT_DIR/LAB5_115550809_task3_${s}.pt" --env-steps "$s"
done
python "$CODE_DIR/test_model_task3.py" --model_path "$PT_DIR/LAB5_115550809_task3_best.pt" --env-steps 3500000
