"""Uniform average of the periodic checkpoints ckpt_<step>.pt of one run.

Usage: python average_checkpoints.py RUN_DIR END_STEP WINDOW OUT_PATH
Averages every checkpoint with END_STEP - WINDOW < step <= END_STEP. The result only
uses weights trained on the first END_STEP env steps, so it is a snapshot for END_STEP.
"""
import glob
import os
import re
import sys

import torch

run_dir, end, window, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
paths = []
for p in glob.glob(os.path.join(run_dir, "ckpt_*.pt")):
    step = int(re.search(r"ckpt_(\d+)\.pt$", p).group(1))
    if end - window < step <= end:
        paths.append((step, p))
paths.sort()
if not paths:
    sys.exit("no checkpoints in window")

avg = None
for _, p in paths:
    sd = torch.load(p, map_location="cpu")
    if avg is None:
        avg = {k: v.clone().double() for k, v in sd.items()}
    else:
        for k in avg:
            avg[k] += sd[k].double()
avg = {k: (v / len(paths)).float() for k, v in avg.items()}
torch.save(avg, out)
print(f"averaged {len(paths)} checkpoints, steps {paths[0][0]}..{paths[-1][0]} -> {out}")
