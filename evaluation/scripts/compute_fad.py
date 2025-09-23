# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute Fréchet Audio Distance (implementation based on UniAudio's methodology)

import os
import argparse
import glob
import sys
# please fisrt git clone the audioldm_eval, and then sys.path.append the audioldm_eval
# sys.path.append('audioldm_eval')
import torch
from audioldm_eval import EvaluationHelper

def calculate_fad(ref_dir, deg_dir, output_file):

    device = torch.device(f"cuda:{0}")
    evaluator = EvaluationHelper(16000, device)
    metrics = evaluator.main(deg_dir, ref_dir)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as output:
        output.write(str(metrics))
