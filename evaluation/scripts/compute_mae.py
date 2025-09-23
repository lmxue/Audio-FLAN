# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute MAE

from scripts.dump_eval_result import dump_result
import numpy as np
import torch

def calculate_mae(targets, preds, output_file, task_type, dataset):
    
    try:
        targets = [float(x) for x in targets]
    except ValueError:
        print(task_type)
        print(f"Error converting elements in targets to float. All elements should be convertible to numbers.")
        raise
    
    try:
        preds = [float(x) for x in preds]
    except ValueError:
        print(task_type)
        print(f"Error converting elements in preds to float. All elements should be convertible to numbers.")
        raise
    
    clean_targets, clean_preds = [], []
    for target, pred in zip(targets, preds):
        if not np.isnan(pred):
            clean_targets.append(target)
            clean_preds.append(pred)
    # print('before:',len(preds), len(clean_preds))
    targets = torch.tensor(clean_targets)
    preds = torch.tensor(clean_preds)
    mae = torch.nn.functional.l1_loss(preds, targets)
    dump_result(mae.item(), output_file, task_type, dataset, 'mae', len(clean_preds))