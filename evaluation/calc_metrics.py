import os
import sys
import io
import numpy as np
import json
import argparse
import whisper
import torch
from collections import defaultdict

from scripts.compute_f0rmse import calculate_f0rmse
from scripts.compute_stoi import calculate_stoi
from scripts.compute_cossim import calculate_speaker_similarity
from scripts.compute_fad import calculate_fad
from scripts.compute_mcd import calculate_mcd
from scripts.compute_pesq import calculate_pesq
from scripts.compute_sdr import calculate_sdr
from scripts.compute_snr import calculate_snr
from scripts.compute_wer import calculate_audio_wer, calculate_content_wer
from scripts.compute_eer import calculate_eer
from scripts.compute_der import calculate_der
from scripts.compute_acc import calculate_f1, calculate_pairwise_f1, calculate_group_f1
from scripts.compute_tor import calculate_tor
from scripts.compute_caption import calculate_caption
from scripts.compute_mae import calculate_mae
from scripts.compute_mse import calculate_mse
from scripts.compute_clap import calculate_clap
from scripts.compute_emosim import calculate_emosim
from scripts.compute_ssmmse import calculate_ssmmse

def load_input_file(input_file):

    if isinstance(input_file, dict):
        return input_file

    elif isinstance(input_file, str) and os.path.isfile(input_file) and input_file.endswith('.jsonl'):

        with open(input_file, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            
        return data_dict

    else:
        raise ValueError(f"Wrong type: {type(input_file)}")



METRIC_FUNC = {
    "f0rmse": calculate_f0rmse,
    "stoi": calculate_stoi,
    "cos_sim": calculate_speaker_similarity,
    "fad": calculate_fad,
    "mcd": calculate_mcd,
    "pesq": calculate_pesq,
    "sdr": calculate_sdr,
    "snr": calculate_snr,
    "audio_wer": calculate_audio_wer,
    "content_wer": calculate_content_wer,
    "eer": calculate_eer, # receive gts and preds as 2 lists including 'same speaker' and 'different speakers'
    "der": calculate_der,
    "mae": calculate_mae,
    "mse": calculate_mse,
    "choice_acc": calculate_f1,
    "sequence_acc": calculate_pairwise_f1,
    "group_choice_acc": calculate_group_f1,
    "caption_metrics": calculate_caption,
    "tor": calculate_tor,
    "clap": calculate_clap,
    "emo_sim": calculate_emosim,
    "ssm_mse": calculate_ssmmse,

}

def calc_metric(metrics, output_file, ref_dir=None, deg_dir=None, input_file=None):

    # check parameters
    if not metrics or not output_file:
        raise ValueError("Both 'metrics' and 'output_file' must be provided")

    if isinstance(metrics, str):
        metrics = [metrics]
    elif isinstance(metrics, list):
        pass
    else:
        raise ValueError("'metrics' should be a str or a list.")

    if ref_dir and deg_dir:
        if input_file is not None:
            raise ValueError("'input_file' should not be provided when 'ref_dir' and 'deg_dir' are given.")
    else:
        if input_file is None:
            raise ValueError("Provide 'ref_dir' and 'deg_dir' for audio dir; or provide 'input_file' for jsonl.")

    #
    for metric in metrics:
        print('------------------------------------------------')
        print(f'computing {metric} score...')
        if metric in ['mae', 'mse', 'choice_acc', 'sequence_acc', 'group_choice_acc', 'content_wer', 'caption_metrics', 'eer', 'der', 'tor']:
            data_dict = load_input_file(input_file)
            '''
            data_dict e.g.:
            {
            'task_type1':
                {'dataset1_name':{'pred': predlist, 'gt': gtlist}}
                {'dataset2_name':{'pred': predlist, 'gt': gtlist}}
                ...
            'task_type2':    
                ...
            }

            '''
            for task_type in data_dict:
                for dataset in data_dict[task_type]:
                    preds = data_dict[task_type][dataset]['pred']
                    targets = data_dict[task_type][dataset]['gt']
                    if len(preds) != len(targets):
                        print(f'Warning: preds({len(preds)}) and targets({len(targets)}) in {task_type} are not in the same length!')
                    METRIC_FUNC[metric](targets, preds, output_file, task_type, dataset)

        elif metric in ['clap']:
            data_dict = load_input_file(input_file)
            METRIC_FUNC[metric](data_dict, deg_dir, output_file)
        
        else:
            METRIC_FUNC[metric](ref_dir, deg_dir, output_file)

        print(f'{metric} score completed.')

if __name__ == "__main__":
    
    pass