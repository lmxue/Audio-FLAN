# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute F0-RMSE (implementation based on Amphion's methodology)

import math
import librosa
import torch
import numpy as np
from utils.util import JsonHParams
# from utils.f0 import get_f0_features_using_parselmouth
from utils.f0 import get_f0_features_using_pyin
from utils.f0 import get_pitch_sub_median
import glob
import os
from tqdm import tqdm

def calculate_f0rmse(ref_dir, deg_dir, output_file):

    hop_length=256
    f0_min=50
    f0_max=1100
    """Compute F0 Root Mean Square Error (RMSE) between the predicted and the ground truth audio.
    ref_dir: path to the ground truth audio.
    deg_dir: path to the predicted audio.
    fs: sampling rate.
    hop_length: hop length.
    f0_min: lower limit for f0.
    f0_max: upper limit for f0.
    pitch_bin: number of bins for f0 quantization.
    pitch_max: upper limit for f0 quantization.
    pitch_min: lower limit for f0 quantization.
    need_mean: subtract the mean value from f0 if "True".
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    """
    fs = None
    method = "dtw"
    need_mean = True

    input_files = glob.glob(f"{deg_dir}/*.wav")
    
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {deg_dir}")

    if os.path.exists(output_file):
        raise RuntimeError(f"File exists: {output_file}")
    else:
        with open(output_file, 'w') as output:
            output.write("audio_name f0rmse\n")

    r0rmse_scores = []
    with open(output_file, 'a') as output:        
        
        for deg_wav in tqdm(input_files):
            ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))

            # Load audio
            if fs != None:
                audio_ref, _ = librosa.load(ref_wav, sr=fs)
                audio_deg, _ = librosa.load(deg_wav, sr=fs)
            else:
                audio_ref, fs = librosa.load(audio_ref)
                audio_deg, fs = librosa.load(audio_deg)

            # Initialize config for f0 extraction
            cfg = JsonHParams()
            cfg.sample_rate = fs
            cfg.hop_size = hop_length
            cfg.f0_min = f0_min
            cfg.f0_max = f0_max
            cfg.pitch_bin = 256
            cfg.pitch_max = f0_max
            cfg.pitch_min = f0_min

            # Extract f0
            f0_ref = get_f0_features_using_pyin(
                audio_ref,
                cfg,
            )

            f0_deg = get_f0_features_using_pyin(
                audio_deg,
                cfg,
            )

            # Subtract mean value from f0
            if need_mean:
                f0_ref = torch.from_numpy(f0_ref)
                f0_deg = torch.from_numpy(f0_deg)

                f0_ref = get_pitch_sub_median(f0_ref).numpy()
                f0_deg = get_pitch_sub_median(f0_deg).numpy()

            # Avoid silence
            min_length = min(len(f0_ref), len(f0_deg))
            if min_length <= 1:
                return 0

            # F0 length alignment
            if method == "cut":
                length = min(len(f0_ref), len(f0_deg))
                f0_ref = f0_ref[:length]
                f0_deg = f0_deg[:length]
            elif method == "dtw":
                _, wp = librosa.sequence.dtw(f0_ref, f0_deg, backtrack=True)
                f0_gt_new = []
                f0_pred_new = []
                for i in range(wp.shape[0]):
                    gt_index = wp[i][0]
                    pred_index = wp[i][1]
                    f0_gt_new.append(f0_ref[gt_index])
                    f0_pred_new.append(f0_deg[pred_index])
                f0_ref = np.array(f0_gt_new)
                f0_deg = np.array(f0_pred_new)
                assert len(f0_ref) == len(f0_deg)

            # Compute RMSE
            f0_mse = np.square(np.subtract(f0_ref, f0_deg)).mean()
            f0_rmse = math.sqrt(f0_mse)

            output.write(f"{os.path.basename(deg_wav)[:-4]} {f0_rmse:.4f}\n")
            
            r0rmse_scores.append(f0_rmse)

        output.write(f"Average {np.mean(r0rmse_scores):.4f}")

    return np.mean(r0rmse_scores)
