# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute SDR (Signal-to-Distortion Ratio)

import os
import jiwer
import torch
import librosa
import mir_eval 
import numpy as np
from fastdtw import fastdtw
import glob
import argparse


def align_audio(reference_audio, estimated_audio):

    distance, path = fastdtw(reference_audio, estimated_audio)
    
    path = np.array(path)
    aligned_reference = reference_audio[path[:, 0]] 
    aligned_estimated = estimated_audio[path[:, 1]] 

    aligned_reference = np.array(aligned_reference)
    aligned_estimated = np.array(aligned_estimated)
    
    return aligned_reference, aligned_estimated

def compute_sdr(reference_path, estimated_path):

    reference_audio, ref_sr = librosa.load(reference_path, sr=None)
    estimated_audio, est_sr = librosa.load(estimated_path, sr=None)
    
    if ref_sr != est_sr:
        raise ValueError("Sample rates of reference and estimated audio do not match.")
    
    reference_audio, estimated_audio = align_audio(reference_audio, estimated_audio)
    
    # 归一化音频（防止幅度影响）
    reference_audio = reference_audio / np.max(np.abs(reference_audio))
    estimated_audio = estimated_audio / np.max(np.abs(estimated_audio))
    
    reference_audio = np.expand_dims(reference_audio, axis=0)
    estimated_audio = np.expand_dims(estimated_audio, axis=0)

    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference_audio, estimated_audio)
    
    return sdr[0], sir[0], sar[0]

def calculate_sdr(ref_dir, deg_dir, output_file):
    
    input_files = glob.glob(f"{deg_dir}/*.wav")

    total_sdr = 0
    count = 0

    if not os.path.exists(output_file):
        with open(output_file, 'w') as output:
            output.write("audio_name sdr\n")

    with open(output_file, 'a') as output:
        
        for deg_wav in tqdm(input_files):
            ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))

            sdr , _ , _ = compute_sdr(ref_wav, deg_wav)
                    
            output.write(f"{os.path.basename(deg_wav)[:-4]} {sdr:.4f}\n")

            total_sdr += sdr               
            count += 1

        output.write(f"Average {total_sdr/count:.4f}")

    print(f"sdr: {total_sdr/count}")
    
