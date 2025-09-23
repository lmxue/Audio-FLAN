# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute SNR (Signal-to-Noise Ratio)

import os
import torch
import argparse
import librosa
from tqdm import tqdm
import numpy as np
from fastdtw import fastdtw
import glob

def load_audio(audio_path):
    return librosa.load(audio_path, sr=16000)

def compute_snr(clean_signal, enhanced_signal):
    
    noise = enhanced_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        snr = np.inf
    else:
        snr = 10 * np.log10(signal_power / noise_power)  # SNR in dB
    
    return snr


def calculate_snr(ref_dir, deg_dir, output_file):
    
    input_files = glob.glob(f"{deg_dir}/*.wav")


    total_snr = 0
    count = 0
    if not os.path.exists(output_file):
        with open(output_file, 'w') as output:
            output.write("audio_name snr\n")

    with open(output_file, 'a') as output:
        for deg_wav in tqdm(input_files):
            ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))

            deg, deg_rate = load_audio(deg_wav)
            ref, ref_rate = load_audio(ref_wav)
            
            distance, wp = fastdtw(deg, ref)
            
            wp = np.array(wp)

            deg = deg[wp[:, 0]]
            ref = ref[wp[:, 1]]

            snr = compute_snr(ref, deg)

            output.write(f"{os.path.basename(deg_wav)[:-4]} {snr:.4f}\n")

            if snr != np.inf:
                total_snr += snr
                count += 1
  
        output.write(f"Average {total_snr/count:.4f}")

    print(f"snr: {total_snr/count}")


