# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute PESQ (implementation based on UniAudio's methodology)

import os
import glob
import argparse
from tqdm import tqdm
from pesq import pesq
from scipy.io import wavfile
import scipy.signal as signal
import librosa


def calculate_pesq(ref_dir, deg_dir, output_file):
    input_files = glob.glob(f"{deg_dir}/*.wav")

    nb_pesq_scores = 0.0
    wb_pesq_scores = 0.0
    cnt = 0
    if not os.path.exists(output_file):
        with open(output_file, 'w') as output:
            output.write("audio_name wb_pesq nb_pesq\n")
    
    with open(output_file, 'a') as output:        
        for deg_wav in tqdm(input_files):
            ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
            ref, ref_rate = librosa.load(ref_wav)
            deg, deg_rate = librosa.load(deg_wav)
            if ref_rate != 16000:
                ref = librosa.resample(ref, orig_sr=ref_rate, target_sr=16000)
                ref_rate = 16000
            if deg_rate != 16000:
                deg = librosa.resample(deg, orig_sr=deg_rate, target_sr=16000)
                deg_rate = 16000

            min_len = min(len(ref), len(deg))
            ref = ref[:min_len]
            deg = deg[:min_len]
            try:
                nb_pesq = pesq(16000, ref, deg, 'nb')
                wb_pesq = pesq(16000, ref, deg, 'wb') # 宽带
                output.write(f"{os.path.basename(deg_wav)[:-4]} {wb_pesq:.4f} {nb_pesq:.4f}\n")

                nb_pesq_scores += nb_pesq
                wb_pesq_scores += wb_pesq
                cnt += 1
            
            except Exception as e:
                print(f"Error processing {os.path.basename(deg_wav)[:-4]}: {e}")
        
        output.write(f"Average {wb_pesq_scores/cnt:.4f} {nb_pesq_scores/cnt:.4f}")
        
    return  wb_pesq_scores/cnt, nb_pesq_scores/cnt


