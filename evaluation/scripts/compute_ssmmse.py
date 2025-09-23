# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute SSM-MSE (Mean Squared Error between the standardized Self-Similarity Matrix)

import librosa
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import os

def compute_ssm(audio_path, sr=16000, feature_type='mfcc', normalize=True):

    y, _ = librosa.load(audio_path, sr=sr)
    
    if feature_type == 'mfcc':
        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    else:
        raise ValueError("Unsupported feature type")

    features = features.T

    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    ssm = 1 - cdist(features, features, metric='cosine')
    
    return ssm

def compute_single_ssm_mse(ssm1, ssm2, standardize=True):

    min_len = min(ssm1.shape[0], ssm2.shape[0])
    ssm1 = ssm1[:min_len, :min_len]
    ssm2 = ssm2[:min_len, :min_len]

    if standardize:
        ssm1 = (ssm1 - np.mean(ssm1)) / np.std(ssm1)
        ssm2 = (ssm2 - np.mean(ssm2)) / np.std(ssm2)

    mse = np.mean((ssm1 - ssm2) ** 2)
    return mse

def calculate_ssmmse(ref_dir, deg_dir, output_file):

    if not os.path.exists(output_file):
        with open(output_file, 'w') as output:
            output.write("audio_name ssm-mse\n")
    
    with open(output_file, 'a') as output:

        ref_files = sorted([f for f in os.listdir(ref_dir)])
        deg_files = sorted([f for f in os.listdir(deg_dir)])
        
        if len(ref_files) != len(deg_files):
            print(f"File quantity mismatched.")
            return

        results = []
        for filename in ref_files:
            ref_path = os.path.join(ref_dir, filename)
            deg_path = os.path.join(deg_dir, filename)
            
            ref_ssm = compute_ssm(ref_path)
            deg_ssm = compute_ssm(deg_path)
            mse = compute_single_ssm_mse(ref_ssm, deg_ssm)

            results.append((filename, mse))
            
            output.write(f"{filename[:-4]} {mse:.4f}\n")
        
        valid_scores = [x[1] for x in results if not np.isnan(x[1])]
        avg_score = np.mean(valid_scores) if valid_scores else 0.0

        output.write(f"Average {avg_score:.4f}")

    return avg_score
            

if __name__ == "__main__":

    pass