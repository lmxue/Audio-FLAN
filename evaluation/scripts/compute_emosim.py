# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute Emotion Similarity

import os
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from funasr import AutoModel

def get_emotion_vector(audio, model):
    try:
        rec_result = model.generate(audio, granularity="utterance", extract_embedding=True)
        return rec_result[0]['feats']
    except Exception as e:
        print(f"Extracting {audio} feature failed: {str(e)}")
        return None

def calculate_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return np.nan
    try:
        return 1 - cosine(vec1, vec2)
    except Exception as e:
        print(f"Count sim error: {str(e)}")
        return np.nan

def calculate_emosim(ref_dir, deg_dir, output_file, sr=16000):

    model = AutoModel(model="iic/emotion2vec_plus_base")

    if not os.path.exists(output_file):
        with open(output_file, 'w') as output:
            output.write("audio_name emo-sim\n")
    
    with open(output_file, 'a') as output:

        ref_files = sorted([f for f in os.listdir(ref_dir)])
        deg_files = sorted([f for f in os.listdir(deg_dir)])
        
        if len(ref_files) != len(deg_files):
            print(f"File quantity mismatched.")
            return

        results = []
        valid_count = 0
        for filename in ref_files:
            ref_path = os.path.join(ref_dir, filename)
            deg_path = os.path.join(deg_dir, filename)
            
            ref_audio, _ = librosa.load(ref_path, sr=sr, mono=True)
            deg_audio, _ = librosa.load(deg_path, sr=sr, mono=True)
            if ref_audio is None or deg_audio is None:
                results.append((filename, np.nan))
                continue
                
            ref_vec = get_emotion_vector(ref_audio, model)
            deg_vec = get_emotion_vector(deg_audio, model)
            
            similarity = calculate_similarity(ref_vec, deg_vec)
            if not np.isnan(similarity):
                valid_count += 1
            results.append((filename, similarity))
            
            output.write(f"{filename[:-4]} {similarity:.4f}\n")
        
        valid_scores = [x[1] for x in results if not np.isnan(x[1])]
        avg_score = np.mean(valid_scores) if valid_scores else 0.0

        output.write(f"Average {avg_score:.4f}")

    return avg_score

if __name__ == "__main__":
    
    pass