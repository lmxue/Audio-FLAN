# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute clap

import librosa
import torch
from transformers import ClapModel, ClapProcessor
import numpy as np
import os
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model.eval()

def compute_pair_similarity(text, audio_path, sr=44100):

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    audio_input = processor(audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True).to(device)
    text_input = processor(text=[text], return_tensors="pt", padding=True).to(device)
    
    # similarity
    with torch.no_grad():
        audio_embed = model.get_audio_features(**audio_input)
        text_embed = model.get_text_features(**text_input)
        similarity = torch.nn.functional.cosine_similarity(audio_embed, text_embed, dim=-1)
    
    return similarity.item()

def calculate_clap(data_dict, deg_dir, output_file):

    '''
    data in data_dict e.g.:
    {
    "audio1_name": "audio1_text",
    "audio2_name": "audio2_text",
    ...
    }
    '''
    pairs = []
    
    for audio_name, audio_text in data_dict.items():
        audio_path = os.path.join(deg_dir, audio_name)
        if not os.path.exists(audio_path):
            print(f"Warning: audio {audio_path} does not exist, skipped.")
            continue
        pairs.append((audio_name, audio_path, audio_text))

    if pairs:
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding="utf-8") as f_out:
                f_out.write("audio_name clap\n")

        similarities = []
        with open(output_file, 'a', encoding="utf-8") as f_out:
            for audio_name, audio_path, text in pairs:
                try:
                    similarity = compute_pair_similarity(text, audio_path)

                    similarities.append(similarity)
                    f_out.write(f"{audio_name} {similarity:.4f}\n")
                    
                except Exception as e:
                    print(f"Failed while dealing with {audio_name} : {str(e)}")
                    continue

        if similarities:
            avg_score = np.mean(similarities)
            with open(output_file, "a", encoding="utf-8") as f_out:
                f_out.write(f"Average {avg_score:.4f}")

            return avg_score


if __name__ == '__main__':
    
    pass