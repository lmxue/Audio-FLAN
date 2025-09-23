# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute WER

import os
import jiwer
import torch
import whisper
import sys
import logging
import warnings
import string
import argparse
from tqdm import tqdm
import numpy as np
import glob
from scripts.dump_eval_result import dump_result
warnings.simplefilter('ignore', FutureWarning)


def asr(audio_path, model):
    transcribed_result = model.transcribe(audio_path, language='en', verbose=False)
    postprocessed_text = transcribed_result['text'].strip()
    return postprocessed_text

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def calculate_audio_wer(ref_dir, deg_dir, output_file):
    
    input_files = glob.glob(f"{deg_dir}/*.wav")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
        model = whisper.load_model("large").to(device)

    if not os.path.exists(output_file):
        with open(output_file, 'w') as output:
            output.write("audio_name wer\n")
    
    wer_list = []

    with open(output_file, 'a') as output:
        for deg_wav in tqdm(input_files):
            ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))  

            reference_lyrics = asr(ref_wav, model)
            generated_lyrics = asr(deg_wav, model)

            reference_lyrics = preprocess_text(reference_lyrics)
            generated_lyrics = preprocess_text(generated_lyrics)

            if not reference_lyrics: # gt is none
                wer = 0.0
            else:
                if not generated_lyrics: # generate is none but gt is not none
                    wer = 1.0
                else:
                    wer = jiwer.wer(reference_lyrics, generated_lyrics)
            
            output.write(f"{os.path.basename(deg_wav)[:-4]} {wer:.4f}\n")

            wer_list.append(wer)

        average_wer = np.mean(wer_list)
        output.write(f"AverageWER {average_wer:.4f}")


def calculate_content_wer(targets, preds, output_file, task_type, dataset):

    wer_list = []

    for reference_lyrics, generated_lyrics in zip(targets, preds):

        reference_lyrics = preprocess_text(reference_lyrics)
        generated_lyrics = preprocess_text(generated_lyrics)

        if not reference_lyrics: # gt is none
            wer = 0.0
        else:
            if not generated_lyrics: # generate is none but gt is not none
                wer = 1.0
            else:
                wer = jiwer.wer(reference_lyrics, generated_lyrics)
        
        wer_list.append(wer)
    
    dump_result(np.mean(wer_list), output_file, task_type, dataset, 'content_wer', len(wer_list))
