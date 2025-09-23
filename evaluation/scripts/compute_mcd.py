# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute Mel-Cepstral Distortion (implementation based on UniAudio's methodology)

import os
import math
import glob
import librosa
import pyworld
import pysptk
import numpy as np
import matplotlib.pyplot as plot
from tqdm import tqdm
import shutil
#from binary_io import BinaryIOCollection
def load_wav(wav_file, sr=16000):
    """
    Load a wav file with librosa.
    :param wav_file: path to wav file
    :param sr: sampling rate
    :return: audio time series numpy array
    """
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav

def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

def wav2mcep_numpy(wavfile, target_directory, sample_rate, FRAME_PERIOD=5.0, alpha=0.65, fft_size=512, mcep_size=34):
    # make relevant directories
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    wavfile_tmp = wavfile.split('RFU')[0]
    if len(wavfile.split('.'))>3:
        source = wavfile_tmp.split('.')[1][-8:]
        target = wavfile_tmp.split('.')[-2][-8:]
        fname = source+'_'+target
    else:
        fname = os.path.basename(wavfile).split('.')[0]
    loaded_wav = load_wav(wavfile, sr=sample_rate)
    # Use WORLD vocoder to spectral envelope
    _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=sample_rate,
                                   frame_period=FRAME_PERIOD, fft_size=fft_size)
    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    np.save(os.path.join(target_directory, fname + '.npy'), mgc, allow_pickle=False)

def average_mcd(ref_mcep_files, synth_mcep_files, cost_function, output_file):
    """
    Calculate the average MCD.
    :param ref_mcep_files: list of strings, paths to MCEP target reference files
    :param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
    :param cost_function: distance metric used
    :returns: average MCD, total frames processed
    """
    min_cost_tot = 0.0
    frames_tot = 0
    if not os.path.exists(output_file):
        with open(output_file, 'w') as output:
            output.write("audio_name mcd\n")
    
    with open(output_file, 'a') as output:        
        for ref in ref_mcep_files:
            for synth in synth_mcep_files:
                # get the trg_ref and conv_synth speaker name and sample id
                ref_fsplit, synth_fsplit = os.path.basename(ref).split('_'), os.path.basename(synth).split('_')
                # ref_spk, ref_id = ref_fsplit[0], ref_fsplit[-1][:3]
                # synth_spk, synth_id = synth_fsplit[2], synth_fsplit[3][:3]
                
                # # if the speaker name is the same and sample id is the same, do MCD
                if os.path.basename(ref) == os.path.basename(synth):
                # load MCEP vectors
                    ref_vec = np.load(ref)
                    ref_frame_no = len(ref_vec)
                    synth_vec = np.load(synth)
                    # dynamic time warping using librosa
                    min_cost, _ = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T, 
                                                        metric=cost_function)
                    min_cost_tot += np.mean(min_cost)
                    frames_tot += ref_frame_no
                    output.write(f"{os.path.basename(ref)[:-4]} {np.mean(min_cost)/ref_frame_no:.4f}\n")
                    
        mean_mcd = min_cost_tot / frames_tot
        output.write(f"Average {mean_mcd:.4f}")
    
    return mean_mcd, frames_tot

def calculate_mcd(ref_dir, deg_dir, output_file, sample_rate=16000):
    
    # Paths to target reference and converted synthesised wavs
    stgan_vc_wav_paths = glob.glob(f"{ref_dir}/*.wav") # ref
    stgan_vc2_wav_paths = glob.glob(f"{deg_dir}/*.wav") # recon
    vc_trg_wavs = glob.glob(f"{ref_dir}/*.wav")
    # using to save the mel file

    parent_dir = os.path.dirname(ref_dir)
    vc_trg_mcep_dir = os.path.join(parent_dir, "temp/mel_r")
    if os.path.exists(vc_trg_mcep_dir):
        shutil.rmtree(vc_trg_mcep_dir)

    os.makedirs(vc_trg_mcep_dir, exist_ok=True)

    vc_conv_wavs = glob.glob(f"{deg_dir}/*.wav")

    vc_conv_mcep_dir = os.path.join(parent_dir, "temp/mel_d")
    if os.path.exists(vc_conv_mcep_dir):
        shutil.rmtree(vc_conv_mcep_dir)
    os.makedirs(vc_conv_mcep_dir, exist_ok=True)

    for wav in tqdm(vc_trg_wavs):
        wav2mcep_numpy(wav, vc_trg_mcep_dir, sample_rate)
    for wav in tqdm(vc_conv_wavs):
        wav2mcep_numpy(wav, vc_conv_mcep_dir, sample_rate)
    vc_trg_refs = glob.glob(f"{vc_trg_mcep_dir}/*")
    vc_conv_synths = glob.glob(f"{vc_conv_mcep_dir}/*")
    cost_function = log_spec_dB_dist
    vc_mcd, vc_tot_frames_used = average_mcd(vc_trg_refs, vc_conv_synths, cost_function, output_file)
    print(f'MCD = {vc_mcd} dB, calculated over a total of {vc_tot_frames_used} frames')

    shutil.rmtree(os.path.dirname(vc_trg_mcep_dir))
