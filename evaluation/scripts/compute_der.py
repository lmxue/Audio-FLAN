# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute DER (Diarization Error Rate)

import re
import warnings
import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from scripts.dump_eval_result import dump_result
warnings.filterwarnings("ignore", category=UserWarning)

def parse_segments(input_string):

    speaker_pattern = r"(\w+):\s*(<[\d.]+s,\s*[\d.]+s>[\.,\s]*)*"
    time_pattern = r"([\d.]+)s,\s*([\d.]+)s"

    r_segments = {}
    speakers = re.findall(speaker_pattern, input_string)

    for speaker, _ in speakers:
        segments = []
        speaker_segments = re.findall(time_pattern, input_string.split(speaker+":")[1])
        
        for start, end in speaker_segments:
            segments.append(Segment(float(start), float(end)))
        
        r_segments[speaker] = segments

    annotation = Annotation()

    for speaker, times in r_segments.items():
        for segment in times:
            annotation[segment] = speaker

    return annotation


def calculate_der(targets, preds, output_file, task_type, dataset):

    der = DiarizationErrorRate()
    der_results = []

    for gt_segments, pred_segments in zip(targets, preds):

        gt_segments = parse_segments(gt_segments)
        pred_segments = parse_segments(pred_segments)
        
        file_der = der(gt_segments, pred_segments)
        
        der_results.append(file_der)
        
    dump_result(np.mean(der_results), output_file, task_type, dataset, 'der', len(der_results))