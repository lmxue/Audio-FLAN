# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute TOR (Time Overlap Ratio)

import re
import numpy as np
from scripts.dump_eval_result import dump_result
from intervaltree import Interval, IntervalTree


def extract_numbers(input_string):
    pattern = r'-?\d+\.\d+|-?\d+'
    numbers = re.findall(pattern, input_string)
    return [float(num) for num in numbers]

def overlap_ratio(gt, pred):

    pred_tree = IntervalTree()

    if len(pred) == 0 or len(pred)%2 != 0:
        return 0
    
    for i in range(0, len(pred), 2):
        start = pred[i]
        end = pred[i + 1]
        if start < end:
            pred_tree.add(Interval(start, end))

    pred_tree.merge_overlaps()

    gt_start, gt_end = gt

    overlap_time = 0.0
    for interval in pred_tree:
        overlap_start = max(interval.begin, gt_start)
        overlap_end = min(interval.end, gt_end)

        if overlap_end > overlap_start:
            overlap_time += overlap_end - overlap_start

    return overlap_time/(gt_end - gt_start)



def calculate_tor(gt_list, pred_list, output_file, task_type, dataset):
    # Time Overlap Ratio

    gt_list = [extract_numbers(gt) for gt in gt_list]
    pred_list = [extract_numbers(pred) for pred in pred_list]
    
    tor_list = []

    for gt, pred in zip(gt_list, pred_list):
        tor_list.append(overlap_ratio(gt, pred))

    dump_result(np.mean(tor_list), output_file, task_type, dataset, 'tor', len(tor_list))

