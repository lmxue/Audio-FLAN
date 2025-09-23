# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute accuracy

import numpy as np
from scripts.dump_eval_result import dump_result

def calculate_f1(correct_answer_list, infer_answer_list, output_file, task_type, dataset):
    f1_list = []
    for infer_answer, correct_answer in zip(infer_answer_list, correct_answer_list):
        f1_list.append(cal_single_f1(infer_answer, correct_answer))
    dump_result(np.mean(f1_list), output_file, task_type, dataset, 'choice_acc', len(f1_list))

def calculate_pairwise_f1(real_chords_list, predicted_chords_list, output_file, task_type, dataset):
    f1_list = []
    for predicted_chords, real_chords in zip(predicted_chords_list, real_chords_list):
        f1_list.append(cal_single_pairwise_f1(predicted_chords, real_chords))
    dump_result(np.mean(f1_list), output_file, task_type, dataset, 'sequence_acc', len(f1_list))

def calculate_group_f1(correct_answer_matrix, infer_answer_matrix, output_file, task_type, dataset):
    f1_list = []
    
    for infer_answer_list, correct_answer_list in zip(infer_answer_matrix, correct_answer_matrix):
        templist = []

        if len(infer_answer_list) < len(correct_answer_list):
            infer_answer_list.extend([] * (len(correct_answer_list) - len(infer_answer_list)))
        elif len(infer_answer_list) > len(correct_answer_list):
            infer_answer_list = infer_answer_list[:len(correct_answer_list)]

        for infer_answer, correct_answer in zip(infer_answer_list, correct_answer_list):
            templist.append(cal_single_f1(infer_answer, correct_answer))
        f1_list.append(np.mean(templist))
    dump_result(np.mean(f1_list), output_file, task_type, dataset, 'group_choice_acc', len(f1_list))

def cal_single_f1(infer_answer, correct_answer): 
    if isinstance(infer_answer, str) and isinstance(correct_answer, str):
        return 1 if infer_answer == correct_answer else 0
    elif isinstance(infer_answer, list) and isinstance(correct_answer, list):
        infer_set = set(infer_answer)
        correct_set = set(correct_answer)
        
        true_positive = len(infer_set & correct_set)
        precision = true_positive / len(infer_set) if infer_set else 0
        recall = true_positive / len(correct_set) if correct_set else 0
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0

def cal_single_pairwise_f1(predicted_chords, real_chords): 

    aligned_real, aligned_predicted = align_chord_sequences(real_chords, predicted_chords)

    scale = len(aligned_real) 
    matrix = np.zeros((scale, scale))
    for i in range(scale):
        for j in range(scale):
            if aligned_real[i] == aligned_real[j]:
                matrix[i, j] += 1
            if aligned_predicted[i] == aligned_predicted[j]:
                matrix[i, j] += 2

    FP, FN, TP = 0, 0, 0
    for i in range(scale):
        for j in range(scale):
            if matrix[i, j] == 1:
                FN += 1
            elif matrix[i, j] == 2:
                FP += 1
            elif matrix[i, j] == 3:
                TP += 1

    precision = TP / (TP+FP) if (TP+FP) else 0
    recall = TP / (TP+FN) if (TP+FN) else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def align_chord_sequences(real_chords, predicted_chords):
    n = len(real_chords) + 1
    m = len(predicted_chords) + 1
    dp = [[0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                if real_chords[i-1] == predicted_chords[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1)

    aligned_real = []
    aligned_predicted = []
    i, j = n - 1, m - 1

    while i > 0 or j > 0:
        if i > 0 and j > 0 and real_chords[i-1] == predicted_chords[j-1]:
            aligned_real.append(real_chords[i-1])
            aligned_predicted.append(predicted_chords[j-1])
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            aligned_real.append('-')
            aligned_predicted.append(predicted_chords[j-1])
            j -= 1
        else:
            aligned_real.append(real_chords[i-1])
            aligned_predicted.append('-')
            i -= 1

    aligned_real.reverse()
    aligned_predicted.reverse()

    return aligned_real, aligned_predicted