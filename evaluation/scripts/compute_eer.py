# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute EER (Equal Error Rate)

import os
import numpy as np
from scripts.dump_eval_result import dump_result

def c_eer(fnr, fpr, scores=None):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))

    if scores is not None:
        score_sort = np.sort(scores)
        return fnr[x1] + a * (fnr[x2] - fnr[x1]), score_sort[x1]

    return fnr[x1] + a * (fnr[x2] - fnr[x1])

def compute_pmiss_pfa_rbst(scores, labels, weights=None):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial socres and their labels. A weights option is also provided to
    equalize the counts over score partitions (if there is such partitioning).
    """

    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]
    if weights is not None:
        weights = weights[sorted_ndx]
    else:
        weights = np.ones((labels.shape), dtype='f8')

    tgt_wghts = weights * (labels == 1).astype('f8')
    imp_wghts = weights * (labels == 0).astype('f8')

    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    return fnr, fpr

def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    c_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det / c_def

def get_id(s):
    s = s.split('SV_')[-1]
    name1 = ''
    new_s = ''
    for i in range(2, len(s)):
        if s[i] != '-':
            name1 = name1 +s[i]
        else:
            new_s = s[i:]
            break
    new_s = new_s.split('_id')[-1]
    name2 = ''
    for i in range(len(new_s)):
        if new_s[i] != '-':
            name2 = name2 + new_s[i]
        else:
            break
    return name1, name2


def compute_metrics(scores, labels, p_target=0.01, c_miss=1, c_fa=1):

    scores = np.hstack(scores)
    labels = np.hstack(labels)
    # print('scores ', scores.shape, labels.shape)
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = c_eer(fnr, fpr, scores)

    # min_dcf = compute_c_norm(fnr,
    #                          fpr,
    #                          p_target=p_target,
    #                          c_miss=c_miss,
    #                          c_fa=c_fa)
    # print("---- {} -----".format(os.path.basename(scores_file)))
    # print("EER = {0:.4f}".format(100 * eer))
    # print("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
    #     p_target, c_miss, c_fa, min_dcf))

    return eer

def calculate_eer(targets, preds, output_file, task_type, dataset):

    '''
    preds should be a list only with 2 types of strings
    '''

    if len(set(targets)) != 2:
        print(task_type, dataset)
        import pdb; pdb.set_trace()
        raise ValueError("Preds list must contain exactly two distinct elements.")
    
    unique_values = list(set(targets))
    
    bool_list = {unique_values[0]: True, unique_values[1]: False}

    pred_list, target_list = [], []
    for pred, target in zip(preds, targets):
        if pred in unique_values:
            pred_list.append(bool_list[pred])
            target_list.append(bool_list[target])

    eer = compute_metrics(pred_list, target_list)

    dump_result(eer, output_file, task_type, dataset, 'eer', len(pred_list))
