# Copyright 2025
# Author     : Audio-FLAN teams
# Description: Compute Bleu, METEOR, ROUGE, CIDEr (implementation based on Qwen-Audio's methodology)


from __future__ import annotations
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path
from utils.caption_evaluation_tools.coco_caption.pycocotools.coco import COCO
from utils.caption_evaluation_tools.coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import itertools
import json
import os
import string
from functools import partial
import re
from utils.caption_evaluation_tools.coco_caption.pycocoevalcap.bleu.bleu import Bleu
from utils.caption_evaluation_tools.coco_caption.pycocoevalcap.meteor.meteor import Meteor
from utils.caption_evaluation_tools.coco_caption.pycocoevalcap.rouge.rouge import Rouge
from utils.caption_evaluation_tools.coco_caption.pycocoevalcap.cider.cider import Cider
# from utils.caption_evaluation_tools.coco_caption.pycocoevalcap.spice.spice import Spice
import torch
from scripts.dump_eval_result import dump_result
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PUNCS = '!,.?;:'

#
ds_collections = {
    'clotho': {'path': 'data/caption/clotho_eval.jsonl', 'style': 'clotho'} # style: clotho, audiocaps, or plain
}



def write_json(data: Union[List[Dict[str, Any]], Dict[str, Any]],
               path: Path) \
        -> None:
    """ Write a dict or a list of dicts into a JSON file
    :param data: Data to write
    :type data: list[dict[str, any]] | dict[str, any]
    :param path: Path to the output file
    :type path: Path
    """
    with path.open("w") as f:
        json.dump(data, f)


def reformat_to_coco(predictions: List[str],
                     ground_truths: List[List[str]],
                     ids: Union[List[int], None] = None) \
        -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ Reformat annotations to the MSCOCO format
    :param predictions: List of predicted captions
    :type predictions: list[str]
    :param ground_truths: List of lists of reference captions
    :type ground_truths: list[list[str]]
    :param ids: List of file IDs. If not given, a running integer\
                is used
    :type ids: list[int] | None
    :return: Predictions and reference captions in the MSCOCO format
    :rtype: list[dict[str, any]]
    """
    # Running number as ids for files if not given
    if ids is None:
        ids = list(range(len(predictions)))

    # Captions need to be in format
    # [{
    #     "audio_id": : int,
    #     "caption"  : str
    # ]},
    # as per the COCO results format.
    pred = []
    ref = {
        'info': {'description': 'Clotho reference captions (2019)'},
        'audio samples': [],
        'licenses': [
            {'id': 1},
            {'id': 2},
            {'id': 3}
        ],
        'type': 'captions',
        'annotations': []
    }
    cap_id = 0
    for audio_id, p, gt in zip(ids, predictions, ground_truths):
        p = p[0] if isinstance(p, list) else p
        pred.append({
            'audio_id': audio_id,
            'caption': p
        })

        ref['audio samples'].append({
            'id': audio_id
        })

        for cap in gt:
            ref['annotations'].append({
                'audio_id': audio_id,
                'id': cap_id,
                'caption': cap
            })
            cap_id += 1

    return pred, ref

class CocoTokenizer:
    def __init__(self, preds_str, trues_str, tem_dir):
        self.evalAudios = []
        self.eval = {}
        self.audioToEval = {}
        pred, ref = reformat_to_coco(preds_str, trues_str)
        self.tmp_dir = Path(tem_dir).parent

        if not self.tmp_dir.is_dir():
            self.tmp_dir.mkdir()
        self.ref_file = self.tmp_dir.joinpath('ref.json')
        self.pred_file = self.tmp_dir.joinpath('pred.json')

        write_json(ref, self.ref_file)
        write_json(pred, self.pred_file)

        self.coco = COCO(str(self.ref_file))
        self.cocoRes = self.coco.loadRes(str(self.pred_file))

        self.params = {'audio_id': self.coco.getAudioIds()}

    def __del__(self):
        # Delete temporary files
        self.ref_file.unlink()
        self.pred_file.unlink()

    def tokenize(self):
        audioIds = self.params['audio_id']
        # audioIds = self.coco.getAudioIds()
        gts = {}
        res = {}
        for audioId in audioIds:
            gts[audioId] = self.coco.audioToAnns[audioId]
            res[audioId] = self.cocoRes.audioToAnns[audioId]

###
        # tokenizer = PTBTokenizer()
        # gts = tokenizer.tokenize(gts)
        # res = tokenizer.tokenize(res)
###
###
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
        gts_tk, res_tk = {}, {}
        for key, _ in gts.items():
            gts_tk[key] = [' '.join(tokenizer.tokenize(gts[key][0]['caption'], truncation=True, max_length=512))]
        for key, _ in res.items():
            res_tk[key] = [' '.join(tokenizer.tokenize(res[key][0]['caption'], truncation=True, max_length=512))]

###

        return res_tk, gts_tk

    def setEval(self, score, method):
        self.eval[method] = score

    def setAudioToEvalAudios(self, scores, audioIds, method):
        for audioId, score in zip(audioIds, scores):
            if not audioId in self.audioToEval:
                self.audioToEval[audioId] = {}
                self.audioToEval[audioId]["audio_id"] = audioId
            self.audioToEval[audioId][method] = score

    def setEvalAudios(self):
        self.evalAudios = [eval for audioId, eval in self.audioToEval.items()]
      
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def calculate_caption(gts, res, output_file, task_type, dataset):
    preds_str = [[preprocess_text(res_item)] for res_item in res]
    references = [[preprocess_text(gts_item)] for gts_item in gts]
    
    tokenizer = CocoTokenizer(preds_str, references, output_file)
    res, gts = tokenizer.tokenize()

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE")
    ]
    f_res = {}

    for scorer, method in scorers:
        # print('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                # print("%s: %0.3f" % (m, sc))
                f_res[m] = sc
        else:
            # print("%s: %0.3f" % (method, score))
            f_res[method] = score
    # f_res["SPIDEr"] = (f_res['CIDEr']+f_res['SPICE']) / 2.

    dump_result(f_res, output_file, task_type, dataset, 'caption_metrics', len(references))
    
    return f_res

if __name__ == '__main__':

    preds_str = ["The dog is playing.","this is a sentence.","this is a sentence.","this is a sentence.","this is a sentence."]

    trues_str = ["The dog is playing in the garden.","this is another sentence.","this is another sentence.","this is another sentence.","this is another sentence."]

    f_res = calculate_caption(preds_str, trues_str)

    print("Evaluation Scores:")
    for metric, score in f_res.items():
        print(f"{metric}: {score:.3f}")