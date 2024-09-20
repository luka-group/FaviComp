from typing import List, Union
import re
import string
from collections import Counter
import numpy as np

def parse_prediction(pred):
    # truncate after to the answer
    stop_idx = -1
    for word in ["Question:", "\n\n", "</s>"]:
        idx = pred.find(word)
        if stop_idx == -1 or (idx != -1 and idx < stop_idx):
            stop_idx = idx
    if stop_idx != -1:
        pred = pred[:stop_idx].strip()
    return pred

######## Evaluation Utils ########

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_score(prediction: str, ground_truths: Union[str, List[str]]):
    correct = np.max([int(normalize_answer(prediction) == normalize_answer(gt)) for gt in ground_truths])
    return correct

def f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    final_metric = {'f1': 0, 'precision': 0, 'recall': 0, 'acc': 0}
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        acc = 1.0 if normalized_ground_truth in normalized_prediction else 0.0
        for k in ['f1', 'precision', 'recall', 'acc']:
            final_metric[k] = max(eval(k), final_metric[k])
    return final_metric['f1'], final_metric['precision'], final_metric['recall'], final_metric['acc']

