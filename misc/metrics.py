__author__ = 'Erdene-Ochir Tuguldur'

import numpy as np
import Levenshtein


def compute_metric(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score
