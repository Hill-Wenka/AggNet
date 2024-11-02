import numpy as np


def sigmoid(scores, k=1, low=-1, high=1):
    scores = np.array(scores)
    return 1 / (1 + np.exp(-k * (scores - (high + low) / 2) / (high - low)))


def sigmoid_inv(prob, k=1, low=-1, high=1):
    prob = np.array(prob)
    return np.log(prob / (1 - prob)) * (high - low) / k + (high + low) / 2
