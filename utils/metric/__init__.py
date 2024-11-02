import sklearn.metrics
from scipy.stats import pearsonr, spearmanr

from .bin_cls import *
from .ece import *
from .sov import *


def pearson_corrcoef(x, y):
    return pearsonr(x, y)[0]


def spearman_corrcoef(x, y):
    return spearmanr(x, y)[0]


def r2_score(y_true, y_pred):
    return sklearn.metrics.r2_score(y_true, y_pred)
