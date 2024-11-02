from .predict_utils import *
from .trainer_utils import *

def seed_everything(seed=42, workers=True):
    # 固定所有的随机种子，确保实验可复现
    L.seed_everything(seed=seed, workers=workers)
