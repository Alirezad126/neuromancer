import torch
import numpy as np
import random
import os
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True