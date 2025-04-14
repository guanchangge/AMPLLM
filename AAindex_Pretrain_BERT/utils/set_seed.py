import torch
import os
import random
import numpy as np
# pytorch Set the random seed (I tried it and it worked)(In case it doesn't work, refer to https://blog.csdn.net/weixin_40400177/article/details/105625873)
def set_seed(seed=42):
    random.seed(seed) # python seed
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed) # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    # torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True # When set to True, cuDNN uses a non-deterministic algorithm to find the most efficient algorithm.
    # torch.backends.cudnn.enabled = True # Pytorch uses CUDANN acceleration, that is, GPU acceleration
# seed_torch(seed=42)
