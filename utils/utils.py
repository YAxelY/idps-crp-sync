import sys
import math
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
# Optional imports if sklearn is not installed in environment, but ips had it.
try:
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError:
    pass

import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# Optional imports
try:
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
except ImportError:
    pass

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3 # GB
    return 0.0

class Timer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.start = time.time()
    def elapsed(self):
        return time.time() - self.start

def adjust_learning_rate(n_epoch_warmup, n_epoch, max_lr, optimizer, dloader, step):
    if step is None: return 
    max_steps = int(n_epoch * len(dloader))
    warmup_steps = int(n_epoch_warmup * len(dloader))
    
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        if max_steps <= 0: q = 1
        else: q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = max_lr * 0.001
        lr = max_lr * q + end_lr * (1 - q)
    
    optimizer.param_groups[0]['lr'] = lr

def save_results(results_dir, epoch, metrics, model_state):
    # Save CSV or JSON logs
    import os
    import json
    
    os.makedirs(results_dir, exist_ok=True)
    
    # metrics is a dict
    log_path = os.path.join(results_dir, "training_log.json")
    
    history = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try: history = json.load(f)
            except: pass
    
    metrics['epoch'] = epoch
    history.append(metrics)
    
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    # Save Checkpoint
    torch.save(model_state, os.path.join(results_dir, f"checkpoint_ep{epoch}.pth"))


def shuffle_batch(x, shuffle_idx=None):
    """ shuffles each instance in batch the same way """
    if not torch.is_tensor(shuffle_idx):
        seq_len = x.shape[1]
        shuffle_idx = torch.randperm(seq_len)
    x = x[:, shuffle_idx]
    return x, shuffle_idx
