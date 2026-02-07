import sys
import math
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict

import torch
from torch import nn

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def adjust_learning_rate(n_epoch_warmup, n_epoch, max_lr, optimizer, dloader, step):
    """
    Set learning rate according to cosine schedule
    """

    max_steps = int(n_epoch * len(dloader))
    warmup_steps = int(n_epoch_warmup * len(dloader))
    
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = max_lr * 0.001
        lr = max_lr * q + end_lr * (1 - q)

    optimizer.param_groups[0]['lr'] = lr

def adjust_sigma(n_epoch_warmup, n_epoch, max_sigma, net, dloader, step):
    """
    Set sigma for DPS according to cosine schedule
    """
    max_steps = int(n_epoch * len(dloader))
    warmup_steps = int(n_epoch_warmup * len(dloader))
    
    # If we are in the warmup phase
    if step < warmup_steps:
        # Linear warmup
        sigma = max_sigma * step / warmup_steps
    # If we are in the decay phase
    else:
        # Subtract warmup steps from step and max_steps
        step -= warmup_steps
        max_steps -= warmup_steps
        
        # Cosine decay
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        # Calculate the end sigma value
        end_sigma = 1e-5
        # Calculate the current sigma value
        sigma = max_sigma * q + end_sigma * (1 - q)
    # Update sigma in the DPS module
    net.dps_topk.sigma = sigma

def shuffle_batch(x, shuffle_idx=None):
    """ shuffles each instance in batch the same way """
    
    if not torch.is_tensor(shuffle_idx):
        seq_len = x.shape[1]
        shuffle_idx = torch.randperm(seq_len)
    x = x[:, shuffle_idx]
    
    return x, shuffle_idx

def shuffle_instance(x, axis, shuffle_idx=None):
    """ shuffles each instance in batch in a different way """

    if not torch.is_tensor(shuffle_idx):
        # get permutation indices
        shuffle_idx = torch.rand(x.shape[:axis+1], device=x.device).argsort(axis)  
    
    idx_expand = shuffle_idx.clone().to(x.device)
    for _ in range(x.ndim-axis-1):
        idx_expand.unsqueeze_(-1)
    # reformat for gather operation
    idx_expand = idx_expand.repeat(*[1 for _ in range(axis+1)], *(x.shape[axis+1:]))  
    
    x = x.gather(axis, idx_expand)

    return x, shuffle_idx

class Logger(nn.Module):
    ''' Stores and computes statistiscs of losses and metrics '''

    def __init__(self, task_dict):
        super().__init__()

        self.task_dict = task_dict
        self.losses_it = defaultdict(list)
        self.losses_epoch = defaultdict(list)
        self.y_preds = defaultdict(list)
        self.y_trues = defaultdict(list)
        self.metrics = defaultdict(list)

    def update(self, next_loss, next_y_pred, next_y_true):

        for task in self.task_dict.values():
            t, t_metr = task['name'], task['metric']
            self.losses_it[t].append(next_loss[t])
            
            if t_metr == 'accuracy':
                y_pred = np.argmax(next_y_pred[t], axis=-1)
            elif t_metr in ['multilabel_accuracy', 'auc']:
                y_pred = next_y_pred[t].tolist()
            self.y_preds[t].extend(y_pred)
            
            self.y_trues[t].extend(next_y_true[t])

    def compute_metric(self):

        for task in self.task_dict.values():
            t = task['name']
            losses = self.losses_it[t]
            self.losses_epoch[t].append(np.mean(losses))

            current_metric = task['metric']
            if current_metric == 'accuracy':
                metric = accuracy_score(self.y_trues[t], self.y_preds[t])
                self.metrics[t].append(metric)
            elif current_metric == 'multilabel_accuracy':
                y_pred = np.array(self.y_preds[t])
                y_true = np.array(self.y_trues[t])
                
                y_pred = np.where(y_pred >= 0.5, 1., 0.)
                correct = np.all(y_pred == y_true, axis=-1).sum()
                total = y_pred.shape[0]
                
                self.metrics[t].append(correct / total)
            elif current_metric == 'auc':
                y_pred = np.array(self.y_preds[t])
                y_true = np.array(self.y_trues[t])
                auc = roc_auc_score(y_true, y_pred)
                self.metrics[t].append(auc)

            # reset per iteration losses, preds and labels
            self.losses_it[t] = []
            self.y_preds[t] = []
            self.y_trues[t] = []


    def print_stats(self, epoch, train, **kwargs):

        print_str = 'Train' if train else 'Test'
        print_str +=  " Epoch: {} \n".format(epoch+1)

        avg_loss = 0
        for task in self.task_dict.values():
            t = task['name']
            metric_name = task['metric']
            mean_loss = self.losses_epoch[t][epoch]
            metric = self.metrics[t][epoch]
           
            avg_loss += mean_loss
 
            print_str += "task: {}, mean loss: {:.5f}, {}: {:.5f}, ".format(t, mean_loss, metric_name, metric)

        avg_loss /= len(self.task_dict.values())
        print_str += "avg. loss over tasks: {:.5f}".format(avg_loss)

        for k, v in kwargs.items():
            print_str += ", {}: {}".format(k, v)
        print_str += "\n"

        print(print_str)

import time
import json
import os

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

class Timer:
    def __init__(self):
        self.start_time = time.time()
    
    def elapsed(self):
        return time.time() - self.start_time

def save_results(results_dir, epoch, metrics, state_dict=None):
    # Save metrics
    log_path = os.path.join(results_dir, 'training_log.json')
    
    # Load existing or create new
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []
        
    log_data.append({'epoch': epoch, 'metrics': metrics})
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    # Save checkpoint if provided
    if state_dict is not None:
        torch.save(state_dict, os.path.join(results_dir, f'checkpoint_ep{epoch}.pth'))