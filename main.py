#!/usr/bin/env python

import os
import yaml
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import Logger, Struct
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST
from data.traffic.traffic_dataset import TrafficSigns
from data.camelyon.camelyon_dataset import CamelyonFeatures
from architecture.idps_net import IDPSNet
from training.trainer import IDPSTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'mnist' # either one of {'mnist', 'camelyon', 'traffic'}

# get config
with open(os.path.join('config', dataset + '_config.yml'), "r") as ymlfile:
    c = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print("Used config:"); pprint(c);
    conf = Struct(**c)

# Ensure results_dir exists and set device
if not hasattr(conf, 'results_dir'):
    conf.results_dir = 'results'
os.makedirs(conf.results_dir, exist_ok=True)
conf.device = device 

# fix the seed for reproducibility
torch.manual_seed(conf.seed)
np.random.seed(conf.seed)

# define datasets and dataloaders
if dataset == 'mnist':
    train_data = MegapixelMNIST(conf, train=True)
    test_data = MegapixelMNIST(conf, train=False)
elif dataset == 'traffic':
    train_data = TrafficSigns(conf, train=True)
    test_data = TrafficSigns(conf, train=False)
elif dataset == 'camelyon':
    train_data = CamelyonFeatures(conf, train=True)
    test_data = CamelyonFeatures(conf, train=False)

train_loader = DataLoader(train_data, batch_size=conf.B_seq, shuffle=True,
    num_workers=conf.n_worker, pin_memory=conf.pin_memory, persistent_workers=True)
test_loader = DataLoader(test_data, batch_size=conf.B_seq, shuffle=False,
    num_workers=conf.n_worker, pin_memory=conf.pin_memory, persistent_workers=True)

# define network
net = IDPSNet(conf).to(device)

loss_nll = nn.NLLLoss()
loss_bce = nn.BCELoss()

criterions = {}
for task in conf.tasks.values():
    criterions[task['name']] = loss_nll if task['act_fn'] == 'softmax' else loss_bce

log_writer_train = Logger(conf.tasks)
log_writer_test = Logger(conf.tasks)

# Trainer
trainer = IDPSTrainer(net, train_data, criterions, log_writer_train, conf)

print(f"Starting training on {dataset}...")
for epoch in range(conf.n_epoch):
    
    print(f"Epoch {epoch+1}/{conf.n_epoch}")
    
    # Train
    trainer.train_epoch(train_loader, epoch)
    log_writer_train.compute_metric()
    log_writer_train.print_stats(epoch, train=True)
    
    # Evaluate
    # Note: evaluate in trainer also uses self.log_writer (which is set to log_writer_train)
    # We should probably pass the test logger to evaluate or handle it differently.
    # But IDPSTrainer as implemented expects one log_writer in init.
    # To fix this without changing trainer.py too much, we can swap the logger temporarily or instantiate a separate trainer for eval?
    # No, cleaner: use trainer's evaluate but pass the test logger or update trainer to accept logger in evaluate.
    # Looking at my trainer.py implementation: `evaluate` uses `self.log_writer`.
    # I should update `evaluate` in `trainer.py` to accept a logger, OR just set `trainer.log_writer = log_writer_test` before eval.
    
    trainer.log_writer = log_writer_test
    trainer.evaluate(test_loader, epoch)
    log_writer_test.compute_metric()
    log_writer_test.print_stats(epoch, train=False)
    
    # Reset for next epoch training
    trainer.log_writer = log_writer_train