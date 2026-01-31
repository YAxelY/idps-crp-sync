import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from architecture.idps_net import IDPSNet
from utils.utils import adjust_learning_rate, get_gpu_memory, Timer, save_results

class IDPSTrainer:
    def __init__(self, model: IDPSNet, dataset, conf):
        self.model = model
        self.dataset = dataset
        self.conf = conf
        self.optimizer = optim.Adam(model.parameters(), lr=conf.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.global_step = 0
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        timer = Timer()
        metrics = {'loss': 0, 'acc': 0}
        total_batches = len(dataloader)
        
        for i, batch_data in enumerate(dataloader):
            # Dynamic LR
            adjust_learning_rate(self.conf.n_epoch_warmup, self.conf.n_epoch, self.conf.lr, 
                                 self.optimizer, dataloader, self.global_step)
            
            # --- PASS 1: SCOUT ---
            low_res_patches = self.dataset.get_scout_data(batch_data).to(self.model.device)
            top_m_indices = self.model.scouting_pass(low_res_patches) # (B, M)
            
            # --- PASS 2: LEARNER (With DPS) ---
            # Move indices to CPU for indexing into dataset (which is on CPU)
            top_m_indices_cpu = top_m_indices.cpu()
            candidate_patches = self.dataset.get_learner_data(batch_data, top_m_indices_cpu).to(self.model.device)
            self.optimizer.zero_grad()
            logits = self.model.training_pass(candidate_patches)
            
            # Labels
            labels = batch_data['label'].clone().detach().to(self.model.device).long()
            
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            
            metrics['loss'] += loss.item()
            metrics['acc'] += acc
            self.global_step += 1
            
        # Average Metrics
        metrics = {k: v / total_batches for k, v in metrics.items()}
        
        # Add perf metrics
        metrics['vram_gb'] = get_gpu_memory()
        metrics['time_sec'] = timer.elapsed()
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        # Save results (Train)
        metrics['phase'] = 'train'
        save_results(self.conf.results_dir, epoch, metrics, self.model.state_dict())
        
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader, epoch):
        self.model.eval()
        timer = Timer()
        metrics = {'loss': 0, 'acc': 0}
        total_batches = len(dataloader)
        
        for i, batch_data in enumerate(dataloader):
            # --- PASS 1: SCOUT ---
            low_res_patches = self.dataset.get_scout_data(batch_data).to(self.model.device)
            top_m_indices = self.model.scouting_pass(low_res_patches) # (B, M)
            
            # --- PASS 2: LEARNER (With DPS) ---
            # Move indices to CPU for indexing into dataset (which is on CPU)
            top_m_indices_cpu = top_m_indices.cpu()
            candidate_patches = self.dataset.get_learner_data(batch_data, top_m_indices_cpu).to(self.model.device)
            logits = self.model.training_pass(candidate_patches)
            
            # Labels
            labels = batch_data['label'].clone().detach().to(self.model.device).long()
            
            loss = self.criterion(logits, labels)
            
            # Metrics
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            
            metrics['loss'] += loss.item()
            metrics['acc'] += acc
            
        # Average Metrics
        metrics = {k: v / total_batches for k, v in metrics.items()}
        
        # Add perf metrics
        metrics['vram_gb'] = get_gpu_memory()
        metrics['time_sec'] = timer.elapsed()
        metrics['phase'] = 'test'
        
        # Save results (Test) - Note: separate log or same? 
        # save_results appends to json, so it will just add another entry.
        # We might want to save just metrics without checking point too often, but it handles it.
        save_results(self.conf.results_dir, epoch, metrics, None) # No checkpoint regarding test usually, or 'best' logic
        
        return metrics

