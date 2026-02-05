import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from architecture.idps_net import IDPSNet
from utils.utils import adjust_learning_rate, get_gpu_memory, Timer, save_results

class IDPSTrainer:
    def __init__(self, model: IDPSNet, dataset, criterions, log_writer, conf):
        self.model = model
        self.dataset = dataset
        self.conf = conf
        self.optimizer = optim.Adam(model.parameters(), lr=conf.lr)
        self.criterions = criterions
        self.log_writer = log_writer
        self.global_step = 0
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        timer = Timer()
        
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
            
            # Forward (returns dict of predictions)
            preds = self.model.training_pass(candidate_patches)
            
            # Compute loss
            loss = 0
            task_losses, task_preds, task_labels = {}, {}, {}
            
            for task in self.conf.tasks.values():
                t_name, t_act = task['name'], task['act_fn']
                criterion = self.criterions[t_name]
                
                label = batch_data[t_name].to(self.model.device)
                pred = preds[t_name].squeeze(-1)
                
                if t_act == 'softmax':
                    pred_loss = torch.log(pred + self.conf.eps)
                    label_loss = label
                else:
                    pred_loss = pred.view(-1)
                    label_loss = label.view(-1).type(torch.float32)

                task_loss = criterion(pred_loss, label_loss)
                
                task_losses[t_name] = task_loss.item()
                task_preds[t_name] = pred.detach().cpu().numpy()
                task_labels[t_name] = label.detach().cpu().numpy()
                loss += task_loss

            loss /= len(self.conf.tasks.values())
            
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            
            # Update log
            self.log_writer.update(task_losses, task_preds, task_labels)

        # Perf metrics can be handled outside or added to log_writer if supported
        # For now, we rely on log_writer.compute_metric() in main loop
        
        return {} # Metrics handled by log_writer

    @torch.no_grad()
    def evaluate(self, dataloader, epoch):
        self.model.eval()
        
        for i, batch_data in enumerate(dataloader):
            # --- PASS 1: SCOUT ---
            low_res_patches = self.dataset.get_scout_data(batch_data).to(self.model.device)
            top_m_indices = self.model.scouting_pass(low_res_patches) 
            
            # --- PASS 2: LEARNER ---
            top_m_indices_cpu = top_m_indices.cpu()
            candidate_patches = self.dataset.get_learner_data(batch_data, top_m_indices_cpu).to(self.model.device)
            
            preds = self.model.training_pass(candidate_patches)
            
             # Compute loss
            loss = 0
            task_losses, task_preds, task_labels = {}, {}, {}
            
            for task in self.conf.tasks.values():
                t_name, t_act = task['name'], task['act_fn']
                criterion = self.criterions[t_name]
                
                label = batch_data[t_name].to(self.model.device)
                pred = preds[t_name].squeeze(-1)
                
                if t_act == 'softmax':
                    pred_loss = torch.log(pred + self.conf.eps)
                    label_loss = label
                else:
                    pred_loss = pred.view(-1)
                    label_loss = label.view(-1).type(torch.float32)

                task_loss = criterion(pred_loss, label_loss)
                
                task_losses[t_name] = task_loss.item()
                task_preds[t_name] = pred.detach().cpu().numpy()
                task_labels[t_name] = label.detach().cpu().numpy()
            
            self.log_writer.update(task_losses, task_preds, task_labels)
        
        return {}

