import sys
import numpy as np
import torch

from utils.utils import adjust_learning_rate, adjust_sigma

def compute_loss(net, preds, labels, conf):
    """ Compute losses for each task and sum them up """
    loss = 0
    task_losses, task_preds, task_labels = {}, {}, {}
    for task in conf.tasks.values():
        t_name = task['name']
        t_act = task['act_fn']
        label = labels[t_name].to(preds.device)
        pred = preds.squeeze(-1) # Assuming single task for now or preds dict

        # BCE or CrossEntropy based on your config. Assuming CrossEntropy for MNIST tasks here.
        criterion = torch.nn.CrossEntropyLoss()
        
        task_loss = criterion(pred, label)
        
        task_losses[t_name] = task_loss.item()
        
        if t_act == 'softmax':
            task_preds[t_name] = torch.softmax(pred, dim=-1).detach().cpu().numpy()
        else:
            task_preds[t_name] = pred.detach().cpu().numpy()
            
        task_labels[t_name] = label.detach().cpu().numpy()

        loss += task_loss

    loss /= len(conf.tasks.values())
    return loss, [task_losses, task_preds, task_labels]

def train_one_epoch(net, criterions, data_loader, optimizer, device, epoch, log_writer, conf):
    net.train()
    times = []

    # Our custom collate_fn ensures data is a list of raw dicts
    for data_it, batch_data_list in enumerate(data_loader, start=epoch * len(data_loader)):
        
        if conf.track_efficiency:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        # Update Schedulers
        adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch, conf.lr, optimizer, data_loader, data_it+1)
        adjust_sigma(conf.warmup_sigma, conf.n_epoch, conf.max_sigma, net, data_loader, data_it+1)
        optimizer.zero_grad()

        optimizer.zero_grad()

        # ---- PASS 1: SCOUT ----
        # 1. Fetch Downsampled Patches Lazy
        patches_low, labels = data_loader.dataset.get_scout_data(batch_data_list)
        patches_low = patches_low.to(device)
        
        # 2. Iterative Filtering (IPS style)
        # top_m_indices: (B, M)
        top_m_indices = net.scouting_pass(patches_low)
        
        # ---- PASS 2: LEARNER ----
        # 1. We already have the M downsampled patches in memory, we just gather them.
        B, N, C, H_low, W_low = patches_low.shape
        D_exp = len(patches_low.shape) - 2 # 3
        
        expanded_indices = top_m_indices.view(B, -1, *(1,)*D_exp).expand(-1, -1, C, H_low, W_low)
        patches_low_m = torch.gather(patches_low, 1, expanded_indices) # (B, M, C, H_low, W_low)

        # 2. Fetch High-Res Patches Lazy
        # We only pass the indices back to the dataset.
        patches_high_m = data_loader.dataset.get_learner_data(batch_data_list, top_m_indices)
        patches_high_m = patches_high_m.to(device)

        # 3. Differentiable Selection & Prediction
        preds = net.training_pass(patches_low_m, patches_high_m)

        # 4. Computed Loss over K High-Res Patches and Backprop
        loss, task_info = compute_loss(net, preds, labels, conf)
        task_losses, task_preds, task_labels = task_info

        loss.backward()
        optimizer.step()

        if conf.track_efficiency:
            end_event.record()
            torch.cuda.synchronize()
            if epoch == conf.track_epoch and data_it > 0:
                times.append(start_event.elapsed_time(end_event))

        log_writer.update(task_losses, task_preds, task_labels)

    if conf.track_efficiency and epoch == conf.track_epoch:
        stats = torch.cuda.memory_stats()
        peak_bytes_requirement = stats["allocated_bytes.all.peak"]
        print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.4f} GB")
        sys.exit()


@torch.no_grad()
def evaluate(net, criterions, data_loader, device, log_writer, conf):
    net.eval()
    
    for batch_data_list in data_loader:
        patches_low, labels = data_loader.dataset.get_scout_data(batch_data_list)
        patches_low = patches_low.to(device)
        
        top_m_indices = net.scouting_pass(patches_low)
        
        B, N, C, H_low, W_low = patches_low.shape
        expanded_indices = top_m_indices.view(B, -1, *(1,)*3).expand(-1, -1, C, H_low, W_low)
        patches_low_m = torch.gather(patches_low, 1, expanded_indices) 

        patches_high_m = data_loader.dataset.get_learner_data(batch_data_list, top_m_indices)
        patches_high_m = patches_high_m.to(device)

        preds = net.training_pass(patches_low_m, patches_high_m)

        _, task_info = compute_loss(net, preds, labels, conf)
        task_losses, task_preds, task_labels = task_info

        log_writer.update(task_losses, task_preds, task_labels)
