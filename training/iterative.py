import sys
import numpy as np
import torch
from utils.utils import adjust_learning_rate, Timer, get_gpu_memory

def compute_loss(net, preds, criterions, labels, conf):
    """
    Obtain predictions, compute losses for each task and get some logging stats
    """
    # preds are already computed in IDPS logic (training_pass returns them)
    
    # Compute losses for each task and sum them up
    loss = 0
    task_losses, task_preds, task_labels = {}, {}, {}
    for task in conf.tasks.values():
        t_name, t_act = task['name'], task['act_fn']

        criterion = criterions[t_name]
        label = labels[t_name]
        pred = preds[t_name].squeeze(-1)

        if t_act == 'softmax':
            pred_loss = torch.log(pred + conf.eps)
            label_loss = label
        else:
            pred_loss = pred.view(-1)
            label_loss = label.view(-1).type(torch.float32)

        task_loss = criterion(pred_loss, label_loss)
        # for logs
        task_losses[t_name] = task_loss.item()
        task_preds[t_name] = pred.detach().cpu().numpy()
        task_labels[t_name] = label.detach().cpu().numpy()

        loss += task_loss
    # Average task losses        
    loss /= len(conf.tasks.values())

    return loss, [task_losses, task_preds, task_labels]


def train_one_epoch(net, criterions, train_data, data_loader, optimizer, device, epoch, log_writer, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    net.train()

    times = [] # only used when tracking efficiency stats
    
    # Loop through dataloader
    for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        
        # If tracking efficiency, record time from here.
        if conf.track_efficiency:
            timer = Timer()

        # Calculate and set new learning rate
        adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch, conf.lr, optimizer, data_loader, data_it+1)
        optimizer.zero_grad()
        
        # --- PASS 1: SCOUT ---
        low_res_patches = train_data.get_scout_data(data).to(device)
        top_m_indices = net.scouting_pass(low_res_patches) # (B, M)
        
        # --- PASS 2: LEARNER (With DPS) ---
        top_m_indices_cpu = top_m_indices.cpu()
        candidate_patches = train_data.get_learner_data(data, top_m_indices_cpu).to(device)

        # Forward (returns dict of predictions)
        preds = net.training_pass(candidate_patches)
        
        # Prepare labels relative to device
        labels = {}
        for task in conf.tasks.values():
            labels[task['name']] = data[task['name']].to(device)

        # Compute loss
        loss, task_info = compute_loss(net, preds, criterions, labels, conf)
        task_losses, task_preds, task_labels = task_info

        # Backpropagate error and update parameters
        loss.backward()
        optimizer.step()

        # If tracking efficiency, log the time and memory usage
        if conf.track_efficiency:
            if epoch == conf.track_epoch and data_it > 0:
                times.append(timer.elapsed())
                print("time: ", times[-1])

        # Update log
        log_writer.update(task_losses, task_preds, task_labels)

    
    if conf.track_efficiency:
        if epoch == conf.track_epoch:
            print("avg. time: ", np.mean(times))

            peak_bytes_requirement = get_gpu_memory()
            print(f"Peak memory requirement: {peak_bytes_requirement:.4f} GB")

            if torch.cuda.is_available():
                print("TORCH.CUDA.MEMORY_SUMMARY: ", torch.cuda.memory_summary())
            sys.exit()


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, criterions, test_data, data_loader, device, log_writer, conf):

    # Set the network to evaluation mode
    net.eval()

    for data in data_loader:
        
        # --- PASS 1: SCOUT ---
        low_res_patches = test_data.get_scout_data(data).to(device)
        top_m_indices = net.scouting_pass(low_res_patches) 
        
        # --- PASS 2: LEARNER ---
        top_m_indices_cpu = top_m_indices.cpu()
        candidate_patches = test_data.get_learner_data(data, top_m_indices_cpu).to(device)
        
        preds = net.training_pass(candidate_patches)
        
        # Prepare labels
        labels = {}
        for task in conf.tasks.values():
            labels[task['name']] = data[task['name']].to(device)

        # Compute loss
        _, task_info = compute_loss(net, preds, criterions, labels, conf)
        task_losses, task_preds, task_labels = task_info

        log_writer.update(task_losses, task_preds, task_labels)
