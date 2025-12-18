import torch
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

# Set backend to avoid display errors on servers
import matplotlib
matplotlib.use('Agg') 

def moving_average(data, window_size=100):
    """Smoothes noisy data using a simple moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def draw_record(run_id):
    # 1. Locate and Load Data
    run_dir = str(run_id)
    path = os.path.join(run_dir, 'training_loss_record.pth')
    
    if not os.path.exists(path):
        print(f"[Error] No record found at: {path}")
        print("Make sure you have trained the model for at least one epoch.")
        return

    print(f"Loading record from {path}...")
    data = torch.load(path)
    
    # Extract metrics
    # Keys: total_loss, action_loss, logits_loss, action_L2_error, awl_action_weight, awl_task_weight
    total_loss = np.array(data['total_loss'])
    action_loss = np.array(data['action_loss'])
    logits_loss = np.array(data['logits_loss'])
    l2_error = np.array(data['action_L2_error'])
    w_action = np.array(data['awl_action_weight'])
    w_task = np.array(data['awl_task_weight'])
    
    iterations = np.arange(len(total_loss))

    # --- Plot 1: Training Losses ---
    plt.figure(figsize=(12, 6))
    
    # Plot raw data faintly
    plt.plot(iterations, total_loss, alpha=0.15, color='gray', label='Total (Raw)')
    
    # Plot smoothed data
    window = 100
    if len(iterations) > window:
        smooth_x = iterations[window-1:]
        plt.plot(smooth_x, moving_average(total_loss, window), color='black', linewidth=2, label='Total (Smooth)')
        plt.plot(smooth_x, moving_average(action_loss, window), color='blue', linewidth=1.5, label='Action Loss')
        plt.plot(smooth_x, moving_average(logits_loss, window), color='green', linewidth=1.5, label='Logits Loss')
    else:
        plt.plot(iterations, total_loss, color='black', label='Total')
        
    plt.title(f"Training Losses (Run {run_id})")
    plt.xlabel("Iterations (Batches)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log') # Log scale is often better for losses
    
    save_path = os.path.join(run_dir, 'graph_losses.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

    # --- Plot 2: Action L2 Error ---
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, l2_error, alpha=0.2, color='orange')
    if len(iterations) > window:
        plt.plot(smooth_x, moving_average(l2_error, window), color='darkorange', linewidth=2, label='L2 Error (Smooth)')
    
    plt.title(f"Action Prediction Error (L2 Distance)")
    plt.xlabel("Iterations")
    plt.ylabel("Euclidean Distance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(run_dir, 'graph_action_error.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

    # --- Plot 3: Adaptive Weights (AWL) ---
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, w_action, label='Action Weight (Precision)', color='blue')
    plt.plot(iterations, w_task, label='Task Weight (Precision)', color='green')
    
    plt.title("Evolution of Automatic Uncertainty Weights")
    plt.xlabel("Iterations")
    plt.ylabel("Weight Value (Higher = More Certain)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(run_dir, 'graph_awl_weights.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Training Records")
    parser.add_argument("--identifier", type=str, required=True, help="Identifier of the training run to plot")
    args = parser.parse_args()
    
    draw_record(args.identifier)
    
# python draw_train_record.py --identifier run_001