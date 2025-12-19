import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VLA_transformer, AutomaticWeightedLoss
import json
from dataset import BendStickDataset
from tqdm import tqdm
from macro import DEBUG_MODE
import numpy as np
import argparse
from util import Logger, check_memory_status
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision('high')
        

class Trainer_VLA_Transformer():
    def __init__(self, identifier:str, num_epochs, init_lr, positive_weight, save_every, validate_every, batch_size):
        self.device = "cuda"
        self.model = VLA_transformer().to(self.device)
        self.num_epochs = num_epochs
        self.identifier = identifier
        self.lr = init_lr
        self.save_every = save_every
        self.validate_every = validate_every

        self.checkpoint_path = os.path.join(self.identifier, "checkpoint.pth")
        self.loss_record_save_path = os.path.join(self.identifier, 'training_loss_record.pth')
        self.config_path = os.path.join(self.identifier, 'config.json')
        self.log_path = os.path.join(self.identifier, "train_log.txt")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        sys.stdout = Logger(self.log_path)
        
        check_memory_status()
        
        if os.path.exists(self.config_path):
            print(f"Using config from {self.config_path}")
            config = json.load(open(self.config_path, 'r'))
            self.batch_size = config.get("batch_size")
            self.threshold = config.get("best_threshold")
            self.current_epoch = config.get("epochs_trained")
            self.awl_action_w = config.get("awl_action_weight")
            self.awl_task_w = config.get("awl_task_weight")
            self.positive_weight = torch.tensor([config.get("positive_weight", positive_weight)], device=self.device)
        else:
            print(f"Starting new training run: {self.identifier}")
            os.makedirs(self.identifier, exist_ok=True)
            self.batch_size = batch_size
            self.awl_action_w = 1
            self.awl_task_w = 1
            self.current_epoch = 0
            self.threshold = 0.5
            self.positive_weight = torch.tensor([positive_weight], device=self.device) # only 1% positive samples for task completion
        
        print(f'Device: {self.device}, batch size: {self.batch_size}, prior_epochs: {self.current_epoch}, pos_weight: {self.positive_weight.item()}, '
              f'awl_action_w: {self.awl_action_w:.4f}, awl_task_w: {self.awl_task_w:.4f}, threshold: {self.threshold:.4f}')
        
        print("Compiling model...")
        self.model = torch.compile(self.model)
        self.awl = AutomaticWeightedLoss(self.awl_action_w, self.awl_task_w, num_tasks=2).to(self.device)
        
        if os.path.exists(self.checkpoint_path):
            print(f"Loaded model weights from {self.checkpoint_path}")
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'weight_decay': 1e-2},
            {'params': self.awl.params, 'weight_decay': 0}
        ], lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=10,
        #     T_mult=2,
        #     eta_min=1e-6
        # )

        print("Loading training dataset...")
        dataset_train = BendStickDataset(data_path="data_train.npz")
        self.dataloader_train = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=4, persistent_workers=True)
        print("Loading validation dataset...")
        dataset_validate = BendStickDataset(data_path="data_validate.npz")
        self.dataloader_validate = DataLoader(dataset=dataset_validate, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=4, persistent_workers=True)
        
        # --- GPU MEMORY ---
        print(f"Loaded {len(dataset_train)} training samples and {len(dataset_validate)} validation samples.")
        
    
    # Better approach: Homoscedastic Uncertainty Weighting
    def weighted_loss(self, action_pd, logits_pd, ground_truth):
        task_gt = ground_truth[:, 3].view(-1, 1) # make sure it doesn't flatten to (N,)
        action_gt = ground_truth[:, :3]
        
        logits_loss = nn.BCEWithLogitsLoss(pos_weight=self.positive_weight)(logits_pd, task_gt)
        
        # action_loss = nn.MSELoss()(action_pd, action_gt) # this is the mean scalar, not per-sample loss
        raw_action_loss = nn.MSELoss(reduction='none')(action_pd, action_gt)
        mask = (1.0 - task_gt).expand_as(raw_action_loss) # Create mask: 1.0 for normal, 0.0 for task completion
        masked_action_loss = raw_action_loss * mask
        action_loss = masked_action_loss.sum() / (mask.sum() + 1e-6) # average only on non-completed tasks
        
        return self.awl(action_loss, logits_loss), action_loss, logits_loss


    def __call__(self):
        self.model.train()
        check_memory_status()
        print('\n' + '='*20 + ' Training ' + '='*20)

        for epoch in range(self.num_epochs):
            epoch_size = len(self.dataloader_train)
            progress_bar = tqdm(enumerate(self.dataloader_train),
                                total=epoch_size,
                                desc=f"{epoch + 1}/{self.num_epochs}",
                                ncols=140)
            
            action_loss_list = []
            logits_loss_list = []
            fuse_loss_list = []
            action_L2_error_list = []
            awl_actionW_list, awl_taskW_list = [], []

            for _, (img1, img2, img3, current_state, gt) in progress_bar:
                
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                img3 = img3.to(self.device)
                current_state = current_state.to(self.device)
                gt = gt.to(self.device)
                
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    action_pd, logits_pd = self.model(img1, img2, img3, current_state)
                    loss, action_l, logits_l = self.weighted_loss(action_pd, logits_pd, gt)
                loss.backward()
                # Prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Get current learned weights for display
                with torch.no_grad():
                    # exp(-s) is roughly the inverse variance (precision)
                    w_action = torch.exp(-self.awl.params[0]).item()
                    w_task = torch.exp(-self.awl.params[1]).item()
                    
                progress_bar.set_postfix({
                    "fuse": f"{loss:.4f}",
                    "act": f"{action_l:.4f}",
                    "logit": f"{logits_l:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    "W_Act": f"{w_action:.4f}", # Monitor this! Should go UP as model learns
                    "W_Tsk": f"{w_task:.4f}"
                })
                
                fuse_loss_list.append(loss.item())
                action_loss_list.append(action_l.item())
                logits_loss_list.append(logits_l.item())
                action_L2_error_list.append(torch.mean(torch.norm(action_pd - gt[:, :3], dim=1)).item())
                awl_actionW_list.append(w_action)
                awl_taskW_list.append(w_task)
            
            # update epoch info in config
            with open(self.config_path, 'w') as f:
                json.dump({
                    "batch_size": self.batch_size,
                    "epochs_trained": self.current_epoch,
                    "current_lr": self.optimizer.param_groups[0]['lr'],
                    "best_threshold": self.threshold, # update every validation
                    "positive_weight": self.positive_weight.item(),
                    "awl_action_weight": w_action, # monitor per epoch
                    "awl_task_weight": w_task
                }, f)
            
            # save loss record
            if os.path.exists(self.loss_record_save_path):
                record = torch.load(self.loss_record_save_path)
                fuse_loss_list = record['total_loss'] + fuse_loss_list
                action_loss_list = record['action_loss'] + action_loss_list
                logits_loss_list = record['logits_loss'] + logits_loss_list
                action_L2_error_list = record['action_L2_error'] + action_L2_error_list
                awl_actionW_list = record['awl_action_weight'] + awl_actionW_list
                awl_taskW_list = record['awl_task_weight'] + awl_taskW_list
            torch.save({
                "total_loss": fuse_loss_list,
                "action_loss": action_loss_list,
                "logits_loss": logits_loss_list,
                "action_L2_error": action_L2_error_list,
                "awl_action_weight": awl_actionW_list,
                "awl_task_weight": awl_taskW_list
            }, self.loss_record_save_path)
            
            if epoch % self.save_every == 0 or epoch == self.num_epochs - 1:
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Saved model to {self.checkpoint_path}")

            if epoch % self.validate_every == 0 or epoch == self.num_epochs - 1:
                self.validate()
                if epoch != self.num_epochs - 1:
                    print('\n' + '='*20 + ' Training ' + '='*20)
                    
            self.scheduler.step()
            self.current_epoch += 1
        


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        n_batches = len(self.dataloader_validate)
        print(f'\n{"="*20} Validation {"="*20}')
        
        loss_sum, action_loss_sum, logits_loss_sum = 0, 0, 0
        all_action_pd, all_logits_pd = [], []
        all_gt_action, all_gt_logit = [], []
        
        bar = tqdm(self.dataloader_validate, total=n_batches, desc="Validating", ncols=100)
        for img1, img2, img3, current_state, gt in bar:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            img3 = img3.to(self.device)
            current_state = current_state.to(self.device)
            gt = gt.to(self.device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                action_pd, logits_pd = self.model(img1, img2, img3, current_state)
                loss, action_l, logits_l = self.weighted_loss(action_pd, logits_pd, gt)
            loss_sum += loss.item()
            action_loss_sum += action_l.item()
            logits_loss_sum += logits_l.item()
            
            all_action_pd.append(action_pd.float().cpu().numpy()) # convert to float32 for numpy
            all_logits_pd.append(torch.sigmoid(logits_pd).float().cpu().numpy())
            all_gt_action.append(gt[:, :3].float().cpu().numpy())
            all_gt_logit.append(gt[:, 3].float().cpu().numpy())
        self.model.train()
        
        y_pred_act = np.concatenate(all_action_pd)
        y_probs = np.concatenate(all_logits_pd).flatten()
        y_true_act = np.concatenate(all_gt_action)
        y_true_cls = np.concatenate(all_gt_logit)

        # 1. Action Error (L2 Euclidean Distance)
        # Axis=1 ensures we calculate distance per sample, then mean
        l2_errors = np.linalg.norm(y_pred_act - y_true_act, axis=1)

        try:
            auc = roc_auc_score(y_true_cls, y_probs)
        except ValueError:
            auc = 0.0 # Handle case where only one class is present

        thresholds = np.linspace(0.0, 1.0, 101)
    
        # Broadcast to shape (101, N_samples)
        pred_matrix = (y_probs[None, :] >= thresholds[:, None]).astype(int)
        y_true_broad = y_true_cls[None, :]

        # Vectorized confusion matrix calculation for all thresholds at once
        # Summing over axis=1 (samples)
        tps = np.sum((pred_matrix == 1) & (y_true_broad == 1), axis=1)
        fps = np.sum((pred_matrix == 1) & (y_true_broad == 0), axis=1)
        tns = np.sum((pred_matrix == 0) & (y_true_broad == 0), axis=1)
        fns = np.sum((pred_matrix == 0) & (y_true_broad == 1), axis=1)

        # Metrics calculation with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            recalls = tps / (tps + fns)
            precisions = tps / (tps + fps)
            accuracies = (tps + tns) / len(y_true_cls)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

        # Replace NaNs (0/0 cases) with 0
        recalls = np.nan_to_num(recalls)
        precisions = np.nan_to_num(precisions)
        f1_scores = np.nan_to_num(f1_scores)

        # Find Best Threshold (by Accuracy)
        best_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_idx]
        self.threshold = (best_threshold + 0.5) * 0.5 # smooth towards 0.5 to avoid extreme
        print(f'Updated threshold: {self.threshold:.4f} with accuracy {accuracies[best_idx]:.4f}')
        
        # 2. Completion Metrics (Classification)
        preds_binary = (y_probs > self.threshold).astype(int)
        
        acc = accuracy_score(y_true_cls, preds_binary)
        f1 = f1_score(y_true_cls, preds_binary, zero_division=0)
        
        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linestyle='--')
        plt.plot(thresholds, precisions, label='Precision', linestyle='--')
        
        plt.axvline(best_threshold, color='r', linestyle=':', label=f'Best Thresh ({best_threshold:.2f})')
        plt.axvline(self.threshold, color='g', linestyle='--', label=f'Current Thresh ({self.threshold:.2f})')
        plt.title(f'Metrics vs Threshold (Epoch {self.current_epoch})')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        filename = os.path.join(self.identifier, 'validation', f'threshold_analysis_epoch_{self.current_epoch}.png')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close() # Close figure to free memory

        # --- Reporting ---
        print(f'Losses      : Total {loss_sum / n_batches:.4f} | Action {action_loss_sum / n_batches:.4f} | Logits {logits_loss_sum / n_batches:.4f}')
        print(f'Action Error: Mean {np.mean(l2_errors):.4f} | Median {np.median(l2_errors):.4f} | Max {np.max(l2_errors):.4f}')
        print(f'Task Predict: AUC {auc:.4f} | Acc {acc:.4f} | F1 {f1:.4f} | TP {tps[best_idx]} | FP {fps[best_idx]} | TN {tns[best_idx]} | FN {fns[best_idx]}')
        print(f'Plot saved to {filename}\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VLA Transformer Model")
    parser.add_argument("--identifier", type=str, required=True, help="Identifier for the training run")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--init_lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--positive_weight", type=float, default=100.0, help="Positive class weight for BCE loss")
    parser.add_argument("--save_every", type=int, default=1, help="Save model every N epochs")
    parser.add_argument("--validate_every", type=int, default=1, help="Validate model every N epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    args = parser.parse_args()
    Trainer_VLA_Transformer(identifier=args.identifier, 
                            num_epochs=args.num_epochs,
                            init_lr=args.init_lr, 
                            positive_weight=args.positive_weight,
                            save_every=args.save_every,
                            validate_every=args.validate_every, 
                            batch_size=args.batch_size)()
    
    
# python train.py --num_epochs 10 --identifier run_001