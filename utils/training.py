import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from .logger import Logger

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class LRScheduler:
    def __init__(self, optimizer, warmup_steps: int, max_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / (10000 - self.warmup_steps)
            lr = self.max_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        diffusion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        logger: Logger
    ):
        self.model = model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = LRScheduler(
            self.optimizer,
            config.training.warmup_steps,
            config.training.learning_rate
        )
        
        self.ema = EMA(model)
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for input_seq, target_seq in pbar:
            input_seq = input_seq.to(self.device, non_blocking=True)
            target_seq = target_seq.to(self.device, non_blocking=True)
            
            if num_batches == 0:
                self.logger.info(f"Input seq shape: {input_seq.shape}")
                self.logger.info(f"Target seq shape: {target_seq.shape}")
            
            batch_loss = 0
            
            if len(input_seq.shape) == 6:  # [batch_size, input_frames, 1, channels, H, W]
                batch_size, input_frames, _, channels, height, width = input_seq.shape
                cond = input_seq.squeeze(2).view(batch_size, input_frames * channels, height, width)
            else:
                raise ValueError(f"Unexpected input_seq shape: {input_seq.shape}")
            
            if len(target_seq.shape) == 6:  # [batch_size, output_frames, 1, channels, H, W]
                batch_size, output_frames, _, channels, height, width = target_seq.shape
                target_seq = target_seq.squeeze(2)
            else:
                raise ValueError(f"Unexpected target_seq shape: {target_seq.shape}")

            for i in range(output_frames):
                target = target_seq[:, i]
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device)
                loss = self.diffusion.p_losses(target, cond, t)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
                self.optimizer.step()
                self.scheduler.step()
                self.ema.update()
                
                batch_loss += loss.item()
                self.global_step += 1
            
            total_loss += batch_loss
            num_batches += 1
            pbar.set_postfix({'loss': batch_loss / output_frames})
        
        avg_loss = total_loss / (num_batches * output_frames)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for input_seq, target_seq in self.val_loader:
            input_seq = input_seq.to(self.device, non_blocking=True)
            target_seq = target_seq.to(self.device, non_blocking=True)
            
            batch_loss = 0
            
            if len(input_seq.shape) == 6:
                batch_size, input_frames, _, channels, height, width = input_seq.shape
                cond = input_seq.squeeze(2).view(batch_size, input_frames * channels, height, width)
            else:
                raise ValueError(f"Unexpected input_seq shape: {input_seq.shape}")
            
            if len(target_seq.shape) == 6:
                batch_size, output_frames, _, channels, height, width = target_seq.shape
                target_seq = target_seq.squeeze(2)
            else:
                raise ValueError(f"Unexpected target_seq shape: {target_seq.shape}")
            
            for i in range(output_frames):
                target = target_seq[:, i]
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device)
                loss = self.diffusion.p_losses(target, cond, t)
                batch_loss += loss.item()
            
            total_loss += batch_loss
            num_batches += 1
        
        avg_loss = total_loss / (num_batches * output_frames)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_shadow': self.ema.shadow,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{self.epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ema.shadow = checkpoint['ema_shadow']
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(self.config.output_dir, 'loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Loss plot saved: {plot_path}")
    
    def train(self):
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Total epochs: {self.config.training.epochs}")
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            if epoch % self.config.training.save_every == 0 or is_best:
                self.save_checkpoint(is_best)
            if epoch % (self.config.training.save_every * 2) == 0:
                self.plot_losses()
        self.logger.info("Training completed!")
        self.plot_losses()