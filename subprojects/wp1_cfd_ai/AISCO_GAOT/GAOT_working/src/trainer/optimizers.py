import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass, field
import psutil
import os

###############
# Config
###############
@dataclass
class OptimizerargsConfig:
    lr: float = 1e-3                    # Initial learning rate
    weight_decay: float = 1e-3          # L2 regularization (weight decay) coefficient
    epoch: int = 100                    # Total number of training epochs
    loss_scale: float = 1.0             # Loss scaling factor
    eval_every_eps: int = 2             # Evaluate every n epochs
    scheduler: str = "mix"              # Learning rate scheduler type, support ['step', 'cos', 'exp', 'mix']
    early_save_metric: str = 'val'      # Metric for early stopping, support ['train', 'val']
    # for mix scheduler
    max_lr: float = 1e-2                # Maximum learning rate for the cosine annealing phase
    min_lr: float = 1e-5                # Minimum learning rate for the cosine annealing phase
    final_lr: float = 1e-5              # Final learning rate for the exponential decay phase
    # for step scheduler
    scheduler_step_size: int = 100      # Step size (number of epochs) for StepLR scheduler
    scheduler_gamma: float = 0.8        # Multiplicative factor for learning rate decay in StepLR scheduler
    scheduler_T_max: int = 100          # Maximum number of iterations (usually total epochs) for CosineAnnealingLR scheduler
    scheduler_eta_min: float = 1e-4     # Minimum learning rate for CosineAnnealingLR scheduler

    # checkpoint/save settings
    save_every_epochs: int = 0          # Save checkpoint every N epochs (0=disabled)
    save_every_steps: int = 0           # Save checkpoint every N training steps (0=disabled)

###############
# Scheduler
###############
class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, cosine_epochs, exp_decay_epochs,
                 initial_lr, max_lr, min_lr, final_lr, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        self.exp_decay_epochs = exp_decay_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.final_lr = final_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # first phase (warm up): initial_lr to max_lr
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.last_epoch / max(1, self.warmup_epochs - 1))
        elif self.last_epoch < self.warmup_epochs + self.cosine_epochs:
            # second stage (cosine): max_lr to min_lr
            epoch = self.last_epoch - self.warmup_epochs
            cosine_ratio = (1 + np.cos(np.pi * epoch / self.cosine_epochs)) / 2
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_ratio
        else:
            # third stage (expontential)： min_lr to final_lr
            epoch = self.last_epoch - self.warmup_epochs - self.cosine_epochs
            decay_steps = max(1, self.exp_decay_epochs - 1)
            lr = self.min_lr * ((self.final_lr / self.min_lr) ** (epoch / decay_steps))
        return [lr for _ in self.optimizer.param_groups]

###############
# Optimizer
###############
class AdamOptimizer:
    optimizer: torch.optim.Adam
    scheduler: torch.optim.lr_scheduler.StepLR
    epoch: int
    batch_size: int
    lr: float
    loss_scale: float
    eval_every_eps: int

    def __init__(self, params, config):
        self.optimizer = torch.optim.Adam(params, lr=config.lr)
        self.epoch = config.epoch
        self.lr = config.lr
        self.loss_scale = config.loss_scale
        self.eval_every_eps = config.eval_every_eps
        self.early_save_metric = config.early_save_metric.lower()

        if self.early_save_metric not in ['train', 'val']:
            raise ValueError("`early_save_metric` must be either 'train' or 'val'.")

        if config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        elif config.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.scheduler_T_max, eta_min=config.scheduler_eta_min)
        elif config.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.scheduler_gamma)
        elif config.scheduler == 'mix':
            warmup_epochs = int(0.02 * self.epoch)
            cosine_epochs = int(0.96 * self.epoch)
            exp_decay_epochs = self.epoch - warmup_epochs - cosine_epochs
            if warmup_epochs == 0:
                warmup_epochs = 1
                cosine_epochs -= 1
            if exp_decay_epochs == 0:
                exp_decay_epochs = 1
                cosine_epochs -= 1
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                total_epochs=self.epoch,
                warmup_epochs=warmup_epochs,
                cosine_epochs=cosine_epochs,
                exp_decay_epochs=exp_decay_epochs,
                initial_lr=self.lr,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                final_lr=config.final_lr
            )
        else:
            self.scheduler = None

    def optimize(self, trainer: 'TrainerBase',
                        description: str = "AdamOptimizer",
                        color: str = "blue"):
        time_total = 0.0
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        epochs = []
        val_epochs = []
        val_losses = []

        # Only rank 0 should display progress bar
        disable_pbar = hasattr(trainer.setup_config, 'rank') and trainer.setup_config.rank != 0
        pbar = tqdm(total=self.epoch, desc=description, colour = color, disable=disable_pbar)
        for epoch in range(self.epoch):
            # Ensure DistributedSampler shuffles differently each epoch
            train_sampler = getattr(trainer.train_loader, 'sampler', None)
            if hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)
            trainer.model.train()
            total_loss = 0.0
            for batch in trainer.train_loader:
                start_time = time.time()
                self.optimizer.zero_grad()
                train_loss = trainer.train_step(batch)
                train_loss.backward()
                self.optimizer.step()
                total_loss += train_loss.item()
                if isinstance(trainer.device, torch.device) and trainer.device.type == 'cuda':
                    torch.cuda.synchronize()
                time_total += time.time() - start_time

            if self.scheduler is not None:
                self.scheduler.step()

            pbar.update(1)

            if (epoch + 1) % self.eval_every_eps == 0:
                train_loss = total_loss / len(trainer.train_loader)
                losses.append(train_loss)
                epochs.append(epoch)
                val_loss = trainer.validate(trainer.val_loader)
                if not disable_pbar:
                    pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
                val_losses.append(val_loss)
                val_epochs.append(epoch)

                if self.early_save_metric == 'val':
                    current_loss = val_loss
                else:
                    current_loss = train_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state = deepcopy(trainer.model.state_dict())

        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        if not disable_pbar:
            pbar.close()

        return {
            "train":{
                "loss": losses,
                "epoch": epochs,
            },
            "valid":{
                "loss": val_losses,
                "epoch": val_epochs,
            },
            "best":{
                "epoch": best_epoch,
                "loss": best_loss,
            },
            "time": time_total
        }

class AdamWOptimizer:
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler._LRScheduler
    epoch: int
    batch_size: int
    lr: float
    loss_scale: float
    eval_every_eps: int

    def __init__(self, params, config):
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        self.epoch = config.epoch
        self.lr = config.lr
        self.loss_scale = config.loss_scale
        self.eval_every_eps = config.eval_every_eps
        self.early_save_metric = config.early_save_metric.lower()

        if self.early_save_metric not in ['train', 'val']:
            raise ValueError("`early_save_metric` must be either 'train' or 'val'.")

        if config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        elif config.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.scheduler_T_max, eta_min=config.scheduler_eta_min)
        elif config.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.scheduler_gamma)
        elif config.scheduler == 'mix':
            warmup_epochs = int(0.02 * self.epoch)
            cosine_epochs = int(0.90 * self.epoch)
            exp_decay_epochs = self.epoch - warmup_epochs - cosine_epochs
            if warmup_epochs == 0:
                warmup_epochs = 1
                cosine_epochs -= 1
            if exp_decay_epochs == 0:
                exp_decay_epochs = 1
                cosine_epochs -= 1
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                total_epochs=self.epoch,
                warmup_epochs=warmup_epochs,
                cosine_epochs=cosine_epochs,
                exp_decay_epochs=exp_decay_epochs,
                initial_lr=self.lr,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                final_lr=config.final_lr
            )
        else:
            self.scheduler = None

    @staticmethod
    def _compute_grad_norms(model):
        """Compute per-component gradient norms for monitoring."""
        norms = {}
        total_norm_sq = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                pnorm = p.grad.data.norm(2).item()
                total_norm_sq += pnorm ** 2
                # Group by top-level component (encoder, processor, decoder, attn_pool, etc.)
                component = name.split('.')[0]
                norms.setdefault(component, 0.0)
                norms[component] += pnorm ** 2
        norms = {f'grad_norm/{k}': v ** 0.5 for k, v in norms.items()}
        norms['grad_norm/total'] = total_norm_sq ** 0.5
        return norms

    def optimize(self, trainer: 'TrainerBase',
                 description: str = "AdamWOptimizer",
                 color: str = "blue"):
        time_total = 0.0
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        epochs = []
        val_epochs = []
        val_losses = []

        # Gradient clipping threshold
        max_grad_norm = 1.0

        # Only rank 0 should display progress bar
        disable_pbar = hasattr(trainer.setup_config, 'rank') and trainer.setup_config.rank != 0
        pbar = tqdm(total=self.epoch, desc=description, colour=color, disable=disable_pbar)
        global_step = 0
        save_every_steps = getattr(trainer.optimizer_config.args, 'save_every_steps', 0)
        for epoch in range(self.epoch):
            # Ensure DistributedSampler shuffles differently each epoch
            train_sampler = getattr(trainer.train_loader, 'sampler', None)
            if hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)
            trainer.model.train()
            total_loss = 0.0
            epoch_grad_norms = {}
            num_batches = 0
            for batch in trainer.train_loader:
                start_time = time.time()
                self.optimizer.zero_grad()
                train_loss = trainer.train_step(batch)
                train_loss.backward()

                # Gradient clipping to stabilise training
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_grad_norm)

                # Accumulate gradient norms (sampled every batch)
                if not disable_pbar:
                    batch_norms = self._compute_grad_norms(trainer.model)
                    for k, v in batch_norms.items():
                        epoch_grad_norms.setdefault(k, 0.0)
                        epoch_grad_norms[k] += v
                    num_batches += 1

                self.optimizer.step()

                total_loss += train_loss.detach()
                global_step += 1

                # Save checkpoint every N steps if configured (rank 0 only)
                if (save_every_steps > 0
                    and global_step % save_every_steps == 0
                    and getattr(trainer.setup_config, 'rank', 0) == 0):
                    try:
                        step_loss = train_loss.detach().cpu().item()
                        trainer.save_ckpt()
                        import logging
                        logging.getLogger(__name__).info(
                            f"Step checkpoint saved at global_step={global_step} (epoch {epoch+1}), loss={step_loss:.4f}"
                        )
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"Failed to save step checkpoint: {e}")

            if self.scheduler is not None:
                self.scheduler.step()

            pbar.update(1)

            if (epoch + 1) % self.eval_every_eps == 0:
                train_loss = total_loss.cpu().item() / len(trainer.train_loader)
                losses.append(train_loss)
                epochs.append(epoch)
                val_loss = trainer.validate(trainer.val_loader) # TODO: change to val_loader
                current_lr = self.optimizer.param_groups[0]['lr']

                if not disable_pbar:
                    pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
                metrics = {
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'lr': current_lr,
                }
                # Add average gradient norms for this epoch
                if epoch_grad_norms and num_batches > 0:
                    for k, v in epoch_grad_norms.items():
                        metrics[k] = v / num_batches
                val_losses.append(val_loss)
                val_epochs.append(epoch)

                metrics['memory'] = psutil.Process(os.getpid()).memory_info().rss/1e9
                trainer.log_metrics(metrics, step = epoch + 1)

                if self.early_save_metric == 'val':
                    current_loss = val_loss
                else:
                    current_loss = train_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state = deepcopy(trainer.model.state_dict())

                 # Save checkpoint every N epochs if configured (rank 0 only)
                if (getattr(trainer.optimizer_config.args, 'save_every_epochs', 0) > 0
                    and (epoch + 1) % trainer.optimizer_config.args.save_every_epochs == 0
                    and getattr(trainer.setup_config, 'rank', 0) == 0):
                    try:
                        trainer.save_checkpoint_artifact(epoch + 1, train_loss, val_loss, is_best=False)
                        if getattr(trainer.setup_config, 'test_during_training', False):
                            trainer.test()
                    except Exception:
                        pass

        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        if not disable_pbar:
            pbar.close()

        return {
            "train": {
                "loss": losses,
                "epoch": epochs,
            },
            "valid": {
                "loss": val_losses,
                "epoch": val_epochs,
            },
            "best": {
                "epoch": best_epoch,
                "loss": best_loss,
            },
            "time": time_total
        }
