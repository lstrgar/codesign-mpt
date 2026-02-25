import torch, os, math
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

class DecayingCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, initial_cycle_length, cycle_growth_factor, lr_decay_factor, min_lr=0, last_epoch=-1):
        self.initial_cycle_length = initial_cycle_length
        self.cycle_growth_factor = cycle_growth_factor
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def _get_cycle_info(self, epoch):
        current_epoch = 0
        current_cycle = 0
        cycle_length = self.initial_cycle_length
        while current_epoch + cycle_length <= epoch:
            current_epoch += cycle_length
            current_cycle += 1
            cycle_length = int(self.initial_cycle_length * (self.cycle_growth_factor ** current_cycle))
        return current_cycle, cycle_length, current_epoch

    def get_lr(self):
        cycle, cycle_length, cycle_start = self._get_cycle_info(self.last_epoch)
        cycle_progress = (self.last_epoch - cycle_start) / cycle_length
        decay = self.lr_decay_factor ** cycle
        cosine = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        return [self.min_lr + (base_lr * decay - self.min_lr) * cosine for base_lr in self.base_lrs]


class MLP(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dims: list[int], output_dim: int, gain: float):
        super().__init__()
        net = []
        dims = [sum(input_dims)] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1], dtype=torch.float32)
            nn.init.xavier_normal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
            net.append(layer)
            if i < len(dims) - 2:
                net.append(nn.LayerNorm(dims[i + 1]))
                net.append(nn.ReLU())
            else:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        mask = (x1 != 0.0)
        valid_counts = mask.sum(dim=-1, keepdim=True)
        mu = torch.sum(x1.masked_fill(~mask, 0), dim=-1, keepdim=True) / valid_counts
        x1 = torch.where(mask, x1 - mu, 0)
        x = torch.cat([x1, x2], dim=-1)
        return self.net(x)


class Brain(nn.Module):
    def __init__(self, config, workspace, outdir, device):
        super().__init__()
        self.config = config
        self.nms = len(workspace["masses"])
        self.nspr = len(workspace["springs"])
        self.outdir = outdir
        if self.outdir:
            os.makedirs(self.outdir, exist_ok=True)
        
        self.device = device
        torch.cuda.set_device(self.device)

        self.network = MLP(
            [self.nms, self.config["n_sin_waves"]],
            self.config["hidden_dims"],
            self.nspr,
            self.config["network_init_gain"]
        )
        
        self.network = self.network.to(self.device)
        if not self.config["needs_grad"]:
            self.network = torch.compile(self.network, mode="reduce-overhead")
        else:
            self.network = torch.compile(self.network)
        
        self.optimizer = torch.optim.Adam(
            params=self.network.parameters(),
            lr=self.config["init_lr"],
            betas=(self.config["beta1"], self.config["beta2"]),
            weight_decay=self.config["weight_decay"]
        )

        self.lr_scheduler = DecayingCosineAnnealingWarmRestarts(
            self.optimizer,
            self.config["cycle_0"],
            self.config["cycle_mult"],
            self.config["lr_decay"],
            self.config["min_lr"]
        )

        self.step_count = 0

        self.init_sine_waves()
        
    def save(self):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "step_count": self.step_count
            }, 
            f"{self.outdir}/{self.step_count}.pth",
        )

    def load(self, fpath, finetune=False):
        checkpoint = torch.load(fpath, weights_only=True)
        self.network.load_state_dict(checkpoint["network"])
        if not finetune:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
            self.step_count = checkpoint["step_count"]

    def init_sine_waves(self):
        sine_waves = torch.zeros(self.config["steps"], self.config["n_sin_waves"], device=self.device)
        for i in range(self.config["n_sin_waves"]):
            sine_waves[:, i] = torch.sin(
                self.config["spring_omega"] * torch.arange(self.config["steps"], device=self.device) * self.config["dt"] + \
                    2 * torch.pi / self.config["n_sin_waves"] * i
            )
        self.sine_waves = sine_waves

    def learn(self):
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config["grad_norm_clip"]
        )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.step_count += 1
    
    def forward(self, irrad: torch.Tensor, t: int) -> torch.Tensor:
        sw = self.sine_waves[t].unsqueeze(0).expand(irrad.shape[0], -1)
        return self.network(irrad, sw)