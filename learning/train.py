import torch

from abc import ABC, abstractmethod
from tqdm import tqdm
import wandb

class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, wandb_project = None):
        super().__init__()
        self.model = model
        self.wandb_project = wandb_project

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float, setup_optimizer: dict):
        return torch.optim.AdamW(self.model.parameters(), lr=lr, **setup_optimizer)

    def train(self, 
              num_epochs: int, 
              device: torch.device, 
              lr: float = 1e-3, 
              setup_optimizer: dict = dict(),
              setup_loss: dict = dict()) -> torch.Tensor:
        if self.wandb_project is not None:
            # Log model architecture
            wandb.watch(self.model)

        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr, setup_optimizer=setup_optimizer)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)), position=0, leave=True)
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**setup_loss)
            loss.backward()
            opt.step()
            
            # Log metrics
            metrics = {"train/loss": loss.item(), "epoch": idx}

            if self.wandb_project is not None:
                wandb.log(metrics)
            
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        # Finish
        self.model.eval()
        
        # Close wandb run
        if self.wandb_project is not None:
            wandb.finish()
    
    def _init_wandb(self, **kwargs):
        return wandb.init(project=self.wandb_project, config={})