import torch

from abc import ABC, abstractmethod
from tqdm import tqdm
import wandb

class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, wandb_record: bool = False):
        super().__init__()
        self.model = model
        self.wandb_record = wandb_record

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, setup_optimizer: dict):
        return torch.optim.AdamW(self.model.parameters(),  **setup_optimizer)

    def train(self, 
              num_epochs: int, 
              device: torch.device, 
              setup_optimizer: dict = dict(),
              setup_loss: dict = dict()) -> torch.Tensor:
        if self.wandb_record:
            # Log model architecture
            wandb.watch(self.model)

        # Start
        self.model.to(device)
        opt = self.get_optimizer(setup_optimizer=setup_optimizer)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)), position=0, leave=True)
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**setup_loss)
            loss.backward()
            opt.step()
            
            # Log metrics
            
            if self.wandb_record and (idx % 10 == 0):
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.model.parameters() if p.grad is not None]))
                
                metrics = {"train/loss": loss.item(), "grad_norm": grad_norm, "epoch": idx}
                wandb.log(metrics)
            
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        # Finish
        self.model.eval()
        
        # Close wandb run
        if self.wandb_project is not None:
            wandb.finish()
    
    def _init_wandb(self, **kwargs):
        return wandb.init(project=self.wandb_project, config={})