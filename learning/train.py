import torch
from inference.path import ConditionalProbabilityPath
from learning.mlp import MLPVectorField

from abc import ABC, abstractmethod
from tqdm import tqdm
import wandb

class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, device: torch.device):
        super().__init__()
        self.model = model.to(device)
        self.device = device


    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Start
        self.model.to(self.device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()

            if epoch % 2 == 0:
                pbar.set_description(f"Training: Epoch {epoch}, loss: {loss:.4f}")

        # Finish
        self.model.eval()

class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField, device: torch.device, **kwargs):
        super().__init__(model, device, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size).to(self.device)
        t = torch.rand(batch_size, 1).to(self.device)
        x = self.path.sample_conditional_path(z,t).to(self.device)

        u_model = self.model(x,t)
        u_ref = self.path.conditional_vector_field(x,z,t)

        return torch.norm(u_model - u_ref) / batch_size