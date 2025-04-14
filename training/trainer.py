from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from logger import WandbLogger

class Trainer(ABC):
    def __init__(self, 
                 model: torch.nn.Module, 
                 logger: WandbLogger = None):
        super().__init__()
        self.model = model
        self.logger = logger

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_optimizer(self, lr: float):
        pass

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        if self.logger:
            self.logger.watch(self.model)

        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()

            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

            if self.logger:
                self.logger.log({"loss": loss.item()}, step=idx)

        self.model.eval()
        if self.logger:
            self.logger.finish()
