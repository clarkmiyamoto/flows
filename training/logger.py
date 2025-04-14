import wandb

class WandbLogger:
    def __init__(self, project: str, config: dict = None):
        self.run = wandb.init(project=project, config=config or {})

    def log(self, metrics: dict, step: int):
        self.run.log(metrics, step=step)

    def watch(self, model):
        wandb.watch(model)

    def finish(self):
        wandb.finish()
