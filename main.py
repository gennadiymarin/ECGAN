from config import TrainingConfig
from trainer import Trainer


config = TrainingConfig()
trainer = Trainer(config)
trainer.train()

