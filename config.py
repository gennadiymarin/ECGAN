from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch


@dataclass
class TrainingConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    n_epochs: int = 50
    batch_size: int = 32
    dataset: str = 'cityscapes'
    data_path: str = '.'
    semantic_classes: int = 19
    c_hidden: int = 64
    ckpt_path = 'checkpoints'

    H = 128
    W = 256

    loss_coefs: Dict = field(default_factory=lambda: {
        'mma_G': 1,
        'mma_D': 1,
        'pix_contr': 1,
        'L1': 1,
        'sim': 1,
        'perc': 10,
        'discr_f': 10
    })

    lr = 0.0001
    beta1 = 0.
    beta2 = 0.999

    project_name = 'ECGAN'

    kernel_size = 3
    padding = 1

    LG_model_name = "nvidia/segformer-b0-finetuned-cityscapes-768-768"
