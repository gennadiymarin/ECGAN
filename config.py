from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TrainingConfig:
    device: str = 'cuda'
    n_epochs: int = 50
    batch_size: int = 64
    dataset: str = 'cityscapes'
    data_path: str = '.'  # TODO
    labels_cnt: int = 35
    c_hidden: int = 64
    loss_coefs: Dict = field(default_factory=lambda: {
        'mma_G': 1,
        'mma_D': 1,
        'pix_contr': 1,
        'L1': 1,
        'sim': 1,
        'perc': 10,
        'discr_f': 10
    })
