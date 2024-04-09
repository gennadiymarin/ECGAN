from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch


@dataclass
class TrainingConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs: int = 50
    batch_size: int = 64
    dataset: str = 'cityscapes'
    data_path: str = '.'  # TODO
    semantic_classes: int = 35
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

    lr = 0.0001
    beta1 = 0.
    beta2 = 0.999

    project_name = 'ECGAN'

    kernel_size = 3
    padding = 1

    LG_model_name = "nvidia/segformer-b0-finetuned-cityscapes-768-768"

    cityscapes_classes = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    cityscapes_palette = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]