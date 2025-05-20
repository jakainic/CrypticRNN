from dataclasses import dataclass
from typing import List, Optional

import torch

def get_device() -> str:
    """
    Determine the best available device for PyTorch.
    Returns 'mps' for Apple Silicon, 'cuda' for NVIDIA GPUs, or 'cpu' as fallback.
    """
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    return "cpu"  # Fallback to CPU

@dataclass
class ModelConfig:
    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    teacher_forcing_ratio: float = 0.5

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    test_size: float = 0.2
    random_seed: int = 42
    device: str = get_device()

@dataclass
class DataConfig:
    data_file: str = "clues_big.csv"
    special_tokens: Optional[List[str]] = None

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ['<pad>', '<sos>', '<eos>']

config = {
    'model': ModelConfig(),
    'training': TrainingConfig(),
    'data': DataConfig()
} 