from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import torch

def get_device() -> str:
    """Get the best available device for PyTorch"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

@dataclass
class ModelConfig:
    embedding_dim: int = 128  # Increased from 64
    enc_hid_dim: int = 128    # Increased from 64
    dec_hid_dim: int = 128    # Increased from 64
    dropout: float = 0.1
    num_layers: int = 1
    teacher_forcing_ratio: float = 0.5
    # Teacher forcing schedule parameters
    use_teacher_forcing_schedule: bool = True
    initial_teacher_forcing: float = 0.95
    final_teacher_forcing: float = 0.3
    # hidden_dim: int = 64  # Deprecated, use enc_hid_dim and dec_hid_dim

@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 0.0001
    min_lr: float = 1e-6
    test_size: float = 0.2
    random_seed: int = 42
    device: str = get_device()
    
    # Learning rate scheduling
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 2
    lr_decay_factor: float = 0.1
    lr_patience: int = 3

    # Checkpoint settings
    save_every_n_epochs: int = 1
    checkpoint_dir: str = 'checkpoints'

@dataclass
class DataConfig:
    data_file: str = 'clues_big.csv'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_clue_length: int = 50
    max_answer_length: int = 20
    min_freq: int = 1
    special_tokens: Optional[List[str]] = None

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ['<pad>', '<sos>', '<eos>']

config: Dict[str, Any] = {
    'model': ModelConfig(),
    'training': TrainingConfig(),
    'data': DataConfig()
} 