"""
The processor will:
1. Load CSV with at least 'clue' and 'answer' columns
2. Clean and normalize text (remove diacritics, keep only alphabetic)
3. Remove rows with missing clues/answers
4. Remove answers containing numbers
5. Tokenize clues (word-level) and answers (character-level)
6. Build vocabularies and create PyTorch DataLoaders
"""

from collections import Counter
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import unicodedata
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rnn'))
from config import config, get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vocabulary:
    """Manages vocabulary building and token-index mapping"""
    def __init__(self, special_tokens: List[str] = None):
        self.token2idx = {}
        self.idx2token = {}
        self.token_counts = Counter()
        
        if special_tokens:
            for token in special_tokens:
                self.add_token(token)
    
    def add_token(self, token: str) -> int:
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]
    
    def build_from_tokens(self, tokens: List[str], min_freq: int = 1):
        """Build vocabulary from list of token lists"""
        # Count tokens
        self.token_counts = Counter(tokens)
        
        # Add tokens that meet frequency threshold
        for token, count in self.token_counts.items():
            if count >= min_freq:
                self.add_token(token)
        
        logger.info(f"Vocabulary size: {len(self.token2idx)}")
        logger.info(f"Sample vocabulary: {list(self.token2idx.items())[:10]}")
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.token2idx.get(token, self.token2idx['<pad>']) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx2token.get(idx, '<unk>') for idx in indices]

    def __len__(self):
        return len(self.token2idx)
    
    def get_token(self, idx):
        return self.idx2token.get(idx, '<UNK>')
    
    def get_idx(self, token):
        return self.token2idx.get(token, self.token2idx.get('<UNK>', 0))

class CrypticDataset(Dataset):
    """PyTorch Dataset for cryptic clues and answers"""
    def __init__(self, clues: List[torch.Tensor], answers: List[torch.Tensor]):
        self.clues = clues
        self.answers = answers

    def __len__(self) -> int:
        return len(self.clues)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.clues[idx], self.answers[idx]

class DataProcessor:
    """Handles data loading, preprocessing, and dataset creation"""
    def __init__(self, data_config=config['data']):
        self.config = data_config
        self.clue_vocab = Vocabulary(data_config.special_tokens)
        self.answer_vocab = Vocabulary(data_config.special_tokens)
        self.device = get_device()
        self.test_data = None  # Store test data for example generation
        logger.info("Initialized DataProcessor")

    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the raw data"""
        logger.info(f"Loading data from {self.config.data_file}")
        df = pd.read_csv(self.config.data_file, on_bad_lines='skip')
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Remove rows with missing values
        df = df.loc[~df.clue.isna() & ~df.answer.isna()]
        logger.info(f"After removing NaN values: {len(df)} rows")
        
        # Clean answers
        df['answer'] = df['answer'].apply(self.clean_text)
        num_mask = df['answer'].astype(str).str.contains(r'\d')
        df = df.loc[~num_mask]
        logger.info(f"After removing numeric answers: {len(df)} rows")
        
        return df

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return text
        # Remove diacritics
        normalized = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        # Keep only alphabetic characters
        return ''.join(c for c in text if c.isalpha())

    @staticmethod
    def tokenize_clue(text: str) -> List[str]:
        """Smart tokenization that preserves contractions but separates other punctuation"""
        import re
        
        # Convert to lowercase first
        text = text.lower()
        
        # Clean and elegant regex pattern:
        # \w+(?:'\w+)? - matches words with optional apostrophe contractions (don't, can't, it's)
        # [^\w\s] - matches any punctuation (non-word, non-whitespace) as separate tokens
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        
        return tokens

    @staticmethod
    def tokenize_answer(text: str) -> List[str]:
        """Tokenize answer at character level"""
        return list(text.lower())

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training and testing"""
        logger.info("Starting data preparation...")
        
        # Load and clean data
        df = self.load_and_clean_data()
        
        # Tokenize
        logger.info("Tokenizing clues and answers...")
        clue_tokens = [self.tokenize_clue(clue) for clue in df['clue']]
        answer_tokens = [self.tokenize_answer(ans) for ans in df['answer']]
        logger.info("Tokenization complete")

        # Build vocabularies
        logger.info("Building vocabularies...")
        # Flatten the token lists for vocabulary building
        flat_clue_tokens = [token for clue in clue_tokens for token in clue]
        flat_answer_tokens = [token for answer in answer_tokens for token in answer]
        self.clue_vocab.build_from_tokens(flat_clue_tokens)
        self.answer_vocab.build_from_tokens(flat_answer_tokens)
        logger.info(f"Vocabulary sizes - Clues: {len(self.clue_vocab)}, Answers: {len(self.answer_vocab)}")
        logger.info(f"Sample clue vocabulary: {list(self.clue_vocab.token2idx.items())[:10]}")
        logger.info(f"Sample answer vocabulary: {list(self.answer_vocab.token2idx.items())[:10]}")

        # Encode sequences
        logger.info("Encoding sequences...")
        clue_ids = [self.encode_sequence(seq, self.clue_vocab) for seq in clue_tokens]
        answer_ids = [self.encode_sequence(seq, self.answer_vocab) for seq in answer_tokens]
        logger.info("Sequence encoding complete")

        # Split data
        logger.info("Splitting data into train and test sets...")
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            clue_ids, answer_ids,
            test_size=config['training'].test_size,
            random_state=config['training'].random_seed
        )
        logger.info(f"Split complete - Train: {len(train_inputs)} samples, Test: {len(test_inputs)} samples")

        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader = self.create_dataloader(train_inputs, train_targets, shuffle=True)
        test_loader = self.create_dataloader(test_inputs, test_targets, shuffle=False)
        logger.info("Data loaders created successfully")

        return train_loader, test_loader

    @staticmethod
    def encode_sequence(seq: List[str], vocab: Vocabulary, add_eos: bool = True) -> torch.Tensor:
        """Encode a sequence of tokens using the vocabulary"""
        tokens = ['<sos>'] + seq + (['<eos>'] if add_eos else [])
        indices = vocab.encode(tokens)
        return torch.tensor(indices)

    def create_dataloader(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        shuffle: bool
    ) -> DataLoader:
        """Create a DataLoader for the given inputs and targets"""
        logger.info(f"Creating DataLoader with batch size {config['training'].batch_size}")
        dataset = CrypticDataset(inputs, targets)
        return DataLoader(
            dataset,
            batch_size=config['training'].batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_batch
        )

    def collate_batch(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for DataLoader"""
        inputs, targets = zip(*batch)
        pad_idx = 0  # <pad> is at index 0
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
        return inputs_padded.to(self.device), targets_padded.to(self.device)

    @property
    def vocab_sizes(self) -> Tuple[int, int]:
        """Get the sizes of the clue and answer vocabularies"""
        return len(self.clue_vocab), len(self.answer_vocab)

    def decode_answer(self, tensor: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode a tensor of token indices back to a string"""
        if tensor.dim() == 2:
            tensor = tensor[0]
            
        indices = tensor.cpu().tolist()
        tokens = self.answer_vocab.decode(indices)
        
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.config.special_tokens and t != '<pad>']
        
        return ''.join(tokens)

    def decode_clue(self, tensor: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode a tensor of token indices back to a string of words"""
        if tensor.dim() == 2:
            tensor = tensor[0]
            
        indices = tensor.cpu().tolist()
        tokens = self.clue_vocab.decode(indices)
        
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.config.special_tokens and t != '<pad>']
        
        return ' '.join(tokens)

    def get_random_test_example(self) -> Tuple[str, str]:
        """Get a random example from the test set"""
        if self.test_data is None:
            # Load and clean data
            df = self.load_and_clean_data()
            
            # Split into train and test
            train_data, test_data = train_test_split(
                df,
                test_size=config['training'].test_size,
                random_state=config['training'].random_seed
            )
            self.test_data = test_data
        
        # Get random example
        example = self.test_data.sample(n=1).iloc[0]
        return example['clue'], example['answer'] 