from collections import Counter
import logging
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import unicodedata

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vocabulary:
    """Custom vocabulary class to replace torchtext"""
    def __init__(self, special_tokens: List[str]):
        self.itos = special_tokens.copy()  # index to string
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(special_tokens)}  # string to index
        self.default_index = 0  # default to <pad>
        
    def build_from_tokens(self, tokens: List[List[str]], min_freq: int = 1):
        """Build vocabulary from list of token lists"""
        # Count token frequencies
        counter = Counter()
        for token_list in tokens:
            counter.update(token_list)
        
        # Add tokens that meet minimum frequency
        for token, count in counter.items():
            if count >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
    
    def __len__(self) -> int:
        return len(self.itos)
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.stoi.get(token, self.default_index) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices back to tokens"""
        return [self.itos[idx] for idx in indices]
    
    def set_default_index(self, index: int):
        """Set the default index for unknown tokens"""
        self.default_index = index

class TextPreprocessor:
    @staticmethod
    def strip_diacritics(text: str) -> str:
        if not isinstance(text, str):
            return text
        normalized = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

    @staticmethod
    def clean_answer(text: str) -> str:
        if not isinstance(text, str):
            return text
        return ''.join(c for c in text if c.isalpha())

    @staticmethod
    def tokenize_chars(text: str) -> List[str]:
        return list(text.lower())

class CrypticDataset(Dataset):
    def __init__(self, clues: List[torch.Tensor], answers: List[torch.Tensor]):
        self.clues = clues
        self.answers = answers

    def __len__(self) -> int:
        return len(self.clues)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.clues[idx], self.answers[idx]

class DataProcessor:
    def __init__(self, data_config=config['data']):
        self.config = data_config
        self.preprocessor = TextPreprocessor()
        self.clue_vocab = Vocabulary(data_config.special_tokens)
        self.answer_vocab = Vocabulary(data_config.special_tokens)

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.config.data_file}")
        df = pd.read_csv(self.config.data_file, on_bad_lines='skip')
        df = df.loc[~df.clue.isna() & ~df.answer.isna()]
        
        logger.info("Preprocessing answers...")
        df['answer'] = df['answer'].apply(self.preprocessor.strip_diacritics)
        num_mask = df['answer'].astype(str).str.contains(r'\d')
        df = df.loc[~num_mask]
        df['answer'] = df['answer'].apply(self.preprocessor.clean_answer)
        
        return df

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        df = self.load_data()
        
        # Tokenize
        clue_tokens = [self.preprocessor.tokenize_chars(clue) for clue in df['clue']]
        answer_tokens = [self.preprocessor.tokenize_chars(ans) for ans in df['answer']]

        # Build vocabularies
        logger.info("Building vocabularies...")
        self.clue_vocab.build_from_tokens(clue_tokens)
        self.answer_vocab.build_from_tokens(answer_tokens)

        # Set default index to <pad>
        self.clue_vocab.set_default_index(0)  # <pad> is at index 0
        self.answer_vocab.set_default_index(0)

        # Encode sequences
        clue_ids = [self.encode_sequence(seq, self.clue_vocab) for seq in clue_tokens]
        answer_ids = [self.encode_sequence(seq, self.answer_vocab) for seq in answer_tokens]

        # Split data
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            clue_ids, answer_ids,
            test_size=config['training'].test_size,
            random_state=config['training'].random_seed
        )

        # Create data loaders
        train_loader = self.create_dataloader(train_inputs, train_targets, shuffle=True)
        test_loader = self.create_dataloader(test_inputs, test_targets, shuffle=False)

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
        inputs, targets = zip(*batch)
        pad_idx = 0  # <pad> is at index 0
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
        return inputs_padded, targets_padded

    @property
    def vocab_sizes(self) -> Tuple[int, int]:
        return len(self.clue_vocab), len(self.answer_vocab)

    def decode_answer(self, tensor: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode a tensor of token indices back to a string.
        Args:
            tensor: Tensor of token indices
            skip_special_tokens: Whether to skip special tokens (<sos>, <eos>, <pad>)
        Returns:
            decoded_text: The decoded string
        """
        if tensor.dim() == 2:
            # If batch dimension present, take first sequence
            tensor = tensor[0]
            
        indices = tensor.cpu().tolist()
        tokens = self.answer_vocab.decode(indices)
        
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.config.special_tokens]
        
        return ''.join(tokens)

    def calculate_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        return_examples: bool = False,
        num_examples: int = 5
    ) -> Tuple[float, List[Tuple[str, str]]]:
        """
        Calculate accuracy between predicted and target answers.
        Args:
            predictions: Tensor of predicted token indices [batch_size, seq_len]
            targets: Tensor of target token indices [batch_size, seq_len]
            return_examples: Whether to return example predictions
            num_examples: Number of examples to return if return_examples is True
        Returns:
            accuracy: Accuracy score
            examples: List of (predicted, target) string pairs if return_examples is True
        """
        correct = 0
        total = 0
        examples = []
        
        for pred, target in zip(predictions, targets):
            pred_text = self.decode_answer(pred)
            target_text = self.decode_answer(target)
            
            if pred_text == target_text:
                correct += 1
            total += 1
            
            if return_examples and len(examples) < num_examples:
                examples.append((pred_text, target_text))
        
        accuracy = correct / total if total > 0 else 0
        
        if return_examples:
            return accuracy, examples
        return accuracy, [] 