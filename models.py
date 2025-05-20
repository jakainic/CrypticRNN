import logging
from typing import Tuple

import torch
import torch.nn as nn

from config import config

logger = logging.getLogger(__name__)

class ClueEncoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Source sequence [batch_size, src_len]
        Returns:
            hidden: Final hidden state [1, batch_size, hid_dim]
        """
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        _, hidden = self.rnn(embedded)  # hidden: [1, batch_size, hid_dim]
        return hidden

class AnswerDecoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Input token indices [batch_size]
            hidden: Hidden state [1, batch_size, hid_dim]
        Returns:
            prediction: Output token probabilities [batch_size, output_dim]
            hidden: Updated hidden state [1, batch_size, hid_dim]
        """
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))  # [batch_size, 1, emb_dim]
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hid_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        return prediction, hidden

class Clue2Ans(nn.Module):
    def __init__(self, encoder: ClueEncoder, decoder: AnswerDecoder, pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(
        self,
        clues: torch.Tensor,
        answers: torch.Tensor,
        teacher_forcing_ratio: float = None
    ) -> torch.Tensor:
        """
        Args:
            clues: Source sequences [batch_size, src_len]
            answers: Target sequences [batch_size, trg_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            outputs: Sequence of output probabilities [batch_size, trg_len, output_dim]
        """
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = config['model'].teacher_forcing_ratio

        batch_size, answers_len = answers.shape
        answers_vocab_size = self.decoder.fc_out.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(
            batch_size,
            answers_len,
            answers_vocab_size,
            device=clues.device
        )

        # Encode the source sequence
        hidden = self.encoder(clues)

        # First input to the decoder is the <sos> token
        input = answers[:, 0]

        for t in range(1, answers_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = answers[:, t] if teacher_force else top1

        return outputs

    def generate(
        self,
        clue: torch.Tensor,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """
        Generate an answer sequence for a single clue
        Args:
            clue: Input sequence [src_len]
            max_length: Maximum length of generated sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
        Returns:
            generated: Generated sequence [seq_len]
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            clue = clue.unsqueeze(0)  # [1, src_len]
            
            # Encode the source sequence
            hidden = self.encoder(clue)  # [1, 1, hid_dim]
            
            # First input is <sos> token
            input = torch.tensor([sos_idx], device=clue.device)
            
            generated = [sos_idx]
            
            for _ in range(max_length):
                # Generate next token
                output, hidden = self.decoder(input, hidden)
                
                # Get the highest probability token
                next_token = output.argmax(1)
                
                # Add token to generated sequence
                generated.append(next_token.item())
                
                # Stop if <eos> token is generated
                if next_token.item() == eos_idx:
                    break
                    
                # Use generated token as next input
                input = next_token

        return torch.tensor(generated, device=clue.device) 