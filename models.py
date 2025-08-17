import logging
from typing import Tuple, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
        # Better initialization for attention layers
        nn.init.xavier_uniform_(self.attn.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v.weight, gain=1.0)
        if self.attn.bias is not None:
            nn.init.zeros_(self.attn.bias)
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: Decoder hidden state [batch_size, dec_hid_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hid_dim * 2]
        Returns:
            attention: Attention weights [batch_size, src_len]
        """
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate energy with improved scaling
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Calculate attention weights with temperature scaling
        attention = self.v(energy).squeeze(2)
        
        # Apply temperature scaling to increase variance
        attention = attention / self.temperature
        
        return F.softmax(attention, dim=1)

class ClueEncoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float = 0.1, max_length: int = 200):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # Add positional embeddings to distinguish positions
        self.positional_embedding = nn.Embedding(max_length, emb_dim)
        
        # Reduce bidirectional LSTM smoothing with single layer + residual connection
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True, num_layers=1, dropout=dropout, batch_first=True)
        
        # Layer normalization to stabilize representations
        self.layer_norm = nn.LayerNorm(enc_hid_dim * 2)
        
        # Projection layer to add non-linearity and position-specific processing
        self.position_projection = nn.Sequential(
            nn.Linear(enc_hid_dim * 2, enc_hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hid_dim * 2, enc_hid_dim * 2)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: Source sequence [batch_size, src_len]
        Returns:
            outputs: Encoder outputs [batch_size, src_len, enc_hid_dim * 2]
            hidden: Raw bidirectional hidden states [2, batch_size, enc_hid_dim]
        """
        batch_size, src_len = src.shape
        
        # Token embeddings
        token_embedded = self.embedding(src)  # [batch_size, src_len, emb_dim]
        
        # Positional embeddings
        positions = torch.arange(src_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        pos_embedded = self.positional_embedding(positions)  # [batch_size, src_len, emb_dim]
        
        # Combine token and positional embeddings
        embedded = self.dropout(token_embedded + pos_embedded)
        
        # LSTM processing
        lstm_outputs, hidden = self.rnn(embedded)
        
        # Layer normalization
        normalized_outputs = self.layer_norm(lstm_outputs)
        
        # Position-specific projection with residual connection
        projected_outputs = self.position_projection(normalized_outputs)
        final_outputs = normalized_outputs + projected_outputs  # Residual connection
        
        return final_outputs, hidden

class AnswerDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,  # Character vocabulary size
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float = 0.1,
        attention_temperature: float = 0.8  # Adjusted from 0.5 for better character diversity
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = Attention(enc_hid_dim, dec_hid_dim, temperature=attention_temperature)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # Character-level CNN for pattern recognition
        self.cnn = nn.Sequential(
            nn.Conv1d(emb_dim, dec_hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dec_hid_dim, dec_hid_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # LSTM for sequence modeling
        self.rnn = nn.LSTM(
            (enc_hid_dim * 2) + dec_hid_dim + emb_dim,
            dec_hid_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Output layers with increased capacity
        self.fc_out = nn.Sequential(
            nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, dec_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hid_dim, dec_hid_dim),  # Additional layer for more capacity
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hid_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

        # Projection layer to map encoder hidden state to decoder hidden size
        self.enc2dec_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def project_encoder_hidden(self, encoder_hidden):
        """
        Project bidirectional encoder hidden states to decoder hidden size
        Args:
            encoder_hidden: (h, c) from encoder, each [2, batch, enc_hid_dim] (single layer bidirectional)
        Returns:
            projected_hidden: (h, c) projected to decoder size [2, batch, dec_hid_dim]
        """
        h, c = encoder_hidden
        
        # Concatenate forward and backward directions for single layer
        # h, c shape: [2, batch, enc_hid_dim] -> [1, batch, enc_hid_dim*2]
        h_combined = torch.cat([h[0:1], h[1:2]], dim=2)  # [1, batch, enc_hid_dim*2]
        c_combined = torch.cat([c[0:1], c[1:2]], dim=2)  # [1, batch, enc_hid_dim*2]
        
        # Duplicate to create 2 layers for decoder
        h_stacked = h_combined.repeat(2, 1, 1)  # [2, batch, enc_hid_dim*2]
        c_stacked = c_combined.repeat(2, 1, 1)  # [2, batch, enc_hid_dim*2]
        
        # Project to decoder hidden size
        h_proj = torch.tanh(self.enc2dec_hidden(h_stacked))  # [2, batch, dec_hid_dim]
        c_proj = torch.tanh(self.enc2dec_hidden(c_stacked))
        
        return (h_proj, c_proj)

    def forward(
        self,
        input: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input: Input token indices [batch_size]
            hidden: Previous hidden state (h, c) [n_layers, batch_size, dec_hid_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hid_dim * 2]
        Returns:
            prediction: Output token probabilities [batch_size, output_dim]
            hidden: Updated hidden state (h, c) [n_layers, batch_size, dec_hid_dim]
        """
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        
        # Apply CNN to embedded input
        cnn_input = embedded.transpose(1, 2)  # [batch_size, emb_dim, 1]
        cnn_output = self.cnn(cnn_input)  # [batch_size, dec_hid_dim, 1]
        cnn_output = cnn_output.transpose(1, 2)  # [batch_size, 1, dec_hid_dim]
        
        # Calculate attention weights
        a = self.attention(hidden[0][-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        
        # Apply attention to encoder outputs
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, enc_hid_dim * 2]
        
        # Combine with embedded input and CNN output
        rnn_input = torch.cat((embedded, weighted, cnn_output), dim=2)
        
        # Process through RNN
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Combine all information for prediction
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        
        return prediction.squeeze(1), hidden

class Clue2Ans(nn.Module):
    def __init__(self, encoder: ClueEncoder, decoder: AnswerDecoder, device: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = 0  # <pad> token is always at index 0

    def forward(
        self,
        src: torch.Tensor,
        trg: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequences [batch_size, src_len]
            trg: Target sequences [batch_size, trg_len] (optional)
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            outputs: Sequence of output probabilities [batch_size, trg_len, output_dim]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1] if trg is not None else 50
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(
            batch_size,
            trg_len,
            trg_vocab_size,
            device=self.device
        )

        # Encode the source sequence
        encoder_outputs, encoder_hidden = self.encoder(src)

        # Project encoder hidden states to decoder hidden size
        hidden = self.decoder.project_encoder_hidden(encoder_hidden)

        # First input to the decoder is the <sos> token
        input = torch.ones(batch_size, dtype=torch.long, device=self.device)  # <sos> for each item in batch

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force and trg is not None else top1

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
            encoder_outputs, encoder_hidden = self.encoder(clue)
            
            # Project encoder hidden states to decoder hidden size
            hidden = self.decoder.project_encoder_hidden(encoder_hidden)
            
            # First input is <sos> token
            input = torch.tensor([sos_idx], device=self.device)
            
            generated = [sos_idx]
            
            for _ in range(max_length):
                # Generate next token
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                
                # Get the highest probability token
                next_token = output.argmax(1)
                
                # Add token to generated sequence
                generated.append(next_token.item())
                
                # Stop if <eos> token is generated
                if next_token.item() == eos_idx:
                    break
                    
                # Use generated token as next input
                input = next_token

        return torch.tensor(generated, device=self.device)

    # Checkpoint methods removed - handled by Trainer class 