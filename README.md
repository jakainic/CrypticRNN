# Cryptic RNN

A sequence-to-sequence neural network for solving cryptic crossword clues. 

This will be part of a larger project with the aim of understanding/improving existing benchmarks for open-source LLM cryptic crossword solving. However, since it seems that much of the problem with LLMs solving cryptics is the mismatch between the relative coarseness of LLM tokenization and the character-level tokenization required for common types of wordplay, I was curious to start the investigation with an RNN encoder-decoder, which tokenizes at the character level.

## Model Architecture and Features

- **Encoder-Decoder Architecture** with bidirectional LSTM and attention
- **Positional Embeddings** for position-aware encoding
- **Teacher Forcing Schedule** with configurable decay
- **Automatic Checkpointing** with best model selection
- **Training Visualizations** and metrics tracking
- **Model Evaluation** tools

### Configuration

Key settings in `config.py`:

```python
# Model
embedding_dim = 128
enc_hid_dim = 128
dec_hid_dim = 128
dropout = 0.1

# Training
batch_size = 16
num_epochs = 20
learning_rate = 0.0001
teacher_forcing_schedule = True  # 0.95 → 0.3 decay
```

## Dependencies

- PyTorch >= 2.0.0
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- tqdm >= 4.65.0

## Future Improvements
- multi-head attention to capture complex clue relationships
- multi-layer LSTM cor better abstraction
- curriculum learning for skill building, better convergence
- increased model capacity
