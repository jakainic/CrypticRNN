# Cryptic RNN

A sequence-to-sequence neural network for solving cryptic crossword clues. This will be part of a larger project with the aim of improving existing benchmarks for open-source LLM cryptic crossword solving. However, since it seems that much of the problem with LLMs solving cryptics is the mismatch between the relative coarseness of LLM tokenization and the character-level tokenization required for common types of wordplay, I was curious to start the investigation with an RNN encoder-decoder, which tokenizes at the character level.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py           # Configuration parameters
├── data_preprocessing.py # Data loading and preprocessing
├── models.py          # Neural network model definitions
├── train.py          # Training and evaluation logic
├── main.py           # Main script to run training
├── clues_big.csv     # Dataset
└── checkpoints/      # Saved model checkpoints
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
- Place your cryptic crossword clues dataset in `clues_big.csv`. I have sourced clues from `cryptics.georgeho.org`.
- The CSV should have columns: 'clue' and 'answer'

## Usage

### Training

To train the model:

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Create the model
3. Train for the specified number of epochs
4. Save checkpoints to the `checkpoints` directory

Training progress will be logged to both console and `training.log`.

### Generating Answers

To generate an answer for a cryptic clue:

```python
from main import generate_answer

clue = "your cryptic clue here"
answer = generate_answer(clue)
print(f"Generated answer: {answer}")
```

## Model Architecture

The model uses a sequence-to-sequence architecture with:
- Character-level encoding
- GRU-based encoder and decoder
- Teacher forcing during training
- Attention mechanism (TODO)

### Configuration

Model and training parameters can be modified in `config.py`:
- Embedding dimension
- Hidden dimension
- Dropout rate
- Batch size
- Learning rate
- Number of epochs
- etc.

## Logging

The project uses Python's logging module with two handlers:
1. StreamHandler for console output
2. FileHandler for logging to `training.log`

## Checkpoints

Model checkpoints are saved:
- After every 5 epochs
- When a new best validation loss is achieved

Checkpoints include:
- Model state
- Optimizer state
- Training history
- Configuration
- Timestamp

## Future Improvements

1. Add attention mechanism
2. Implement beam search for better answer generation
3. Add data augmentation
4. Add model evaluation metrics
5. Add early stopping
6. Add learning rate scheduling
7. Add cross-validation
8. Add model interpretability features 
