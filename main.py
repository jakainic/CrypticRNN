import logging
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split

from config import config
from data_preprocessing import DataProcessor
from models import Clue2Ans, ClueEncoder, AnswerDecoder
from train import Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize data processor with robust CSV reading
    logger.info("Initializing data processor...")
    try:
        # Read CSV with more robust parameters
        data = pd.read_csv(
            config['data'].data_file,
            quoting=1,  # QUOTE_ALL
            escapechar='\\',
            on_bad_lines='warn'  # Warn about problematic lines
        )
        
        # Keep only the columns we need
        if 'clue' in data.columns and 'answer' in data.columns:
            data = data[['clue', 'answer']]
        else:
            raise ValueError("CSV must contain 'clue' and 'answer' columns")
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Save cleaned data
        data.to_csv('cleaned_data.csv', index=False)
        config['data'].data_file = 'cleaned_data.csv'
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise
    
    data_processor = DataProcessor()
    
    # Create datasets
    logger.info("Creating data loaders...")
    train_loader, test_loader = data_processor.prepare_data()
    
    # Initialize model
    logger.info("Initializing model...")
    encoder = ClueEncoder(
        input_dim=len(data_processor.clue_vocab),
        emb_dim=config['model'].embedding_dim,
        enc_hid_dim=config['model'].enc_hid_dim,
        dec_hid_dim=config['model'].dec_hid_dim,
        dropout=config['model'].dropout,
        max_length=200
    )
    
    decoder = AnswerDecoder(
        output_dim=len(data_processor.answer_vocab),
        emb_dim=config['model'].embedding_dim,
        enc_hid_dim=config['model'].enc_hid_dim,
        dec_hid_dim=config['model'].dec_hid_dim,
        dropout=config['model'].dropout
    )
    
    model = Clue2Ans(
        encoder=encoder,
        decoder=decoder,
        device=config['training'].device
    )
    
    # Model uses PyTorch default initialization
    
    # Initialize trainer with teacher forcing schedule
    logger.info("Initializing trainer with teacher forcing schedule...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        data_processor=data_processor,
        initial_teacher_forcing=config['model'].initial_teacher_forcing,
        final_teacher_forcing=config['model'].final_teacher_forcing,
        use_teacher_forcing_schedule=config['model'].use_teacher_forcing_schedule
    )
    
    # Create output directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('visualizations').mkdir(exist_ok=True)
    
    # Train model
    logger.info("Starting training...")
    metrics = trainer.train()
    logger.info("Training completed!")
    
    # Save final model
    trainer.save_checkpoint('checkpoints/final_model.pt')
    logger.info("Model saved to checkpoints/final_model.pt")

def generate_answer(clue: str, model_path: str = 'checkpoints/best_model.pt'):
    """Generate an answer for a given clue using a trained model"""
    # Load data processor to get vocabularies
    data_processor = DataProcessor()
    train_loader, test_loader = data_processor.prepare_data()
    
    # Load model from checkpoint
    trainer = Trainer.load_checkpoint(
        model_path,
        train_loader,
        test_loader,
        data_processor
    )
    model = trainer.model
    
    # Use the improved inference function
    from inference import generate_answer as inference_generate_answer
    answer = inference_generate_answer(model, clue, data_processor)
    
    # Print prediction
    print("\nPrediction:")
    print(f"Clue: {clue}")
    print(f"Generated Answer: {answer}")
    print("-" * 50)
    
    return answer

if __name__ == '__main__':
    main() 