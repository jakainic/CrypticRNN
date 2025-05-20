import logging

from config import config
from data_preprocessing import DataProcessor
from models import ClueEncoder, AnswerDecoder, Clue2Ans
from train import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_model(input_dim: int, output_dim: int) -> Clue2Ans:
    """Create and initialize the model"""
    encoder = ClueEncoder(
        input_dim=input_dim,
        emb_dim=config['model'].embedding_dim,
        hid_dim=config['model'].hidden_dim,
        dropout=config['model'].dropout
    )
    
    decoder = AnswerDecoder(
        output_dim=output_dim,
        emb_dim=config['model'].embedding_dim,
        hid_dim=config['model'].hidden_dim,
        dropout=config['model'].dropout
    )
    
    return Clue2Ans(encoder, decoder, pad_idx=1)  # Assuming pad_idx=1

def main():
    """Main training script"""
    logger.info("=" * 50)
    logger.info("Starting Cryptic Crossword Solver Training")
    logger.info("=" * 50)
    logger.info(f"Device: {config['training'].device}")
    logger.info(f"Batch size: {config['training'].batch_size}")
    logger.info(f"Number of epochs: {config['training'].num_epochs}")
    logger.info(f"Learning rate: {config['training'].learning_rate}")
    logger.info("-" * 50)
    
    logger.info("Initializing data processor...")
    data_processor = DataProcessor()
    
    # Prepare data and get data loaders
    train_loader, test_loader = data_processor.prepare_data()
    
    # Get vocabulary sizes for encoder and decoder
    input_dim, output_dim = data_processor.vocab_sizes
    logger.info(f"Vocabulary sizes - Input: {input_dim}, Output: {output_dim}")
    
    # Create model
    logger.info("Creating model...")
    model = setup_model(input_dim, output_dim)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        data_processor=data_processor
    )
    
    # Train model
    logger.info("Starting training...")
    logger.info("-" * 50)
    results = trainer.train()
    
    # Log final results
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Final train accuracy: {results['train_accuracies'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {results['val_accuracies'][-1]:.4f}")
    logger.info("=" * 50)
    logger.info(f"Model checkpoints saved in: checkpoints/")
    logger.info(f"Training log saved in: training.log")

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
    model.eval()
    
    # Preprocess and encode the clue
    preprocessor = data_processor.preprocessor
    clue_tokens = preprocessor.tokenize_chars(clue)
    clue_tensor = data_processor.encode_sequence(clue_tokens, data_processor.clue_vocab)
    clue_tensor = clue_tensor.to(trainer.device)
    
    # Generate answer
    generated = model.generate(clue_tensor)
    
    # Decode the answer
    answer = data_processor.decode_answer(generated)
    
    return answer

if __name__ == '__main__':
    main() 