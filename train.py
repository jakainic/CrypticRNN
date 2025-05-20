from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from config import config
from models import Clue2Ans, ClueEncoder, AnswerDecoder
from data_preprocessing import DataProcessor

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: Clue2Ans,
        train_loader: DataLoader,
        test_loader: DataLoader,
        data_processor: DataProcessor,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        device: str = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_processor = data_processor
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=model.pad_idx)
        self.optimizer = optimizer or optim.Adam(
            model.parameters(),
            lr=config['training'].learning_rate
        )
        self.device = device or config['training'].device
        self.model.to(self.device)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self) -> Tuple[float, float]:
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        for batch_idx, (clue, answer) in enumerate(self.train_loader):
            clue, answer = clue.to(self.device), answer.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(clue, answer)
            
            # Get predictions for accuracy calculation
            predictions = output.argmax(dim=-1)  # [batch_size, seq_len]
            all_predictions.extend(predictions.cpu())
            all_targets.extend(answer.cpu())
            
            # Reshape output and target for loss calculation
            output = output[:, 1:].reshape(-1, output.shape[-1])
            answer = answer[:, 1:].reshape(-1)
            
            loss = self.criterion(output, answer)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                # Calculate accuracy for this batch
                accuracy, examples = self.data_processor.calculate_accuracy(
                    predictions,
                    answer.view(predictions.shape),
                    return_examples=True,
                    num_examples=2
                )
                
                logger.info(f'Train Batch: {batch_idx}/{len(self.train_loader)} '
                          f'Loss: {loss.item():.4f} '
                          f'Accuracy: {accuracy:.4f}')
                
                # Log some example predictions
                for pred, target in examples:
                    logger.info(f'Prediction: {pred} | Target: {target}')

        avg_loss = total_loss / len(self.train_loader)
        
        # Calculate overall accuracy for the epoch
        accuracy, _ = self.data_processor.calculate_accuracy(
            torch.stack(all_predictions),
            torch.stack(all_targets)
        )
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate the model on the test set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for clue, answer in self.test_loader:
                clue, answer = clue.to(self.device), answer.to(self.device)
                
                output = self.model(clue, answer, teacher_forcing_ratio=0.0)
                
                # Get predictions for accuracy calculation
                predictions = output.argmax(dim=-1)  # [batch_size, seq_len]
                all_predictions.extend(predictions.cpu())
                all_targets.extend(answer.cpu())
                
                output = output[:, 1:].reshape(-1, output.shape[-1])
                answer = answer[:, 1:].reshape(-1)
                
                loss = self.criterion(output, answer)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        
        # Calculate accuracy and get some examples
        accuracy, examples = self.data_processor.calculate_accuracy(
            torch.stack(all_predictions),
            torch.stack(all_targets),
            return_examples=True,
            num_examples=5
        )
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        # Log some example predictions
        logger.info("\nValidation Examples:")
        for pred, target in examples:
            logger.info(f'Prediction: {pred} | Target: {target}')
        
        return avg_loss, accuracy

    def train(self, num_epochs: int = None) -> Dict[str, Any]:
        """Train the model for multiple epochs"""
        if num_epochs is None:
            num_epochs = config['training'].num_epochs

        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            logger.info(f"Epoch: {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss
        }

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint"""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        logger.info(f"Saved checkpoint to {checkpoint_dir / filename}")

    @classmethod
    def load_checkpoint(
        cls,
        filename: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        data_processor: DataProcessor
    ) -> 'Trainer':
        """Load model from checkpoint"""
        checkpoint_path = Path('checkpoints') / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        saved_config = checkpoint['config']
        
        # Recreate model architecture
        encoder = ClueEncoder(
            input_dim=saved_config['model'].input_dim,
            emb_dim=saved_config['model'].embedding_dim,
            hid_dim=saved_config['model'].hidden_dim,
            dropout=saved_config['model'].dropout
        )
        
        decoder = AnswerDecoder(
            output_dim=saved_config['model'].output_dim,
            emb_dim=saved_config['model'].embedding_dim,
            hid_dim=saved_config['model'].hidden_dim,
            dropout=saved_config['model'].dropout
        )
        
        model = Clue2Ans(encoder, decoder, pad_idx=1)  # Assuming pad_idx=1
        model.load_state_dict(checkpoint['model_state_dict'])
        
        trainer = cls(model, train_loader, test_loader, data_processor)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.train_accuracies = checkpoint.get('train_accuracies', [])
        trainer.val_accuracies = checkpoint.get('val_accuracies', [])
        trainer.best_val_loss = checkpoint['best_val_loss']
        
        return trainer 