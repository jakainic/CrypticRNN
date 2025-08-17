from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from config import config
from data_preprocessing import DataProcessor
from models import Clue2Ans, ClueEncoder, AnswerDecoder
from visualization import create_training_visualization

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
        device: str = None,
        start_epoch: int = 0,
        # Teacher forcing schedule parameters
        initial_teacher_forcing: float = None,
        final_teacher_forcing: float = None,
        use_teacher_forcing_schedule: bool = True
    ):
        logger.info("Initializing Trainer...")
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
        self.start_epoch = start_epoch
        
        # Teacher forcing schedule
        self.use_teacher_forcing_schedule = use_teacher_forcing_schedule
        if self.use_teacher_forcing_schedule:
            self.initial_teacher_forcing = initial_teacher_forcing or 0.95
            self.final_teacher_forcing = final_teacher_forcing or 0.3
            logger.info(f"Teacher forcing schedule: {self.initial_teacher_forcing:.2f} -> {self.final_teacher_forcing:.2f}")
        else:
            self.initial_teacher_forcing = config['model'].teacher_forcing_ratio
            self.final_teacher_forcing = config['model'].teacher_forcing_ratio
            logger.info(f"Fixed teacher forcing ratio: {self.initial_teacher_forcing:.2f}")
        
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize learning rate scheduler
        logger.info("Setting up learning rate scheduler...")
        if config['training'].lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training'].num_epochs,
                eta_min=config['training'].min_lr
            )
            logger.info("Using CosineAnnealingLR scheduler")
        elif config['training'].lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=5,
                gamma=config['training'].lr_decay_factor
            )
            logger.info("Using StepLR scheduler")
        elif config['training'].lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config['training'].lr_decay_factor,
                patience=config['training'].lr_patience,
                min_lr=config['training'].min_lr
            )
            logger.info("Using ReduceLROnPlateau scheduler")
        else:
            self.scheduler = None
            logger.info("No learning rate scheduler used")
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_length_accuracies = []
        self.val_length_accuracies = []
        self.learning_rates = []
        self.teacher_forcing_ratios = []
        logger.info("Trainer initialization complete")

    def calculate_accuracy(self, pred, target):
        """
        Calculate accuracy metrics for predictions.
        
        Args:
            pred: Predicted sequences [batch_size, seq_len]
            target: Target sequences [batch_size, seq_len]
            
        Returns:
            dict: Dictionary containing various accuracy metrics
        """
        # Ensure both tensors have the same shape
        min_seq_len = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_seq_len]
        target = target[:, :min_seq_len]
        
        # Calculate exact match accuracy
        exact_matches = (pred == target).all(dim=1).float().mean()
        
        # Calculate token-level accuracy
        token_accuracy = (pred == target).float().mean()
        
        # Calculate length accuracy
        pred_lengths = (pred != 0).sum(dim=1)
        target_lengths = (target != 0).sum(dim=1)
        length_accuracy = (pred_lengths == target_lengths).float().mean()
        
        return {
            'exact_match': exact_matches.item(),
            'token_accuracy': token_accuracy.item(),
            'length_accuracy': length_accuracy.item()
        }

    def train_epoch(self, teacher_forcing_ratio: float = None) -> Tuple[float, float, float]:
        """Train the model for one epoch"""
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = config['model'].teacher_forcing_ratio
            
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_length_correct = 0
        total_sequences = 0
        
        num_batches = len(self.train_loader)
        logger.info(f"Starting epoch training with {num_batches} batches (TF ratio: {teacher_forcing_ratio:.3f})")
        
        for batch_idx, (clues, answers) in enumerate(self.train_loader):
            if batch_idx % 10 == 0:  # Log every 10 batches
                logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")
                
            clues, answers = clues.to(self.device), answers.to(self.device)
            
            self.optimizer.zero_grad()
            # Pass teacher forcing ratio to model
            output = self.model(clues, answers, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # Reshape output and target for loss calculation
            output_for_loss = output[:, 1:].reshape(-1, output.shape[-1])
            target_for_loss = answers[:, 1:].reshape(-1)
            loss = self.criterion(output_for_loss, target_for_loss)
            loss.backward()
            
            # Gradient clipping to prevent explosion and stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate sequence-level accuracy (use original shapes)
            batch_predictions = output[:, 1:].argmax(dim=2)
            accuracy = self.calculate_accuracy(batch_predictions, answers[:, 1:])
            total_correct += int(accuracy['exact_match'] * answers.shape[0])
            total_length_correct += int(accuracy['length_accuracy'] * answers.shape[0])
            total_sequences += answers.shape[0]
        
        avg_loss = total_loss / len(self.train_loader)
        word_accuracy = total_correct / total_sequences
        length_accuracy = total_length_correct / total_sequences
        
        logger.info(f"Epoch training complete - Loss: {avg_loss:.4f}, Word Accuracy: {word_accuracy:.4f}, Length Accuracy: {length_accuracy:.4f}")
        return avg_loss, word_accuracy, length_accuracy

    def evaluate(self) -> Tuple[float, float, float]:
        """Evaluate the model on the test set"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_length_correct = 0
        total_sequences = 0
        
        num_batches = len(self.test_loader)
        logger.info(f"Starting evaluation with {num_batches} batches")
        
        with torch.no_grad():
            for batch_idx, (clues, answers) in enumerate(self.test_loader):
                if batch_idx % 10 == 0:  # Log every 10 batches
                    logger.info(f"Processing evaluation batch {batch_idx + 1}/{num_batches}")
                    
                clues, answers = clues.to(self.device), answers.to(self.device)
                # Always use 0 teacher forcing during evaluation
                output = self.model(clues, answers, teacher_forcing_ratio=0.0)
                
                # Reshape output and target for loss calculation
                output_for_loss = output[:, 1:].reshape(-1, output.shape[-1])
                target_for_loss = answers[:, 1:].reshape(-1)
                loss = self.criterion(output_for_loss, target_for_loss)
                total_loss += loss.item()
                
                # Calculate sequence-level accuracy (use original shapes)
                batch_predictions = output[:, 1:].argmax(dim=2)
                accuracy = self.calculate_accuracy(batch_predictions, answers[:, 1:])
                total_correct += int(accuracy['exact_match'] * answers.shape[0])
                total_length_correct += int(accuracy['length_accuracy'] * answers.shape[0])
                total_sequences += answers.shape[0]
        
        avg_loss = total_loss / len(self.test_loader)
        word_accuracy = total_correct / total_sequences
        length_accuracy = total_length_correct / total_sequences
        
        logger.info(f"Evaluation complete - Loss: {avg_loss:.4f}, Word Accuracy: {word_accuracy:.4f}, Length Accuracy: {length_accuracy:.4f}")
        return avg_loss, word_accuracy, length_accuracy

    def save_checkpoint(self, path: str, epoch: int = None):
        """Save model checkpoint with all necessary training state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': epoch if epoch is not None else self.start_epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_length_accuracies': self.train_length_accuracies,
            'val_length_accuracies': self.val_length_accuracies,
            'learning_rates': self.learning_rates,
            'config': config  # Save config for consistency
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        data_processor: DataProcessor
    ) -> 'Trainer':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=config['training'].device)
        
        # Initialize model
        encoder = ClueEncoder(
            input_dim=len(data_processor.clue_vocab.token2idx),
            emb_dim=config['model'].embedding_dim,
            enc_hid_dim=config['model'].enc_hid_dim,
            dec_hid_dim=config['model'].dec_hid_dim,
            dropout=config['model'].dropout,
            max_length=200
        )
        
        decoder = AnswerDecoder(
            output_dim=len(data_processor.answer_vocab.token2idx),
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
        
        # Initialize trainer with the loaded epoch
        trainer = cls(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            data_processor=data_processor,
            start_epoch=checkpoint.get('current_epoch', 0)
        )
        
        # Load state
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint.get('scheduler_state_dict'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load metrics
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.train_accuracies = checkpoint.get('train_accuracies', [])
        trainer.val_accuracies = checkpoint.get('val_accuracies', [])
        trainer.train_length_accuracies = checkpoint.get('train_length_accuracies', [])
        trainer.val_length_accuracies = checkpoint.get('val_length_accuracies', [])
        trainer.learning_rates = checkpoint.get('learning_rates', [])
        
        logger.info(f"Loaded checkpoint from epoch {trainer.start_epoch}")
        return trainer

    @classmethod
    def resume_training(
        cls,
        checkpoint_path: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        data_processor: DataProcessor,
        additional_epochs: int = None
    ) -> 'Trainer':
        """Resume training from a checkpoint"""
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Load the trainer from checkpoint
        trainer = cls.load_checkpoint(checkpoint_path, train_loader, test_loader, data_processor)
        
        # Continue training
        epochs_to_train = additional_epochs or (config['training'].num_epochs - trainer.start_epoch)
        if epochs_to_train > 0:
            logger.info(f"Continuing training for {epochs_to_train} more epochs...")
            trainer.train(num_epochs=epochs_to_train)
        else:
            logger.info("Training already complete!")
        
        return trainer

    def calculate_teacher_forcing_ratio(self, epoch: int, total_epochs: int) -> float:
        """Calculate the teacher forcing ratio for the current epoch"""
        if not self.use_teacher_forcing_schedule:
            return self.initial_teacher_forcing
        
        # Linear decay from initial to final teacher forcing ratio
        # Progress should be based on the current epoch relative to total epochs from the beginning
        progress = epoch / (total_epochs - 1) if total_epochs > 1 else 0
        return self.initial_teacher_forcing - (self.initial_teacher_forcing - self.final_teacher_forcing) * progress

    def train(self, num_epochs: int = None) -> Dict[str, Any]:
        """Train the model for multiple epochs"""
        if num_epochs is None:
            num_epochs = config['training'].num_epochs

        # Calculate the final epoch we'll reach
        final_epoch = self.start_epoch + num_epochs
        
        logger.info(f"Starting training from epoch {self.start_epoch + 1} for {num_epochs} epochs...")
        logger.info(f"Will train until epoch {final_epoch}")
        logger.info(f"Training configuration:")
        logger.info(f"- Learning rate: {config['training'].learning_rate}")
        logger.info(f"- Batch size: {config['training'].batch_size}")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Scheduler: {config['training'].lr_scheduler}")
        
        for epoch in range(self.start_epoch, final_epoch):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{final_epoch}")
            logger.info(f"{'='*50}")
            
            # Calculate teacher forcing ratio for this epoch (using final epoch for proper scheduling)
            current_tf = self.calculate_teacher_forcing_ratio(epoch, final_epoch)
            
            train_loss, train_acc, train_length_acc = self.train_epoch(teacher_forcing_ratio=current_tf)
            val_loss, val_acc, val_length_acc = self.evaluate()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_length_accuracies.append(train_length_acc)
            self.val_length_accuracies.append(val_length_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            self.teacher_forcing_ratios.append(current_tf)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = f"{config['training'].checkpoint_dir}/best_model.pt"
                self.save_checkpoint(best_model_path, epoch)
                logger.info(f"New best model saved! (val_loss: {val_loss:.3f})")
            
            # Save regular checkpoint
            if (epoch + 1) % config['training'].save_every_n_epochs == 0:
                epoch_model_path = f"{config['training'].checkpoint_dir}/model_epoch_{epoch+1}.pt"
                self.save_checkpoint(epoch_model_path, epoch)
            
            # Update start epoch
            self.start_epoch = epoch + 1

        # Create final visualizations
        logger.info("Creating final training visualizations...")
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,  # Word accuracy
            'val_accuracies': self.val_accuracies,  # Word accuracy
            'train_length_accuracies': self.train_length_accuracies,
            'val_length_accuracies': self.val_length_accuracies,
            'learning_rates': self.learning_rates,
            'teacher_forcing_ratios': self.teacher_forcing_ratios
        }
        
        create_training_visualization(
            metrics,
            self.model,
            self.data_processor,
            self.test_loader,
            output_dir='visualizations/final'
        )
        
        return metrics
