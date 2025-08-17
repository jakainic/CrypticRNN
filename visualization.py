from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import logging

from models import Clue2Ans
from data_preprocessing import DataProcessor
from config import config

logger = logging.getLogger(__name__)

def create_training_visualization(
    metrics: Dict[str, List[float]],
    model: nn.Module,
    data_processor,
    test_loader,
    output_dir: str = 'visualizations'
) -> None:
    """Create visualizations of training metrics and model predictions"""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Plot training metrics  
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Loss plot
    axes[0, 0].plot(metrics['train_losses'], label='Training Loss')
    axes[0, 0].plot(metrics['val_losses'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy plot
    axes[0, 1].plot(metrics['train_accuracies'], label='Training Accuracy')
    axes[0, 1].plot(metrics['val_accuracies'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Length accuracy plot
    axes[0, 2].plot(metrics['train_length_accuracies'], label='Training Length Accuracy')
    axes[0, 2].plot(metrics['val_length_accuracies'], label='Validation Length Accuracy')
    axes[0, 2].set_title('Training and Validation Length Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Length Accuracy')
    axes[0, 2].legend()
    
    # Learning rate plot
    axes[1, 0].plot(metrics['learning_rates'])
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    
    # Teacher forcing ratio plot (if available)
    if 'teacher_forcing_ratios' in metrics and metrics['teacher_forcing_ratios']:
        axes[1, 1].plot(metrics['teacher_forcing_ratios'], 'r-', linewidth=2)
        axes[1, 1].set_title('Teacher Forcing Ratio Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Teacher Forcing Ratio')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Teacher Forcing Data', ha='center', va='center')
        axes[1, 1].set_title('Teacher Forcing Ratio')
    
    # Remove the empty subplot
    axes[1, 2].axis('off')
    
    # Save plots
    plt.tight_layout()
    plt.savefig(output_path / 'training_metrics.png')
    plt.close()
    
    # Generate and visualize some example predictions
    model.eval()
    with torch.no_grad():
        # Get a few examples from the test set
        test_examples = []
        for batch in test_loader:
            clues, answers = batch
            test_examples.extend(list(zip(clues, answers)))
            if len(test_examples) >= 5:  # Get 5 examples
                break
        
        # Create prediction visualization
        fig, axes = plt.subplots(len(test_examples), 1, figsize=(15, 4*len(test_examples)))
        if len(test_examples) == 1:
            axes = [axes]
        
        for i, (clue, answer) in enumerate(test_examples):
            # Move to device
            clue = clue.to(model.device)
            answer = answer.to(model.device)
            
            # Get model prediction
            output = model(clue.unsqueeze(0))  # Add batch dimension
            pred = output.argmax(dim=-1).squeeze(0)
            
            # Decode predictions
            pred_text = data_processor.decode_answer(pred)
            answer_text = data_processor.decode_answer(answer)
            clue_text = data_processor.decode_clue(clue)
            
            # Plot
            axes[i].text(0.1, 0.7, f'Clue: {clue_text}', fontsize=12)
            axes[i].text(0.1, 0.5, f'True Answer: {answer_text}', fontsize=12)
            axes[i].text(0.1, 0.3, f'Predicted: {pred_text}', fontsize=12)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'example_predictions.png')
        plt.close()
    
    logger.info(f"Visualizations saved to {output_path}") 