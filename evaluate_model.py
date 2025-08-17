#!/usr/bin/env python3
"""
Model Evaluation Script
======================

Evaluate trained cryptic crossword models with comprehensive metrics.

Usage:
    python evaluate_model.py --checkpoint checkpoints/best_model.pt
    python evaluate_model.py --checkpoint checkpoints/model_epoch_15.pt --samples 100
"""

import argparse
import pandas as pd
import torch
from typing import List, Dict
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

from train import Trainer
from data_preprocessing import DataProcessor
from inference import generate_answer, batch_generate_answers
from config import config

def load_model_for_evaluation(checkpoint_path: str):
    """Load model and data processor for evaluation"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create data processor
    data_processor = DataProcessor()
    train_loader, test_loader = data_processor.prepare_data()
    
    # Load model from checkpoint
    trainer = Trainer.load_checkpoint(
        checkpoint_path, train_loader, test_loader, data_processor
    )
    
    model = trainer.model
    model.eval()
    
    return model, data_processor, test_loader

def evaluate_model_comprehensive(
    model, 
    data_processor: DataProcessor,
    test_loader,
    num_samples: int = None
) -> Dict:
    """Comprehensive evaluation with multiple metrics"""
    
    print("Starting evaluation...")
    
    # Get test data
    test_clues = []
    test_answers = []
    
    total_samples = 0
    for batch_clues, batch_answers in test_loader:
        for i in range(len(batch_clues)):
            if num_samples and total_samples >= num_samples:
                break
            
            # Decode clue and answer
            clue_tokens = [data_processor.clue_vocab.idx2token[idx.item()] 
                          for idx in batch_clues[i] if idx.item() != 0]
            answer_tokens = [data_processor.answer_vocab.idx2token[idx.item()] 
                           for idx in batch_answers[i] if idx.item() != 0]
            
            clue = ' '.join(clue_tokens).replace('<pad>', '').replace('<sos>', '').replace('<eos>', '').strip()
            answer = ' '.join(answer_tokens).replace('<pad>', '').replace('<sos>', '').replace('<eos>', '').strip()
            
            if clue and answer:
                test_clues.append(clue)
                test_answers.append(answer)
                total_samples += 1
        
        if num_samples and total_samples >= num_samples:
            break
    
    print(f"Evaluating on {len(test_clues)} samples...")
    
    # Generate answers
    generated_answers = batch_generate_answers(model, test_clues, data_processor)
    
    # Calculate metrics
    metrics = calculate_detailed_metrics(generated_answers, test_answers, test_clues)
    
    # Add model info
    metrics['model_info'] = {
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(model.device),
        'test_samples': len(test_clues)
    }
    
    return metrics

def calculate_detailed_metrics(generated: List[str], actual: List[str], clues: List[str]) -> Dict:
    """Calculate detailed evaluation metrics for crossword answers (character-based)"""
    
    metrics = {}
    
    # Basic accuracy
    exact_matches = sum(1 for g, a in zip(generated, actual) if g.lower().strip() == a.lower().strip())
    total = len(generated)
    metrics['exact_match_accuracy'] = exact_matches / total
    
    # Character-level length metrics (correct for crosswords)
    char_length_diffs = [abs(len(g) - len(a)) for g, a in zip(generated, actual)]
    metrics['avg_char_length_difference'] = sum(char_length_diffs) / len(char_length_diffs)
    metrics['correct_char_length_ratio'] = sum(1 for d in char_length_diffs if d == 0) / len(char_length_diffs)
    
    # First character accuracy (analogous to first word)
    first_char_matches = sum(1 for g, a in zip(generated, actual) 
                           if g and a and g[0].lower() == a[0].lower())
    metrics['first_char_accuracy'] = first_char_matches / total
    
    # Performance by clue length (words in clue)
    clue_length_performance = defaultdict(list)
    for clue, gen, act in zip(clues, generated, actual):
        clue_len = len(clue.split())  # Clues can have multiple words
        clue_length_performance[clue_len].append(gen.lower().strip() == act.lower().strip())
    
    metrics['performance_by_clue_length'] = {
        length: sum(correct) / len(correct) 
        for length, correct in clue_length_performance.items()
    }
    
    # Performance by answer length (characters in answer)
    answer_length_performance = defaultdict(list)
    for gen, act in zip(generated, actual):
        ans_len = len(act)  # Character length for crossword answers
        answer_length_performance[ans_len].append(gen.lower().strip() == act.lower().strip())
    
    metrics['performance_by_answer_length'] = {
        length: sum(correct) / len(correct) 
        for length, correct in answer_length_performance.items()
    }
    
    # Store examples
    metrics['examples'] = [
        {
            'clue': clue,
            'actual': actual_ans,
            'generated': gen_ans,
            'correct': gen_ans.lower().strip() == actual_ans.lower().strip(),
            'clue_length_words': len(clue.split()),
            'actual_length_chars': len(actual_ans),
            'generated_length_chars': len(gen_ans),
            'char_length_diff': abs(len(gen_ans) - len(actual_ans))
        }
        for clue, actual_ans, gen_ans in zip(clues, actual, generated)
    ]
    
    return metrics

def print_evaluation_report(metrics: Dict, model_name: str = "Model"):
    """Print comprehensive evaluation report for crossword models"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*60}")
    
    # Basic metrics
    print(f"\nBASIC METRICS:")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"First Character Accuracy: {metrics['first_char_accuracy']:.4f} ({metrics['first_char_accuracy']*100:.2f}%)")
    print(f"Correct Length Ratio: {metrics['correct_char_length_ratio']:.4f} ({metrics['correct_char_length_ratio']*100:.2f}%)")
    print(f"Average Character Length Difference: {metrics['avg_char_length_difference']:.2f} chars")
    
    # Performance by clue length (words in clue)
    print(f"\nPERFORMANCE BY CLUE LENGTH (words in clue):")
    clue_perf = metrics['performance_by_clue_length']
    for length in sorted(clue_perf.keys()):
        print(f"{length} words: {clue_perf[length]:.3f} ({clue_perf[length]*100:.1f}%)")
    
    # Performance by answer length (characters in answer)
    print(f"\nPERFORMANCE BY ANSWER LENGTH (characters):")
    ans_perf = metrics['performance_by_answer_length']
    sorted_lengths = sorted(ans_perf.keys())
    for length in sorted_lengths[:10]:  # Show first 10 lengths
        count = sum(1 for ex in metrics['examples'] if ex['actual_length_chars'] == length)
        print(f"{length} chars: {ans_perf[length]:.3f} ({ans_perf[length]*100:.1f}%) [{count} samples]")
    if len(sorted_lengths) > 10:
        print(f"... and {len(sorted_lengths) - 10} other lengths")
    
    # Model info
    model_info = metrics['model_info']
    print(f"\nMODEL INFO:")
    print(f"Parameters: {model_info['num_parameters']:,}")
    print(f"Device: {model_info['device']}")
    print(f"Test Samples: {model_info['test_samples']}")
    
    # Show some examples
    print(f"\nEXAMPLE PREDICTIONS:")
    correct_examples = [ex for ex in metrics['examples'] if ex['correct']][:3]
    incorrect_examples = [ex for ex in metrics['examples'] if not ex['correct']][:3]
    
    print("CORRECT PREDICTIONS:")
    for i, ex in enumerate(correct_examples, 1):
        print(f"{i}. Clue: {ex['clue']}")
        print(f"Answer: {ex['actual']} ✓ ({ex['actual_length_chars']} chars)")
        print()
    
    print("INCORRECT PREDICTIONS:")
    for i, ex in enumerate(incorrect_examples, 1):
        print(f"{i}. Clue: {ex['clue']}")
        print(f"Expected: {ex['actual']} ({ex['actual_length_chars']} chars)")
        print(f"Generated: {ex['generated']} ({ex['generated_length_chars']} chars)")
        print(f"Char diff: {ex['char_length_diff']}")
        print()
    
    print(f"\nCrossword-Specific Notes:")
    print(f"All metrics are character-based (appropriate for crosswords)")
    print(f"Answers should be single words with no spaces")
    print(f"Character length matters more than word count")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained cryptic crossword models")
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, data_processor, test_loader = load_model_for_evaluation(args.checkpoint)
        print(f"Model loaded successfully")
        
        # Evaluate
        metrics = evaluate_model_comprehensive(
            model, data_processor, test_loader, 
            num_samples=args.samples
        )
        
        # Print report
        model_name = Path(args.checkpoint).stem
        print_evaluation_report(metrics, model_name)
        
        # Save results if requested
        if args.save_results:
            output_path = f"evaluation_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                # Remove examples from saved results to keep file size manageable
                save_metrics = {k: v for k, v in metrics.items() if k != 'examples'}
                json.dump(save_metrics, f, indent=2)
            print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your checkpoint path is correct and the model is compatible.")

if __name__ == "__main__":
    main() 