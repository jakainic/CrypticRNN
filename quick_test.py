import logging
import tempfile
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def analyze_attention_weights(model, data_processor, test_loader, num_examples=5):
    """Analyze attention weight patterns for diagnostic insights"""
    print("\n" + "="*60)
    print("ATTENTION WEIGHT ANALYSIS")
    print("="*60)
    
    model.eval()
    attention_stats = {
        'entropies': [],
        'variances': [],
        'max_weights': [],
        'min_weights': [],
        'focus_positions': [],
        'uniformity_scores': []
    }
    
    with torch.no_grad():
        for batch_idx, (clues, answers) in enumerate(test_loader):
            if batch_idx >= num_examples:
                break
                
            clues = clues.to(model.device)
            batch_size, clue_len = clues.shape
            
            # Get encoder outputs
            encoder_outputs, encoder_hidden = model.encoder(clues)
            hidden = model.decoder.project_encoder_hidden(encoder_hidden)
            
            # Analyze attention for each sequence in batch
            for seq_idx in range(min(batch_size, 3)):  # Analyze first 3 sequences
                print(f"\nSequence {batch_idx*batch_size + seq_idx + 1}:")
                
                # Get non-padding positions
                non_pad_mask = clues[seq_idx] != 0
                valid_length = non_pad_mask.sum().item()
                clue_text = data_processor.decode_clue(clues[seq_idx])
                
                print(f"Clue: '{clue_text}' (length: {valid_length})")
                
                # Calculate attention weights
                attention_weights = model.decoder.attention(
                    hidden[0][-1][seq_idx:seq_idx+1], 
                    encoder_outputs[seq_idx:seq_idx+1]
                )
                
                # Focus on valid positions only
                valid_attention = attention_weights[0][:valid_length]
                
                # Statistical analysis
                entropy = -torch.sum(valid_attention * torch.log(valid_attention + 1e-8)).item()
                variance = torch.var(valid_attention).item()
                max_weight = torch.max(valid_attention).item()
                min_weight = torch.min(valid_attention).item()
                focus_pos = torch.argmax(valid_attention).item()
                
                # Uniformity score (how close to uniform distribution)
                uniform_prob = 1.0 / valid_length
                uniformity = 1.0 - torch.sum(torch.abs(valid_attention - uniform_prob)).item() / 2.0
                
                print(f"Attention Statistics:")
                print(f"Entropy: {entropy:.4f} (max: {np.log(valid_length):.4f})")
                print(f"Variance: {variance:.6f}")
                print(f"Range: [{min_weight:.4f}, {max_weight:.4f}]")
                print(f"Focus Position: {focus_pos} (of {valid_length-1})")
                print(f"Uniformity Score: {uniformity:.4f} (1.0 = uniform)")
                
                # Show attention distribution
                print(f"Attention Weights: {[f'{w:.3f}' for w in valid_attention.tolist()]}")
                
                # Store stats
                attention_stats['entropies'].append(entropy)
                attention_stats['variances'].append(variance)
                attention_stats['max_weights'].append(max_weight)
                attention_stats['min_weights'].append(min_weight)
                attention_stats['focus_positions'].append(focus_pos / max(1, valid_length-1))  # Normalized
                attention_stats['uniformity_scores'].append(uniformity)
    
    # Overall statistics
    print(f"\nOVERALL ATTENTION ANALYSIS:")
    print(f"Average Entropy: {np.mean(attention_stats['entropies']):.4f} ± {np.std(attention_stats['entropies']):.4f}")
    print(f"Average Variance: {np.mean(attention_stats['variances']):.6f} ± {np.std(attention_stats['variances']):.6f}")
    print(f"Weight Range: [{np.mean(attention_stats['min_weights']):.4f}, {np.mean(attention_stats['max_weights']):.4f}]")
    print(f"Average Uniformity: {np.mean(attention_stats['uniformity_scores']):.4f} ± {np.std(attention_stats['uniformity_scores']):.4f}")
    
    # Diagnostic assessment
    avg_variance = np.mean(attention_stats['variances'])
    avg_uniformity = np.mean(attention_stats['uniformity_scores'])
    
    if avg_variance > 0.005:
        print(f"GOOD: High attention variance ({avg_variance:.6f}) indicates focused attention")
    elif avg_variance > 0.002:
        print(f"MODERATE: Attention variance ({avg_variance:.6f}) could be improved")
    else:
        print(f"POOR: Low attention variance ({avg_variance:.6f}) indicates uniform attention")
    
    if avg_uniformity < 0.8:
        print(f"GOOD: Low uniformity ({avg_uniformity:.4f}) indicates focused attention")
    else:
        print(f"WARNING: High uniformity ({avg_uniformity:.4f}) indicates scattered attention")
    
    return attention_stats

def analyze_encoder_diversity(model, data_processor, test_loader, num_examples=3):
    """Analyze encoder output diversity to diagnose position similarity issues"""
    print("\n" + "="*60)
    print("ENCODER OUTPUT DIVERSITY ANALYSIS")
    print("="*60)
    
    model.eval()
    diversity_stats = {
        'position_similarities': [],
        'adjacent_similarities': [],
        'dimension_variances': [],
        'output_ranges': [],
        'dead_dimensions': []
    }
    
    with torch.no_grad():
        for batch_idx, (clues, answers) in enumerate(test_loader):
            if batch_idx >= num_examples:
                break
                
            clues = clues.to(model.device)
            encoder_outputs, _ = model.encoder(clues)
            batch_size, seq_len, hidden_dim = encoder_outputs.shape
            
            for seq_idx in range(min(batch_size, 2)):  # Analyze first 2 sequences
                print(f"\n🔬 Sequence {batch_idx*batch_size + seq_idx + 1}:")
                
                # Get valid positions (non-padding)
                non_pad_mask = clues[seq_idx] != 0
                valid_outputs = encoder_outputs[seq_idx][non_pad_mask].cpu().numpy()
                valid_length = len(valid_outputs)
                clue_text = data_processor.decode_clue(clues[seq_idx])
                
                print(f"Clue: '{clue_text}' (length: {valid_length})")
                
                if valid_length < 2:
                    continue
                
                # Position similarity analysis
                similarities = cosine_similarity(valid_outputs)
                upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
                
                # Adjacent position similarity
                adjacent_sims = []
                for i in range(valid_length - 1):
                    sim = cosine_similarity([valid_outputs[i]], [valid_outputs[i+1]])[0][0]
                    adjacent_sims.append(sim)
                
                # Dimension variance analysis
                dim_variances = np.var(valid_outputs, axis=0)
                dead_dims = (dim_variances < 1e-6).sum()
                
                # Output range analysis
                output_std = np.std(valid_outputs)
                output_range = np.max(valid_outputs) - np.min(valid_outputs)
                
                print(f"Position Similarity:")
                print(f"Overall Mean: {upper_triangle.mean():.4f} ± {upper_triangle.std():.4f}")
                print(f"Adjacent Mean: {np.mean(adjacent_sims):.4f} ± {np.std(adjacent_sims):.4f}")
                print(f"Range: [{upper_triangle.min():.4f}, {upper_triangle.max():.4f}]")
                
                print(f"Output Statistics:")
                print(f"Standard Deviation: {output_std:.4f}")
                print(f"Value Range: {output_range:.4f}")
                print(f"Dead Dimensions: {dead_dims}/{hidden_dim} ({dead_dims/hidden_dim*100:.1f}%)")
                print(f"Avg Dim Variance: {dim_variances.mean():.6f}")
                
                # Store stats
                diversity_stats['position_similarities'].extend(upper_triangle)
                diversity_stats['adjacent_similarities'].extend(adjacent_sims)
                diversity_stats['dimension_variances'].append(dim_variances.mean())
                diversity_stats['output_ranges'].append(output_range)
                diversity_stats['dead_dimensions'].append(dead_dims)
    
    # Overall assessment
    print(f"\nOVERALL ENCODER DIVERSITY:")
    avg_pos_sim = np.mean(diversity_stats['position_similarities'])
    avg_adj_sim = np.mean(diversity_stats['adjacent_similarities'])
    avg_dead_dims = np.mean(diversity_stats['dead_dimensions'])
    
    print(f"Position Similarity: {avg_pos_sim:.4f} ± {np.std(diversity_stats['position_similarities']):.4f}")
    print(f"Adjacent Similarity: {avg_adj_sim:.4f} ± {np.std(diversity_stats['adjacent_similarities']):.4f}")
    print(f"Average Dead Dims: {avg_dead_dims:.1f}")
    print(f"Dimension Variance: {np.mean(diversity_stats['dimension_variances']):.6f}")
    
    # Diagnostic assessment
    if avg_adj_sim < 0.5:
        print(f"EXCELLENT: Low adjacent similarity ({avg_adj_sim:.4f}) - positions well distinguished")
    elif avg_adj_sim < 0.7:
        print(f"GOOD: Moderate adjacent similarity ({avg_adj_sim:.4f}) - reasonable position distinction")
    elif avg_adj_sim < 0.85:
        print(f"WARNING: High adjacent similarity ({avg_adj_sim:.4f}) - positions somewhat similar")
    else:
        print(f"PROBLEM: Very high adjacent similarity ({avg_adj_sim:.4f}) - attention may struggle")
    
    if avg_dead_dims > 25:
        print(f"PROBLEM: Too many dead dimensions ({avg_dead_dims:.1f}) - check initialization")
    elif avg_dead_dims > 10:
        print(f"WARNING: Some dead dimensions ({avg_dead_dims:.1f}) - could be improved")
    else:
        print(f"GOOD: Few dead dimensions ({avg_dead_dims:.1f})")
    
    return diversity_stats

def analyze_prediction_patterns(predictions, true_answers, data_processor):
    """Analyze prediction patterns for repetitive output detection"""
    print("\n" + "="*60)
    print("PREDICTION PATTERN ANALYSIS")
    print("="*60)
    
    # Character usage analysis
    all_pred_chars = ''.join(predictions)
    char_counts = {}
    for char in all_pred_chars:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    total_chars = len(all_pred_chars)
    char_diversity = len(set(all_pred_chars))
    vocab_size = len(data_processor.answer_vocab)
    
    print(f"Character Usage:")
    print(f"Total Characters Generated: {total_chars}")
    print(f"Unique Characters Used: {char_diversity}/{vocab_size} ({char_diversity/vocab_size*100:.1f}%)")
    print(f"Character Entropy: {-sum([(count/total_chars) * np.log(count/total_chars) for count in char_counts.values()]):.3f}")
    
    # Most common characters
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Top Characters: {sorted_chars[:5]}")
    
    # Repetitive pattern detection
    repetitive_count = 0
    loop_patterns = []
    
    for pred in predictions:
        # Simple repetition detection
        if len(pred) > 2:
            # Check for repeated characters
            if len(set(pred)) == 1:
                repetitive_count += 1
                loop_patterns.append(f"All '{pred[0]}'")
            # Check for repeated short patterns
            elif len(pred) > 3:
                for pattern_len in range(1, min(4, len(pred)//2 + 1)):
                    pattern = pred[:pattern_len]
                    if pred.count(pattern) >= 3:
                        repetitive_count += 1
                        loop_patterns.append(f"'{pattern}' x{pred.count(pattern)}")
                        break
    
    print(f"\nRepetitive Pattern Detection:")
    print(f"Repetitive Predictions: {repetitive_count}/{len(predictions)} ({repetitive_count/len(predictions)*100:.1f}%)")
    if loop_patterns:
        print(f"Common Patterns: {set(loop_patterns)}")
    
    # Length distribution
    lengths = [len(pred) for pred in predictions]
    true_lengths = [len(ans) for ans in true_answers]
    
    print(f"\nLength Analysis:")
    print(f"Predicted Lengths: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} (range: {min(lengths)}-{max(lengths)})")
    print(f"True Lengths: {np.mean(true_lengths):.1f} ± {np.std(true_lengths):.1f} (range: {min(true_lengths)}-{max(true_lengths)})")
    print(f"Length Correlation: {np.corrcoef(lengths, true_lengths)[0,1]:.3f}")
    
    # Quality indicators
    exact_matches = sum(1 for p, t in zip(predictions, true_answers) if p.lower() == t.lower())
    length_matches = sum(1 for p, t in zip(predictions, true_answers) if len(p) == len(t))
    
    print(f"\n🎯 Quality Indicators:")
    print(f"Exact Matches: {exact_matches}/{len(predictions)} ({exact_matches/len(predictions)*100:.1f}%)")
    print(f"Length Matches: {length_matches}/{len(predictions)} ({length_matches/len(predictions)*100:.1f}%)")
    
    # Overall assessment
    if repetitive_count == 0:
        print(f"EXCELLENT: No repetitive patterns detected!")
    elif repetitive_count < len(predictions) * 0.1:
        print(f"GOOD: Few repetitive patterns ({repetitive_count/len(predictions)*100:.1f}%)")
    elif repetitive_count < len(predictions) * 0.3:
        print(f"WARNING: Some repetitive patterns ({repetitive_count/len(predictions)*100:.1f}%)")
    else:
        print(f"PROBLEM: Many repetitive patterns ({repetitive_count/len(predictions)*100:.1f}%)")
    
    if char_diversity > vocab_size * 0.3:
        print(f"GOOD: High character diversity ({char_diversity/vocab_size*100:.1f}%)")
    else:
        print(f"WARNING: Low character diversity ({char_diversity/vocab_size*100:.1f}%)")

def analyze_model_internals(model):
    """Analyze model internal statistics"""
    print("\n" + "="*60)
    print("MODEL INTERNAL ANALYSIS")
    print("="*60)
    
    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Layer-wise parameter analysis
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LSTM, nn.Embedding)):
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {params:,} parameters")
    
    # Weight statistics for key layers
    print(f"\nWeight Statistics:")
    
    # Attention weights
    if hasattr(model.decoder.attention, 'attn'):
        attn_weights = model.decoder.attention.attn.weight
        print(f"Attention Linear: mean={attn_weights.mean():.6f}, std={attn_weights.std():.6f}")
    
    # Encoder embedding
    emb_weights = model.encoder.embedding.weight
    print(f"Encoder Embedding: mean={emb_weights.mean():.6f}, std={emb_weights.std():.6f}")
    
    # Positional embedding
    pos_weights = model.encoder.positional_embedding.weight
    print(f"Positional Embedding: mean={pos_weights.mean():.6f}, std={pos_weights.std():.6f}")
    
    print(f"\nModel analysis complete!")

def main():
    # Create a medium subset of data for better training
    logger.info("Creating medium test dataset (1000 examples)...")
    try:
        # Read CSV with more robust parameters
        data = pd.read_csv(
            'clues_big.csv',
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
        
        # Create medium test dataset (1000 examples for better learning)
        test_data = data.sample(n=1000, random_state=42)  # Use 1000 examples
        test_data.to_csv('medium_test_data.csv', index=False)
        # Override data file for quick test while keeping original model config
        config['data'].data_file = 'medium_test_data.csv'
        config['training'].test_size = 0.2  # 20% test, 80% train (800/200 split)
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise
    
    # Initialize data processor
    logger.info("Initializing data processor...")
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
    
    # Initialize model weights to reduce repetitive outputs
    # Model uses PyTorch default initialization
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        data_processor=data_processor
    )
    
    # Create output directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('visualizations').mkdir(exist_ok=True)
    
    # Train model
    logger.info("Starting training...")
    metrics = trainer.train()
    logger.info("Training completed!")
    
    # Save final model
    trainer.save_checkpoint('checkpoints/quick_test_model.pt')
    logger.info("Model saved to checkpoints/quick_test_model.pt")
    
    # Test the model with comprehensive diagnostics
    logger.info("Running comprehensive model diagnostics including attention analysis, encoder diversity, and prediction patterns...")
    test_improved_predictions(trainer.model, data_processor, test_loader)



def analyze_prediction_quality(predictions, true_answers):
    """Analyze the quality of predictions to track improvement"""
    stats = {
        'total': len(predictions),
        'exact_matches': 0,
        'length_matches': 0,
        'contains_loops': 0,
        'reasonable_length': 0,
        'avg_length': 0,
        'unique_chars': set(),
        'common_issues': []
    }
    
    total_length = 0
    
    for pred, true in zip(predictions, true_answers):
        pred_clean = pred.strip().lower()
        true_clean = true.strip().lower()
        
        # Exact match
        if pred_clean == true_clean:
            stats['exact_matches'] += 1
        
        # Length match (good sign even if content wrong)
        if len(pred_clean) == len(true_clean):
            stats['length_matches'] += 1
        
        # Check for loops (repeated patterns)
        if len(pred_clean) > 3:
            # Simple loop detection: check for repeated substrings
            for i in range(1, len(pred_clean)//2 + 1):
                substring = pred_clean[:i]
                if pred_clean.count(substring) >= 3:
                    stats['contains_loops'] += 1
                    break
        
        # Reasonable length (3-15 characters)
        if 3 <= len(pred_clean) <= 15:
            stats['reasonable_length'] += 1
        
        total_length += len(pred_clean)
        
        # Collect unique characters
        stats['unique_chars'].update(pred_clean)
    
    stats['avg_length'] = total_length / len(predictions) if predictions else 0
    stats['unique_char_count'] = len(stats['unique_chars'])
    
    # Calculate percentages
    stats['exact_match_pct'] = (stats['exact_matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
    stats['length_match_pct'] = (stats['length_matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
    stats['loop_pct'] = (stats['contains_loops'] / stats['total'] * 100) if stats['total'] > 0 else 0
    stats['reasonable_length_pct'] = (stats['reasonable_length'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    return stats

def test_improved_predictions(model, data_processor, test_loader):
    """Test the model with comprehensive diagnostic analysis"""
    logger.info("Running comprehensive model diagnostics...")
    
    # Run model internal analysis first
    analyze_model_internals(model)
    
    # Analyze encoder diversity
    encoder_stats = analyze_encoder_diversity(model, data_processor, test_loader)
    
    # Analyze attention patterns
    attention_stats = analyze_attention_weights(model, data_processor, test_loader)
    
    # Collect test examples for prediction analysis
    test_clues = []
    test_answers = []
    
    for batch in test_loader:
        clues, answers = batch
        for clue, answer in zip(clues, answers):
            test_clues.append(data_processor.decode_clue(clue))
            test_answers.append(data_processor.decode_answer(answer))
        if len(test_clues) >= 20:  # Test on 20 examples
            break
    
    # Generate predictions with standard method
    logger.info("Generating predictions for pattern analysis...")
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for clue in test_clues:
            # Generate prediction using standard model method
            clue_tokens = data_processor.tokenize_clue(clue)
            clue_indices = data_processor.clue_vocab.encode(['<sos>'] + clue_tokens + ['<eos>'])
            clue_tensor = torch.tensor(clue_indices).to(model.device)
            
            # Generate with the model
            generated = model.generate(clue_tensor, max_length=15)
            pred = data_processor.decode_answer(generated)
            predictions.append(pred)
    
    # Analyze prediction patterns
    analyze_prediction_patterns(predictions, test_answers, data_processor)
    
    # Analyze prediction quality (existing function)
    stats = analyze_prediction_quality(predictions, test_answers)
    
    # Comprehensive summary
    print("\n" + "="*60)
    print("COMPREHENSIVE DIAGNOSTIC SUMMARY")
    print("="*60)
    
    # Attention health score
    avg_attention_variance = np.mean(attention_stats['variances'])
    avg_attention_uniformity = np.mean(attention_stats['uniformity_scores'])
    
    attention_score = 0
    if avg_attention_variance > 0.005:
        attention_score += 2
    elif avg_attention_variance > 0.002:
        attention_score += 1
    
    if avg_attention_uniformity < 0.8:
        attention_score += 2
    elif avg_attention_uniformity < 0.9:
        attention_score += 1
    
    # Encoder health score
    avg_adjacent_sim = np.mean(encoder_stats['adjacent_similarities'])
    avg_dead_dims = np.mean(encoder_stats['dead_dimensions'])
    
    encoder_score = 0
    if avg_adjacent_sim < 0.5:
        encoder_score += 2
    elif avg_adjacent_sim < 0.7:
        encoder_score += 1
    
    if avg_dead_dims < 10:
        encoder_score += 2
    elif avg_dead_dims < 25:
        encoder_score += 1
    
    # Prediction health score
    prediction_score = 0
    if stats['exact_match_pct'] > 5:
        prediction_score += 2
    elif stats['exact_match_pct'] > 0:
        prediction_score += 1
    
    if stats['loop_pct'] < 10:
        prediction_score += 2
    elif stats['loop_pct'] < 30:
        prediction_score += 1
    
    if stats['reasonable_length_pct'] > 70:
        prediction_score += 1
    
    total_score = attention_score + encoder_score + prediction_score
    max_score = 9
    
    print(f"HEALTH SCORES:")
    print(f"Attention Mechanism: {attention_score}/4 {'✅' if attention_score >= 3 else '⚠️' if attention_score >= 2 else '🚨'}")
    print(f"Encoder Diversity: {encoder_score}/4 {'✅' if encoder_score >= 3 else '⚠️' if encoder_score >= 2 else '🚨'}")
    print(f"Prediction Quality: {prediction_score}/5 {'✅' if prediction_score >= 3 else '⚠️' if prediction_score >= 2 else '🚨'}")
    print(f"OVERALL HEALTH: {total_score}/{max_score} ({total_score/max_score*100:.0f}%)")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if attention_score < 3:
        print(f"ATTENTION: Increase temperature scaling or improve encoder diversity")
    
    if encoder_score < 3:
        print(f"ENCODER: Consider stronger positional embeddings or layer norm adjustments")
    
    if prediction_score < 3:
        print(f"PREDICTIONS: May need more training epochs or data")
    
    if total_score >= 7:
        print(f"EXCELLENT: Model is performing well across all metrics!")
    elif total_score >= 5:
        print(f"GOOD: Model is functioning reasonably well")
    else:
        print(f"NEEDS WORK: Consider architectural or training improvements")
    
    # Show example predictions with attention context
    print(f"\nEXAMPLE PREDICTIONS WITH CONTEXT:")
    print("=" * 50)
    for i in range(min(5, len(test_clues))):
        match_status = "[MATCH]" if predictions[i].lower() == test_answers[i].lower() else "[WRONG]"
        length_status = "[CORRECT LEN]" if len(predictions[i]) == len(test_answers[i]) else "[WRONG LEN]"
        print(f"{match_status} {length_status} Clue: {test_clues[i]}")
        print(f" Expected: {test_answers[i]} (len: {len(test_answers[i])})")
        print(f" Predicted: {predictions[i]} (len: {len(predictions[i])})")
        print("")
    
    return stats

def run_diagnostics_on_checkpoint(checkpoint_path='checkpoints/quick_test_model.pt'):
    """Run comprehensive diagnostics on a saved model checkpoint"""
    logger.info(f"Loading model from {checkpoint_path} for diagnostic analysis...")
    
    # Initialize data processor
    data_processor = DataProcessor()
    train_loader, test_loader = data_processor.prepare_data()
    
    # Load model from checkpoint
    trainer = Trainer.load_checkpoint(
        checkpoint_path,
        train_loader,
        test_loader,
        data_processor
    )
    
    logger.info("Running comprehensive diagnostics on loaded model...")
    return test_improved_predictions(trainer.model, data_processor, test_loader)

if __name__ == '__main__':
    import sys
    
    # Check if user wants to run diagnostics on existing checkpoint
    if len(sys.argv) > 1 and sys.argv[1] == '--diagnose':
        checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/quick_test_model.pt'
        run_diagnostics_on_checkpoint(checkpoint_path)
    else:
        main() 