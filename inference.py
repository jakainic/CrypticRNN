import torch
from typing import List, Optional
from data_preprocessing import DataProcessor
from models import Clue2Ans

def generate_answer(
    model: Clue2Ans,
    clue: str,
    data_processor: DataProcessor,
    max_length: int = 50,
    temperature: float = 1.0
) -> str:
    """Generate an answer for a given cryptic clue
    
    Args:
        model: Trained Clue2Ans model
        clue: Input cryptic clue
        data_processor: DataProcessor instance with vocabularies
        max_length: Maximum length of generated answer
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated answer string
    """
    model.eval()
    with torch.no_grad():
        # Tokenize and encode the clue
        clue_tokens = data_processor.tokenize_clue(clue)
        clue_indices = data_processor.clue_vocab.encode(clue_tokens)
        
        # Add batch dimension and move to device
        src = torch.tensor(clue_indices).unsqueeze(1).to(model.device)
        
        # Generate answer
        output = model(src, teacher_forcing_ratio=0.0)
        
        # Apply temperature
        if temperature != 1.0:
            output = output / temperature
        
        # Get the most likely token at each step
        output_indices = output.argmax(2).squeeze(1).cpu()
        
        # Decode the answer
        answer = data_processor.decode_answer(output_indices)
        
        # Remove special tokens and stop at <eos>
        answer = answer.replace('<pad>', '').replace('<sos>', '').replace('<eos>', '')
        if '<eos>' in answer:
            answer = answer[:answer.index('<eos>')]
        
        return answer.strip()

def batch_generate_answers(
    model: Clue2Ans,
    clues: List[str],
    data_processor: DataProcessor,
    max_length: int = 50,
    temperature: float = 1.0
) -> List[str]:
    """Generate answers for multiple clues
    
    Args:
        model: Trained Clue2Ans model
        clues: List of input cryptic clues
        data_processor: DataProcessor instance with vocabularies
        max_length: Maximum length of generated answers
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        List of generated answer strings
    """
    return [generate_answer(model, clue, data_processor, max_length, temperature) 
            for clue in clues]

def evaluate_model(
    model: Clue2Ans,
    test_clues: List[str],
    test_answers: List[str],
    data_processor: DataProcessor
) -> dict:
    """Evaluate model performance on a test set
    
    Args:
        model: Trained Clue2Ans model
        test_clues: List of test clues
        test_answers: List of correct answers
        data_processor: DataProcessor instance with vocabularies
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    correct = 0
    total = len(test_clues)
    generated_answers = batch_generate_answers(model, test_clues, data_processor)
    
    for gen, true in zip(generated_answers, test_answers):
        if gen.lower() == true.lower():
            correct += 1
    
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'generated_answers': generated_answers
    } 