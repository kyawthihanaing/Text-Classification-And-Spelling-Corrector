#!/usr/bin/env python3
"""
Advanced evaluation script using ensemble ranking system
"""
import sys
import random
from pathlib import Path
sys.path.append('spelling')

from src.assets import load_vocab, load_word_freq
from src.candidates import load_symspell_words, CandidateGenerator  
from src.advanced_rank import AdvancedRanker
from src.text import normalize, tokenize

def add_typo(word):
    """Add a single typo to a word"""
    if len(word) < 2:
        return word
    
    typo_type = random.choice(['delete', 'insert', 'substitute', 'transpose'])
    
    if typo_type == 'delete':
        i = random.randint(0, len(word) - 1)
        return word[:i] + word[i+1:]
    elif typo_type == 'insert':
        i = random.randint(0, len(word))
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:i] + char + word[i:]
    elif typo_type == 'substitute':
        i = random.randint(0, len(word) - 1)
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:i] + char + word[i+1:]
    elif typo_type == 'transpose':
        if len(word) < 2:
            return word
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    
    return word

def main():
    print("Loading assets for advanced evaluation...")
    
    # Load components
    vocab = load_vocab()
    word_freqs = load_word_freq()
    symspell_words = load_symspell_words()
    
    generator = CandidateGenerator(symspell_words, vocab, radius=2)
    ranker = AdvancedRanker(word_freqs)
    
    # Test on a sample of vocabulary
    test_words = random.sample([w for w in vocab if 4 <= len(w) <= 12], 100)
    print(f"Testing on {len(test_words)} words...")
    
    results = {'top1': 0, 'top3': 0, 'top5': 0}
    total_cases = 0
    examples = []
    
    for i, word in enumerate(test_words):
        if i % 25 == 0:
            print(f"Progress: {i}/{len(test_words)}")
        
        # Create typo
        typo = add_typo(word)
        if typo == word or typo in vocab:  # Skip if no typo or typo is valid word
            continue
        
        total_cases += 1
        
        # Generate candidates
        candidates = generator.generate(typo, use_symspell=True, max_return=100)
        
        if not candidates:
            continue
        
        # Create context sentence
        sentence = ['this', typo, 'is', 'important']
        
        # Rank candidates
        suggestions = ranker.suggest(candidates, typo, sentence, 1, top_k=5)
        
        # Check accuracy
        if suggestions and suggestions[0] == word:
            results['top1'] += 1
        if word in suggestions[:3]:
            results['top3'] += 1
        if word in suggestions[:5]:
            results['top5'] += 1
        
        # Store example
        if len(examples) < 5:
            correct_marker = "✓" if word in suggestions[:5] else "✗"
            examples.append(f"{correct_marker} {word} → {typo} → {suggestions}")
    
    # Calculate accuracies
    if total_cases > 0:
        acc1 = results['top1'] / total_cases * 100
        acc3 = results['top3'] / total_cases * 100
        acc5 = results['top5'] / total_cases * 100
        
        print(f"\n=== ADVANCED EVALUATION RESULTS ===")
        print(f"Total test cases: {total_cases}")
        print(f"Top-1 accuracy: {acc1:.1f}% ({results['top1']}/{total_cases})")
        print(f"Top-3 accuracy: {acc3:.1f}% ({results['top3']}/{total_cases})")
        print(f"Top-5 accuracy: {acc5:.1f}% ({results['top5']}/{total_cases})")
        
        print(f"\n=== EXAMPLE CORRECTIONS ===")
        for example in examples:
            print(example)
        
        print(f"\n=== CSV FORMAT FOR REPORT ===")
        print(f"Metric,Value")
        print(f"Top-1 Accuracy,{acc1/100:.3f}")
        print(f"Top-3 Accuracy,{acc3/100:.3f}")
        print(f"Top-5 Accuracy,{acc5/100:.3f}")
        print(f"Test Cases,{total_cases}")
    else:
        print("No valid test cases generated")

if __name__ == "__main__":
    main()