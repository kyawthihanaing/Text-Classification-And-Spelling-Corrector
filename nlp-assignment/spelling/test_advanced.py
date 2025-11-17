#!/usr/bin/env python3
"""
Quick evaluation of advanced ranking system
"""
import sys
import random
sys.path.append('.')

from src.assets import load_vocab, load_word_freq
from src.candidates import load_symspell_words, CandidateGenerator  
from src.advanced_rank import AdvancedRanker

def test_advanced_system():
    print("Loading components for advanced evaluation...")
    
    vocab = load_vocab()
    word_freqs = load_word_freq()
    symspell_words = load_symspell_words()
    generator = CandidateGenerator(symspell_words, vocab, radius=2)
    ranker = AdvancedRanker(word_freqs)
    
    print("Testing key cases...")
    test_cases = [
        ('teh', 'the'),
        ('helo', 'hello'), 
        ('wrold', 'world'),
        ('machne', 'machine'),
        ('algortm', 'algorithm'),
        ('papre', 'paper'),
        ('ths', 'this'),
        ('comput', 'compute'),
        ('experinces', 'experiences'),
        ('algorihm', 'algorithm')
    ]
    
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    
    for typo, correct in test_cases:
        candidates = generator.generate(typo, use_symspell=True, max_return=50)
        if not candidates:
            print(f"{typo} -> {correct}: No candidates found")
            continue
        
        sentence = ['this', typo, 'is', 'important']
        suggestions = ranker.suggest(candidates, typo, sentence, 1, top_k=5)
        
        # Check rankings
        if suggestions and suggestions[0] == correct:
            correct_top1 += 1
        if correct in suggestions[:3]:
            correct_top3 += 1
        if correct in suggestions[:5]:
            correct_top5 += 1
        
        rank = suggestions.index(correct) + 1 if correct in suggestions else -1
        status = f"Found at rank {rank}" if rank > 0 else "Not found"
        print(f"{typo} -> {correct}: {status}")
        print(f"  Suggestions: {suggestions}")
    
    total = len(test_cases)
    print(f"\n=== ADVANCED SYSTEM RESULTS ===")
    print(f"Top-1 accuracy: {correct_top1/total*100:.1f}% ({correct_top1}/{total})")
    print(f"Top-3 accuracy: {correct_top3/total*100:.1f}% ({correct_top3}/{total})")
    print(f"Top-5 accuracy: {correct_top5/total*100:.1f}% ({correct_top5}/{total})")

if __name__ == "__main__":
    test_advanced_system()