#!/usr/bin/env python3
import sys
sys.path.append('spelling')

from src.assets import load_vocab, load_word_freq
from src.candidates import load_symspell_words, CandidateGenerator  
from src.advanced_rank import AdvancedRanker

def test_advanced_ranking():
    print("=== ADVANCED RANKING TEST ===")
    
    # Load assets
    vocab = load_vocab()
    word_freqs = load_word_freq()
    symspell_words = load_symspell_words()
    
    # Initialize components
    generator = CandidateGenerator(symspell_words, vocab, radius=2)
    ranker = AdvancedRanker(word_freqs)
    
    # Test cases
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
    
    for misspelled, expected in test_cases:
        # Generate candidates
        candidates = generator.generate(misspelled, use_symspell=True, max_return=50)
        
        # Rank with advanced system
        sentence_tokens = ['the', misspelled, 'is', 'important']
        token_index = 1
        ranked = ranker.suggest(candidates, misspelled, sentence_tokens, token_index, top_k=5)
        
        print(f"'{misspelled}' → {ranked}")
        
        # Check if expected word is in top suggestions
        if expected in ranked:
            rank = ranked.index(expected) + 1
            print(f"  ✓ Expected '{expected}' found at rank {rank}")
        else:
            print(f"  ✗ Expected '{expected}' not in top suggestions")
        print()

if __name__ == "__main__":
    test_advanced_ranking()