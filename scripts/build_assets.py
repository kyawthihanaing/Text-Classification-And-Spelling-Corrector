#!/usr/bin/env python3
"""
Build spelling correction assets from raw corpus text.

This script processes the raw arXiv abstracts to generate:
1. Vocabulary file (vocab.json) - unique words with frequencies
2. Language model file (trigrams.json) - trigram probabilities for text generation
3. Word frequency file (word_freq.txt) - for SymSpell dictionary

Usage:
    python scripts/build_assets.py --corpus spelling/data/raw/arxiv_abstracts.txt
"""

import argparse
import json
import re
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words using the same pattern as count_tokens.py
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens (words)
    """
    # Same tokenization pattern as count_tokens.py
    token_pattern = r'\b[A-Za-z]+(?:\'[A-Za-z]+)?\b'
    tokens = re.findall(token_pattern, text.lower())
    return tokens

def build_vocabulary(corpus_file: Path) -> Dict[str, int]:
    """
    Build vocabulary from corpus with word frequencies.
    
    Args:
        corpus_file: Path to the raw corpus text file
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    logger.info(f"Building vocabulary from {corpus_file}")
    
    word_counts = Counter()
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                logger.info(f"Processed {line_num} lines...")
            
            tokens = tokenize_text(line.strip())
            word_counts.update(tokens)
    
    logger.info(f"Found {len(word_counts)} unique words")
    return dict(word_counts)

def build_trigram_model(corpus_file: Path, vocab: Dict[str, int], min_freq: int = 5) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Build trigram language model from corpus.
    
    Args:
        corpus_file: Path to the raw corpus text file
        vocab: Vocabulary dictionary
        min_freq: Minimum frequency threshold for words to include
        
    Returns:
        Dictionary mapping (word1, word2) -> {word3: count}
    """
    logger.info(f"Building trigram model from {corpus_file}")
    
    # Filter vocabulary to frequent words only
    frequent_words = {word for word, freq in vocab.items() if freq >= min_freq}
    logger.info(f"Using {len(frequent_words)} frequent words (freq >= {min_freq})")
    
    trigrams = defaultdict(Counter)
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                logger.info(f"Processed {line_num} lines for trigrams...")
            
            tokens = tokenize_text(line.strip())
            
            # Filter to frequent words only
            tokens = [token for token in tokens if token in frequent_words]
            
            # Generate trigrams
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
                trigrams[(w1, w2)][w3] += 1
    
    # Convert to regular dict for JSON serialization
    trigram_dict = {}
    for (w1, w2), w3_counts in trigrams.items():
        key = f"{w1} {w2}"
        trigram_dict[key] = dict(w3_counts)
    
    logger.info(f"Built {len(trigram_dict)} trigram contexts")
    return trigram_dict

def save_vocab(vocab: Dict[str, int], output_file: Path):
    """Save vocabulary to JSON file."""
    logger.info(f"Saving vocabulary to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(vocab)} words to vocabulary file")

def save_trigrams(trigrams: Dict[str, Dict[str, int]], output_file: Path):
    """Save trigram model to JSON file."""
    logger.info(f"Saving trigram model to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trigrams, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(trigrams)} trigram contexts to model file")

def save_word_freq(vocab: Dict[str, int], output_file: Path):
    """
    Save word frequencies in SymSpell format (word frequency).
    
    Args:
        vocab: Vocabulary dictionary
        output_file: Output file for SymSpell dictionary
    """
    logger.info(f"Saving word frequencies to {output_file}")
    
    # Sort by frequency (descending) then alphabetically
    sorted_words = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, freq in sorted_words:
            f.write(f"{word} {freq}\n")
    
    logger.info(f"Saved {len(vocab)} word frequencies for SymSpell")

def main():
    parser = argparse.ArgumentParser(description="Build spelling correction assets from corpus")
    parser.add_argument("--corpus", required=True, type=Path, 
                       help="Path to raw corpus text file")
    parser.add_argument("--min-freq", type=int, default=5,
                       help="Minimum word frequency for trigram model (default: 5)")
    
    args = parser.parse_args()
    
    if not args.corpus.exists():
        logger.error(f"Corpus file not found: {args.corpus}")
        return 1
    
    # Create output directories
    vocab_dir = Path("spelling/data/processed")
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    # Build vocabulary
    vocab = build_vocabulary(args.corpus)
    
    # Save vocabulary
    vocab_file = vocab_dir / "vocab.json"
    save_vocab(vocab, vocab_file)
    
    # Build and save trigram model
    trigrams = build_trigram_model(args.corpus, vocab, args.min_freq)
    trigram_file = vocab_dir / "trigrams.json"
    save_trigrams(trigrams, trigram_file)
    
    # Save word frequencies for SymSpell
    word_freq_file = vocab_dir / "word_freq.txt"
    save_word_freq(vocab, word_freq_file)
    
    logger.info("✅ All spelling assets built successfully!")
    logger.info(f"📊 Vocabulary: {len(vocab):,} words")
    logger.info(f"📊 Trigrams: {len(trigrams):,} contexts")
    logger.info(f"📁 Files saved to: {vocab_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())