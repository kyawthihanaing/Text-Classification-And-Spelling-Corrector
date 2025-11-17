#!/usr/bin/env python3
"""
Build SymSpell dictionary from processed word frequencies.

This script creates a SymSpell dictionary file from the word frequency data
for fast spell checking and correction suggestions.

Usage:
    python scripts/build_symspell.py --word-freq spelling/data/processed/word_freq.txt
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def generate_edits(word: str, max_distance: int = 2) -> Set[str]:
    """
    Generate all possible edits within max_distance of the input word.
    
    This includes:
    - Deletions (remove one character)
    - Insertions (add one character)  
    - Substitutions (replace one character)
    - Transpositions (swap adjacent characters)
    
    Args:
        word: Input word to generate edits for
        max_distance: Maximum edit distance (1 or 2)
        
    Returns:
        Set of all possible edit variants
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    def edits1(w):
        """Generate all edits within distance 1."""
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        deletes = [left + right[1:] for left, right in splits if right]
        transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
        replaces = [left + char + right[1:] for left, right in splits if right for char in letters]
        inserts = [left + char + right for left, right in splits for char in letters]
        return set(deletes + transposes + replaces + inserts)
    
    # Distance 1 edits
    edits_1 = edits1(word)
    
    if max_distance == 1:
        return edits_1
    
    # Distance 2 edits
    edits_2 = set()
    for edit in edits_1:
        edits_2.update(edits1(edit))
    
    return edits_1.union(edits_2)

def build_symspell_dict(word_freq_file: Path, output_file: Path, max_distance: int = 2, min_freq: int = 5):
    """
    Build SymSpell dictionary from word frequency file.
    
    Args:
        word_freq_file: Input word frequency file (word freq format)
        output_file: Output SymSpell dictionary file
        max_distance: Maximum edit distance for suggestions
        min_freq: Minimum frequency threshold for including words
    """
    logger.info(f"Building SymSpell dictionary from {word_freq_file}")
    logger.info(f"Max edit distance: {max_distance}, Min frequency: {min_freq}")
    
    # Load word frequencies
    word_freq = {}
    with open(word_freq_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                freq = int(parts[1])
                if freq >= min_freq:
                    word_freq[word] = freq
    
    logger.info(f"Loaded {len(word_freq)} words with frequency >= {min_freq}")
    
    # Build SymSpell dictionary
    symspell_dict = {}
    
    for word_num, (word, freq) in enumerate(word_freq.items(), 1):
        if word_num % 1000 == 0:
            logger.info(f"Processed {word_num}/{len(word_freq)} words...")
        
        # Add the word itself
        if word not in symspell_dict or symspell_dict[word] < freq:
            symspell_dict[word] = freq
        
        # Add all edit variants
        edits = generate_edits(word, max_distance)
        for edit in edits:
            # Only add if this edit leads to the current word and current word has higher frequency
            if edit not in symspell_dict:
                symspell_dict[edit] = freq
            else:
                # Keep the highest frequency
                symspell_dict[edit] = max(symspell_dict[edit], freq)
    
    logger.info(f"Generated {len(symspell_dict)} dictionary entries")
    
    # Save SymSpell dictionary
    logger.info(f"Saving SymSpell dictionary to {output_file}")
    
    # Sort by frequency (descending) then alphabetically
    sorted_entries = sorted(symspell_dict.items(), key=lambda x: (-x[1], x[0]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, freq in sorted_entries:
            f.write(f"{word} {freq}\n")
    
    logger.info(f"✅ SymSpell dictionary saved with {len(symspell_dict):,} entries")

def build_simple_symspell_dict(word_freq_file: Path, output_file: Path, min_freq: int = 10):
    """
    Build a simpler SymSpell dictionary that just copies high-frequency words.
    This is faster to build and still effective for common misspellings.
    
    Args:
        word_freq_file: Input word frequency file
        output_file: Output SymSpell dictionary file  
        min_freq: Minimum frequency threshold
    """
    logger.info(f"Building simple SymSpell dictionary from {word_freq_file}")
    logger.info(f"Min frequency: {min_freq}")
    
    entries_written = 0
    
    with open(word_freq_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                freq = int(parts[1])
                
                if freq >= min_freq:
                    outfile.write(f"{word} {freq}\n")
                    entries_written += 1
    
    logger.info(f"✅ Simple SymSpell dictionary saved with {entries_written:,} entries")

def main():
    parser = argparse.ArgumentParser(description="Build SymSpell dictionary from word frequencies")
    parser.add_argument("--word-freq", required=True, type=Path,
                       help="Path to word frequency file")
    parser.add_argument("--max-distance", type=int, default=2,
                       help="Maximum edit distance (default: 2)")
    parser.add_argument("--min-freq", type=int, default=5,
                       help="Minimum word frequency (default: 5)")
    parser.add_argument("--simple", action="store_true",
                       help="Build simple dictionary (just copy high-freq words)")
    
    args = parser.parse_args()
    
    if not args.word_freq.exists():
        logger.error(f"Word frequency file not found: {args.word_freq}")
        return 1
    
    # Create output directory
    output_dir = Path("spelling/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "symspell_dict.txt"
    
    if args.simple:
        build_simple_symspell_dict(args.word_freq, output_file, args.min_freq)
    else:
        build_symspell_dict(args.word_freq, output_file, args.max_distance, args.min_freq)
    
    logger.info(f"📁 SymSpell dictionary saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())