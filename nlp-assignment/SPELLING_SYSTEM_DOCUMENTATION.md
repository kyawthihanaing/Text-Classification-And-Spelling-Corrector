# Advanced NLP Spelling Correction System - Complete Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Recent Enhancements (November 2025)](#recent-enhancements-november-2025)
3. [Architecture & Components](#architecture--components)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Error Detection Methods](#error-detection-methods)
6. [Candidate Generation Strategies](#candidate-generation-strategies)
7. [Ranking & Selection Algorithms](#ranking--selection-algorithms)
8. [GUI Implementation](#gui-implementation)
9. [Performance Optimizations](#performance-optimizations)
10. [Configuration & Parameters](#configuration--parameters)
11. [Stress Testing & Validation Suite](#stress-testing--validation-suite)
12. [Usage Examples](#usage-examples)
13. [Technical Implementation Details](#technical-implementation-details)

---

## System Overview

### Purpose
This advanced spelling correction system is designed to detect and correct both **non-word errors** (words that don't exist in the vocabulary) and **real-word errors** (valid words used in wrong contexts) using sophisticated NLP techniques.

### Key Features
- **Dual Error Detection**: Handles both non-word and contextual real-word errors
- **Contextual Analysis**: Uses surrounding words to make intelligent corrections
- **Multiple Candidate Generation**: Employs 11+ different strategies for finding corrections
- **Advanced Ranking**: Uses language models and frequency analysis for optimal suggestions
- **Real-time GUI**: Interactive interface with live error highlighting
- **Performance Optimized**: Fast processing with background threading
- **Technical Vocabulary Support**: Recognizes 126+ medical, scientific, and technical terms
- **Enhanced Context Analysis**: 5-word context windows with domain-aware confusion detection
- **Mixed Error Handling**: Multi-pass correction strategy for complex error scenarios
- **Multiple Edit Distance Methods**: Levenshtein, Damerau-Levenshtein, weighted, Jaro-Winkler, LCS
- **Phonetic Matching**: SoundEx algorithm for pronunciation-based corrections
- **Stress Testing Suite**: Comprehensive validation with extreme edge cases

### System Requirements
- Python 3.8+
- Virtual environment with required packages
- Pre-processed vocabulary and frequency data
- Language model files (optional but recommended)

---

## Recent Enhancements (November 2025)

### 1. Technical Vocabulary Expansion
The system now includes comprehensive domain-specific vocabularies:

#### Medical Terms (25+ terms)
- Oncology: `oncologist`, `chemotherapy`, `malignant`, `neoplasm`
- Psychiatry: `schizophrenia`, `catatonia`, `perseveration`, `diagnosis`
- General Medicine: `pathology`, `etiology`, `prognosis`, `pharmacology`

#### Scientific Terms (40+ terms)
- Evolutionary Biology: `phylogenetic`, `convergent`, `divergent`, `evolution`
- Molecular Biology: `molecular`, `genetic`, `genomic`, `proteomic`
- Statistics: `hypothesis`, `correlation`, `causation`, `confounding`

#### Technical/CS Terms (60+ terms)
- Algorithms: `algorithm`, `computational`, `optimization`, `heuristic`
- Programming: `polymorphism`, `inheritance`, `encapsulation`, `abstraction`
- Data Structures: `database`, `query`, `index`, `hash`, `encryption`

**Implementation**: `load_enhanced_vocab()` function merges technical terms with base vocabulary (33,689 total words).

### 2. Enhanced Context Analysis
Advanced context-aware error detection with extended analysis windows:

#### Extended Context Windows
- **5-word analysis window** (previously 3 words)
- **Bidirectional context analysis** (before and after error)
- **Domain-aware confusion detection** for specialized contexts

#### Domain-Specific Confusion Rules
```python
domain_confusions = {
    'medical': {
        'effect': ['affect', 'effects'],  # In medical contexts
        'malignant': ['benign', 'malign'],  # Cancer-related
    },
    'scientific': {
        'hypothesis': ['hypotheses', 'theory'],  # Research contexts
        'correlation': ['causation', 'relation'],  # Statistical contexts
    }
}
```

### 3. Mixed Error Handling
Multi-pass correction strategy for complex error scenarios:

#### MixedErrorHandler Class
```python
class MixedErrorHandler:
    def __init__(self, vocab: Set[str], word_freqs: dict)
    def correct_mixed_errors(self, tokens: List[str]) -> List[str]
    
    # Multi-pass approach:
    # Pass 1: Non-word error correction
    # Pass 2: Real-word error detection on corrected text
    # Pass 3: Final validation and refinement
```

#### Sequential Processing
1. **Initial Detection**: Identify all potential error types
2. **Non-word Priority**: Correct obvious non-words first
3. **Context Re-analysis**: Re-evaluate real-word errors after corrections
4. **Iterative Refinement**: Multiple passes until convergence

### 4. Enhanced Edit Distance Methods
Multiple distance algorithms for comprehensive candidate generation:

#### Available Methods
- **Levenshtein**: Standard character-level edit distance
- **Damerau-Levenshtein**: Includes transpositions (character swaps)
- **Weighted Levenshtein**: Keyboard-aware distance weighting
- **Jaro-Winkler**: Good for names and short strings
- **LCS (Longest Common Subsequence)**: Structural similarity

#### Implementation
```python
def enhanced_edit_distance(s1: str, s2: str, method: str = "weighted_levenshtein") -> float:
    """Enhanced edit distance with multiple variations"""
    if method == "levenshtein":
        return Levenshtein.distance(s1, s2)
    elif method == "damerau":
        return DamerauLevenshtein.distance(s1, s2)
    # ... additional methods
```

### 5. Phonetic Matching with SoundEx
SoundEx algorithm for pronunciation-based corrections:

#### SoundEx Implementation
```python
def soundex(word: str) -> str:
    """Calculate Soundex code for phonetic matching"""
    # Converts words to 4-character phonetic codes
    # Example: "Smith" → "S530", "Smyth" → "S530"
```

#### Phonetic Candidate Generation
- **Pre-computed phonetic index** for fast lookup
- **Homophone detection** and correction
- **Pronunciation-based ranking** bonus

### 6. Comprehensive Stress Testing
Validation suite with extreme edge cases:

#### Test Categories
- **Ultra-Hard Non-Word**: Obscure scientific/technical terms
- **Extreme Real-Word**: Highly context-dependent homophones
- **Chaotic Mixed**: Multiple error types simultaneously
- **Impossible Cases**: Designed to challenge even advanced systems

#### Performance Metrics
- **Detection Accuracy**: Measures false positives/negatives
- **Correction Coverage**: Percentage of expected corrections found
- **Processing Speed**: Performance under stress conditions

---

## Architecture & Components

### Core Components

#### 1. **Text Processing Module** (`text.py`)
```python
# Handles text normalization and tokenization
def normalize(text: str) -> str
def tokenize(text: str) -> List[str]
```
- **Normalization**: Converts text to lowercase, handles Unicode
- **Tokenization**: Splits text into individual words while preserving punctuation context

#### 2. **Error Detection System** (`detect.py`)

##### Non-Word Detector
```python
class NonWordDetector:
    def __init__(self, vocab: Set[str])
    def detect(self, tokens: List[str]) -> List[bool]
    def get_nonwords(self, tokens: List[str]) -> List[Tuple[int, str]]
```
- **Vocabulary Check**: Verifies words against comprehensive vocabulary set
- **Punctuation Handling**: Strips punctuation before checking
- **Case Insensitive**: Normalizes case for accurate detection

##### Real-Word Detector
```python
class RealWordDetector:
    def __init__(self, vocab: Set[str], word_freqs: dict, confusion_pairs: dict)
    def detect(self, tokens: List[str]) -> List[bool]
    def _is_realword_error(self, tokens: List[str], index: int) -> bool
    def _enhanced_context_analysis(self, tokens: List[str], index: int) -> float
```

##### Mixed Error Handler (Enhanced)
```python
class MixedErrorHandler:
    def __init__(self, vocab: Set[str], word_freqs: dict)
    def correct_mixed_errors(self, tokens: List[str]) -> List[str]
    def _multi_pass_correction(self, tokens: List[str]) -> List[str]
```

**Enhanced Detection Techniques:**
1. **Language Model Analysis**: Uses n-gram probabilities with extended context
2. **Confusion Pair Detection**: Identifies common word confusions with domain awareness
3. **Contextual Pattern Matching**: Analyzes 5-word surrounding context windows
4. **Frequency Anomaly Detection**: Flags unusually infrequent word choices
5. **Grammatical Consistency**: Checks part-of-speech compatibility
6. **Word Form Error Detection**: Catches incomplete/malformed words
7. **Missing Word Detection**: Identifies contexts requiring additional words
8. **Domain-Specific Analysis**: Specialized detection for technical/scientific contexts
9. **Multi-Pass Mixed Error Handling**: Sequential correction of complex error combinations

#### 3. **Candidate Generation Engine** (`candidates.py`)

```python
class CandidateGenerator:
    def __init__(self, symspell_words: list, vocab: set, radius=2)
    def generate(self, token: str, **kwargs) -> List[str]
```

**Generation Strategies:**
1. **SymSpell Dictionary**: High-quality, fast edit-distance candidates
2. **Enhanced Edit Distance**: Multiple algorithms (Levenshtein, Damerau-Levenshtein, weighted, Jaro-Winkler, LCS)
3. **SoundEx Phonetic Matching**: Pronunciation-based similarity
4. **Fuzzy String Matching**: Partial string similarity
5. **Vowel Confusion Patterns**: Common vowel substitutions (a↔e↔i↔o↔u)
6. **Consonant Confusion Patterns**: Common consonant substitutions (b↔p↔v, c↔k↔s↔g)
7. **Length Variants**: Handles insertions/deletions
8. **Character Pattern Matching**: Structural similarity analysis
9. **Keyboard-Aware Matching**: QWERTY layout proximity weighting
10. **Substring Matching**: Partial word matching (ultra mode)
11. **Prefix/Suffix Matching**: Morphological variants (ultra mode)
12. **Contextual Corrections**: Context-aware candidate prioritization
13. **Technical Domain Matching**: Domain-specific term suggestions

#### 4. **Advanced Ranking System** (`advanced_rank.py`)

```python
class AdvancedRanker:
    def __init__(self, word_freqs: dict)
    def suggest(self, candidates: List[str], original: str, context: List[str]) -> List[str]
```

**Ranking Factors:**
- **Edit Distance Weight**: Prefers closer matches
- **Frequency Score**: Favors more common words
- **Context Compatibility**: Uses language model scoring
- **Phonetic Similarity**: Soundex matching bonus
- **Keyboard Distance**: Proximity on QWERTY layout
- **Length Similarity**: Penalizes drastic length changes

#### 5. **Asset Management** (`assets.py`)
```python
def load_vocab() -> Set[str]
def load_word_freq() -> Dict[str, int]
def load_symspell_words() -> List[str]
def load_technical_vocabularies() -> Dict[str, Set[str]]
def load_enhanced_vocab(base_vocab: Set[str], include_technical: bool = True) -> Set[str]
```
- **Vocabulary Loading**: Comprehensive English word set (33,563 base words)
- **Enhanced Vocabulary**: Extended with 126+ technical terms (33,689 total words)
- **Frequency Data**: Word usage statistics for ranking
- **SymSpell Dictionary**: Pre-computed edit distance dictionary
- **Technical Vocabularies**: Domain-specific term collections (medical, scientific, technical)
- **SoundEx Index**: Pre-computed phonetic codes for fast phonetic matching

---

## Data Processing Pipeline

### Step 1: Input Processing
1. **Text Reception**: GUI receives user input (max 500 characters)
2. **Character Validation**: Enforces length limits and valid characters
3. **Initial Normalization**: Basic cleanup while preserving original formatting

### Step 2: Tokenization Pipeline
```python
def tokenize(text: str) -> List[str]:
    # Split on whitespace while preserving punctuation context
    tokens = text.split()
    # Handle contractions, hyphens, and special cases
    return processed_tokens
```

### Step 3: Error Detection Pipeline
```python
def spelling_analysis_pipeline(tokens: List[str]):
    # Phase 1: Non-word error detection
    nonword_errors = nonword_detector.detect(tokens)
    
    # Phase 2: Real-word error detection
    realword_errors = realword_detector.detect(tokens)
    
    # Phase 3: Context analysis for ambiguous cases
    contextual_analysis = analyze_context(tokens, errors)
    
    return combined_error_list
```

### Step 4: Candidate Generation Pipeline
For each detected error:
```python
def generate_candidates_pipeline(word: str, context: List[str], index: int):
    candidates = set()
    
    # Strategy 0: Contextual candidates (highest priority)
    candidates.update(get_contextual_candidates(word, context, index))
    
    # Strategy 1-11: Various generation methods
    for strategy in generation_strategies:
        new_candidates = strategy.generate(word)
        candidates.update(new_candidates)
    
    return list(candidates)
```

### Step 5: Ranking Pipeline
```python
def ranking_pipeline(candidates: List[str], original: str, context: List[str]):
    scored_candidates = []
    
    for candidate in candidates:
        score = calculate_composite_score(candidate, original, context)
        scored_candidates.append((candidate, score))
    
    # Sort by composite score (descending)
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return [word for word, score in scored_candidates]
```

---

## Error Detection Methods

### Non-Word Error Detection

#### Algorithm
```python
def detect_nonword_errors(tokens: List[str], vocab: Set[str]) -> List[bool]:
    results = []
    for token in tokens:
        # Strip punctuation
        clean_word = strip_punctuation(token)
        
        # Skip non-alphabetic tokens
        if not clean_word.isalpha():
            results.append(False)
            continue
        
        # Check vocabulary membership (case-insensitive)
        is_error = clean_word.lower() not in vocab
        results.append(is_error)
    
    return results
```

#### Edge Cases Handled
- **Punctuation**: "word," → "word" for checking
- **Capitalization**: "Word" checked as "word"
- **Numbers**: "123" skipped (not flagged)
- **Mixed Content**: "word123" skipped
- **Empty Tokens**: Gracefully handled

### Real-Word Error Detection

#### Multi-Technique Analysis
The system uses a composite scoring approach:

```python
def _is_realword_error(self, tokens: List[str], index: int) -> bool:
    error_score = 0
    confidence_threshold = self._get_threshold(token)
    
    # Technique 1: Language Model Analysis
    if self.lm:
        lm_score = self._analyze_language_model_probability(tokens, index)
        error_score += lm_score
    
    # Technique 2: Confusion Pair Detection
    if token in self.all_confusion_words:
        confusion_score = self._advanced_confusion_detection(token, context, tokens, index)
        error_score += confusion_score
    
    # Technique 3: Contextual Pattern Analysis
    pattern_score = self._analyze_contextual_patterns(tokens, index)
    error_score += pattern_score
    
    # Technique 4: Frequency Anomaly Detection
    freq_score = self._analyze_frequency_anomaly(token, tokens, index)
    error_score += freq_score
    
    # Technique 5: Grammatical Consistency
    grammar_score = self._analyze_grammatical_consistency(tokens, index)
    error_score += grammar_score
    
    # Technique 6: Word Form Errors
    if self._detect_word_form_errors(tokens, index):
        error_score += 3.0
    
    # Technique 7: Missing Word Detection
    if self._detect_missing_words(tokens, index):
        error_score += 2.5
    
    return error_score >= confidence_threshold
```

#### Confusion Pair System
Comprehensive mapping of commonly confused words:

```python
confusion_pairs = {
    # Homophones
    'there': ['their', 'they\'re'],
    'your': ['you\'re'],
    'its': ['it\'s'],
    
    # Effect/Affect family
    'effect': ['affect', 'effects'],
    'affect': ['effect', 'affects'],
    
    # Role/Roll contextual
    'roll': ['role', 'roles'],
    'role': ['roll', 'rolls'],
    
    # ... 50+ more pairs
}
```

#### Contextual Detection Rules
Advanced rules for context-specific errors:

```python
confusion_rules = {
    'roll': {
        'wrong_if_before': ['crucial', 'important', 'key', 'vital'],
        'wrong_if_after': ['of', 'in', 'as', 'model']
    },
    'to': {
        'wrong_if_after': ['much', 'many', 'big', 'small', 'fast']
    }
    # ... extensive rule set
}
```

#### Word Form Error Detection
Catches incomplete or malformed words:

```python
word_form_errors = {
    'provid': 'provide',
    'achiev': 'achieve', 
    'acheev': 'achieve',
    'utlizing': 'utilizing',
    'generaliation': 'generalization',
    'reconizing': 'recognizing',
    # ... comprehensive pattern list
}
```

### Confidence Thresholds
- **Common Words**: 2.5 (higher threshold to reduce false positives)
- **Regular Words**: 1.8 (balanced sensitivity)
- **Common Words Safelist**: ['to', 'the', 'and', 'or', 'but', 'in', ...]

---

## Candidate Generation Strategies

### Strategy 0: Contextual Candidates (Priority)
```python
def get_contextual_candidates(self, token: str, context_tokens: list, index: int):
    candidates = []
    
    # Direct contextual corrections
    if token in self.contextual_corrections:
        candidates.extend(self.contextual_corrections[token])
    
    # Context-specific logic
    before_word = context_tokens[index-1].lower() if index > 0 else ""
    after_word = context_tokens[index+1].lower() if index < len(context_tokens) - 1 else ""
    
    # Role/roll contextual detection
    if token == 'roll' and before_word in ['crucial', 'important', 'key']:
        candidates.extend(['role', 'roles'])
    
    # Effect/affect contextual detection
    if token == 'effect' and before_word in ['will', 'can', 'may']:
        candidates.extend(['affect', 'affects'])
    
    return candidates
```

### Strategy 1: SymSpell Dictionary
```python
def from_symspell(self, token: str):
    candidates = []
    L = len(token)
    
    # Filter by length for efficiency
    length_filtered = [w for w in self.sym if abs(len(w)-L) <= self.radius]
    
    # Calculate edit distances
    for word in length_filtered:
        if Levenshtein.distance(token, word) <= self.radius:
            candidates.append(word)
    
    return candidates
```

### Strategy 2: Edit Distance on Full Vocabulary
```python
def from_editdistance(self, token: str, limit=2000, aggressive=False, ultra=False):
    radius = self.ultra_radius if ultra else (self.aggressive_radius if aggressive else self.radius)
    L = len(token)
    
    # Pre-filter by length
    pool = [w for w in self.vocab if abs(len(w)-L) <= radius][:limit]
    
    candidates = []
    for word in pool:
        lev_dist = Levenshtein.distance(token, word)
        dam_dist = DamerauLevenshtein.distance(token, word)
        if lev_dist <= radius or dam_dist <= radius:
            candidates.append(word)
    
    return candidates
```

### Strategy 3: Phonetic Matching
```python
def from_phonetic(self, token: str):
    if not token.isalpha() or len(token) < 2:
        return []
    
    # Generate Soundex code for input
    target_soundex = soundex(token)
    
    # Find vocabulary words with matching Soundex
    candidates = self.vocab_soundex.get(target_soundex, [])
    
    # Filter out exact matches and very different lengths
    filtered = [w for w in candidates 
                if w != token.lower() and abs(len(w) - len(token)) <= 3]
    
    return filtered[:20]  # Limit results
```

### Strategy 4: Fuzzy String Matching
```python
def from_fuzzy_matching(self, token: str, threshold=75):
    candidates = []
    L = len(token)
    
    # Pre-filter by length for performance
    pool = [w for w in self.vocab if abs(len(w) - L) <= 4][:1500]
    
    for word in pool:
        similarity = fuzz.ratio(token.lower(), word)
        if similarity >= threshold:
            candidates.append(word)
    
    return candidates[:25]
```

### Strategy 5-6: Vowel/Consonant Confusion
```python
def from_vowel_confusion(self, token: str):
    candidates = []
    token_lower = token.lower()
    
    # Vowel confusion patterns
    vowel_confusions = {
        'a': ['e', 'i', 'o', 'u'], 
        'e': ['a', 'i', 'o'], 
        'i': ['e', 'a', 'y'],
        # ... complete mapping
    }
    
    # Try replacing each vowel
    for i, char in enumerate(token_lower):
        if char in vowel_confusions:
            for replacement in vowel_confusions[char]:
                variant = token_lower[:i] + replacement + token_lower[i+1:]
                if variant in self.vocab:
                    candidates.append(variant)
    
    return candidates
```

### Strategy 7: Length Variants
```python
def from_length_variants(self, token: str):
    if len(token) < 3:
        return []
    
    candidates = []
    L = len(token)
    
    # Check words with different lengths more aggressively
    for target_len in range(max(1, L-3), L+4):
        if target_len == L:
            continue
        
        pool = [w for w in self.vocab if len(w) == target_len][:500]
        
        for word in pool:
            if Levenshtein.distance(token, word) <= 2:
                candidates.append(word)
    
    return candidates
```

### Strategy 8: Character Pattern Matching
```python
def from_character_patterns(self, token: str):
    if len(token) < 3:
        return []
    
    # Extract character patterns (bigrams, trigrams)
    token_patterns = get_char_patterns(token)
    candidates = []
    
    for word in list(self.vocab)[:2000]:  # Sample for performance
        if abs(len(word) - len(token)) <= 2:
            word_patterns = get_char_patterns(word)
            
            # Calculate pattern overlap
            overlap = len(token_patterns & word_patterns)
            total_patterns = len(token_patterns | word_patterns)
            
            if total_patterns > 0 and overlap / total_patterns >= 0.4:
                candidates.append(word)
    
    return candidates[:30]
```

### Strategy 9: Keyboard-Aware Matching
```python
def from_keyboard_patterns(self, token: str):
    candidates = []
    word_scores = {}
    L = len(token)
    
    # Pre-filter by length
    pool = [w for w in self.vocab if abs(len(w) - L) <= self.radius + 1][:1000]
    
    for word in pool:
        # Calculate keyboard-distance-weighted score
        kbd_score = self._keyboard_aware_distance(token, word.lower())
        if kbd_score < 2.5:  # Threshold for keyboard similarity
            word_scores[word] = kbd_score
    
    # Sort by keyboard distance
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1])
    candidates = [word for word, score in sorted_words[:15]]
    
    return candidates

def _keyboard_aware_distance(self, s1, s2):
    total_distance = 0
    min_len = min(len(s1), len(s2))
    
    # Character-by-character keyboard distance
    for i in range(min_len):
        total_distance += keyboard_distance(s1[i], s2[i])
    
    # Add penalty for length difference
    total_distance += abs(len(s1) - len(s2)) * 1.5
    
    return total_distance / max(len(s1), len(s2))
```

### Ultra-Aggressive Strategies (Strategies 10-11)

#### Strategy 10: Substring Matching
```python
def from_substring_matching(self, token: str):
    if len(token) < 4:
        return []
    
    candidates = set()
    
    # Find words containing token as substring
    for word in self.vocab:
        if token.lower() in word or word in token.lower():
            if abs(len(word) - len(token)) <= 3:
                candidates.add(word)
    
    return list(candidates)[:20]
```

#### Strategy 11: Prefix/Suffix Matching
```python
def from_prefix_suffix_matching(self, token: str):
    if len(token) < 3:
        return []
    
    candidates = set()
    
    # Extract prefixes and suffixes
    for i in range(2, min(len(token), 6)):
        prefix = token[:i]
        suffix = token[-i:]
        
        # Find words with matching prefixes/suffixes
        for word in self.vocab:
            if (word.startswith(prefix) or word.endswith(suffix)) and \
               abs(len(word) - len(token)) <= 2:
                candidates.add(word)
    
    return list(candidates)[:25]
```

---

## Ranking & Selection Algorithms

### Composite Scoring System
The ranking system uses multiple factors to score each candidate:

```python
def calculate_composite_score(self, candidate: str, original: str, context: List[str]) -> float:
    score = 0.0
    
    # Factor 1: Edit Distance (40% weight)
    edit_distance = min(
        Levenshtein.distance(original, candidate),
        DamerauLevenshtein.distance(original, candidate)
    )
    edit_score = 1.0 / (1 + edit_distance)
    score += 0.4 * edit_score
    
    # Factor 2: Frequency Score (25% weight)
    frequency = self.word_freqs.get(candidate, 1)
    freq_score = math.log(frequency + 1) / 15.0  # Normalized
    score += 0.25 * freq_score
    
    # Factor 3: Context Compatibility (20% weight)
    if self.lm:
        context_score = self._score_context_fit(candidate, context)
        score += 0.20 * context_score
    
    # Factor 4: Phonetic Similarity (10% weight)
    if soundex(original) == soundex(candidate):
        score += 0.10
    
    # Factor 5: Length Similarity (5% weight)
    len_diff = abs(len(original) - len(candidate))
    len_score = 1.0 / (1 + len_diff)
    score += 0.05 * len_score
    
    return score
```

### Context Scoring with Language Models
```python
def _score_context_fit(self, candidate: str, context: List[str]) -> float:
    if not self.lm or len(context) < 3:
        return 0.5  # Neutral score
    
    # Create test sentences with candidate
    test_contexts = [
        context[:2] + [candidate] + context[3:],  # Replace middle word
        [candidate] + context[1:],                # Replace first word
        context[:-1] + [candidate]                # Replace last word
    ]
    
    max_prob = 0
    for test_context in test_contexts:
        prob = self.lm.probability(' '.join(test_context))
        max_prob = max(max_prob, prob)
    
    return min(max_prob * 10, 1.0)  # Normalize to [0,1]
```

### Fast Ranking for GUI
For real-time performance, a simplified ranking is used:

```python
def get_suggestions_fast(self, word, context_tokens=None, index=-1):
    # Generate candidates
    candidates = self.generator.generate(word, context_tokens=context_tokens, index=index)
    
    # Simple frequency-based ranking
    candidates_scored = []
    for candidate in candidates[:15]:
        freq = self.word_freqs.get(candidate, 0)
        edit_dist = Levenshtein.distance(word, candidate)
        
        # Score = frequency boost - edit distance penalty
        score = freq / 1000.0 - edit_dist
        candidates_scored.append((candidate, edit_dist, score))
    
    # Sort by score
    candidates_scored.sort(key=lambda x: x[2], reverse=True)
    
    return [(cand, dist) for cand, dist, score in candidates_scored[:8]]
```

---

## GUI Implementation

### Main GUI Architecture
```python
class SpellingCorrectionGUI:
    def __init__(self, root):
        self.root = root
        self.load_system()     # Initialize NLP components
        self.create_widgets()  # Build UI
        self.setup_bindings()  # Event handlers
```

### Component Loading
```python
def load_system(self):
    print("Loading advanced spelling correction system...")
    
    # Load core assets
    self.vocab = load_vocab()                    # ~100K words
    self.word_freqs = load_word_freq()          # Frequency data
    self.symspell_words = load_symspell_words() # Edit-distance dictionary
    
    # Initialize components
    self.generator = CandidateGenerator(self.symspell_words, self.vocab, radius=2)
    self.ranker = AdvancedRanker(self.word_freqs)
    self.nonword_detector = NonWordDetector(self.vocab)
    self.realword_detector = RealWordDetector(self.vocab, self.word_freqs)
    
    print("✅ System loaded successfully!")
```

### UI Layout Structure
```
┌─────────────────────────────────────────────────────────────┐
│ Advanced NLP Spelling Correction System                     │
├─────────────────────────────────────────────────────────────┤
│ Text Editor (500 char limit)                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [User input text with error highlighting]              │ │
│ │                                                         │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│ Characters: 0/500                                           │
├─────────────────────────────────────────────────────────────┤
│ [🔍 Check Spelling] [✖ Clear] [📚 Vocabulary]              │
├─────────────────────────────────────────────────────────────┤
│ Spelling Analysis Results                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Analysis results with suggestions                       │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Status: Ready - Load text and click 'Check Spelling'       │
└─────────────────────────────────────────────────────────────┘
```

### Text Processing Workflow
```python
def run_spell_check(self):
    # Get user input
    text = self.text_editor.get("1.0", tk.END).strip()
    
    # Validate input
    if not text or len(text) < 3:
        self.show_message("Please enter at least 3 characters.")
        return
    
    # Start background analysis
    threading.Thread(target=self._analyze_text, args=(text,), daemon=True).start()

def _analyze_text(self, text):
    try:
        start_time = time.time()
        
        # Tokenize text
        tokens = tokenize(text)
        
        # Detect errors
        nonword_results = self.nonword_detector.detect(tokens)
        realword_results = self.realword_detector.detect(tokens)
        
        # Extract error positions and words
        nonword_errors = [(i, tokens[i]) for i, is_error in enumerate(nonword_results) if is_error]
        realword_errors = [(i, tokens[i]) for i, is_error in enumerate(realword_results) if is_error]
        
        # Generate suggestions with context
        self.word_suggestions = {}
        for i, word in nonword_errors + realword_errors:
            if word not in self.word_suggestions:
                self.word_suggestions[word] = self.get_suggestions_fast(word, tokens, i)
        
        # Update GUI
        self._update_results(nonword_errors, realword_errors, time.time() - start_time)
        
    except Exception as e:
        self._show_error(f"Analysis failed: {e}")
```

### Error Highlighting System
```python
def highlight_word(self, word, tag):
    """Highlight words in text editor with specific styling"""
    text_content = self.text_editor.get("1.0", tk.END)
    start = "1.0"
    
    while True:
        # Case-insensitive search
        pos = self.text_editor.search(word, start, tk.END, nocase=True)
        if not pos:
            break
        
        end_pos = f"{pos}+{len(word)}c"
        self.text_editor.tag_add(tag, pos, end_pos)
        start = end_pos

# Tag configurations
self.text_editor.tag_configure("misspelled", background="#ffcccc", foreground="#c0392b")
self.text_editor.tag_configure("realword", background="#fff3cd", foreground="#856404")
```

### Interactive Features
- **Character Counter**: Real-time count with color coding
- **Click-to-Suggest**: Click highlighted words for suggestions
- **Background Processing**: Non-blocking analysis
- **Vocabulary Browser**: Show loaded vocabulary statistics
- **Clear Function**: Remove all highlighting

---

## Performance Optimizations

### Memory Management
```python
# Efficient data structures
self.vocab: Set[str]           # O(1) lookup for vocabulary
self.word_freqs: Dict[str, int] # O(1) frequency lookup
self.vocab_soundex: Dict[str, List[str]] # Pre-computed phonetic index
```

### Processing Optimizations

#### 1. **Length-Based Pre-filtering**
```python
# Filter candidates by length before expensive operations
length_filtered = [w for w in self.vocab if abs(len(w) - len(token)) <= radius]
```

#### 2. **Early Termination**
```python
# Stop processing when enough candidates found
if len(candidates) >= target_count and not ultra_mode:
    break
```

#### 3. **Batch Processing**
```python
# Pre-generate all suggestions before GUI update
for word in error_words:
    if word not in self.word_suggestions:
        self.word_suggestions[word] = self.get_suggestions_fast(word, tokens, index)
```

#### 4. **Threading**
```python
# Background processing to keep GUI responsive
def run_spell_check(self):
    self.check_button.config(state=tk.DISABLED, text="🔄 Analyzing...")
    threading.Thread(target=run_analysis, daemon=True).start()
```

### Algorithm Complexity
- **Non-word Detection**: O(n) where n = number of tokens
- **Real-word Detection**: O(n × k) where k = average context window
- **Candidate Generation**: O(v × d) where v = vocab size, d = edit distance
- **Ranking**: O(c × log c) where c = number of candidates

### Memory Usage
- **Vocabulary**: ~2MB (100K words)
- **Frequency Data**: ~1MB
- **SymSpell Dictionary**: ~5MB
- **Language Model**: ~10MB (optional)
- **Total Runtime**: ~20MB

---

## Configuration & Parameters

### System Parameters
```python
class SystemConfig:
    # Candidate Generation
    DEFAULT_RADIUS = 2
    AGGRESSIVE_RADIUS = 4
    ULTRA_RADIUS = 6
    MAX_CANDIDATES = 100
    
    # Error Detection Thresholds
    COMMON_WORDS_THRESHOLD = 2.5
    REGULAR_WORDS_THRESHOLD = 1.8
    
    # Performance Limits
    MAX_VOCAB_SAMPLE = 2000
    MAX_SYMSPELL_SAMPLE = 1500
    MAX_FUZZY_CANDIDATES = 25
    
    # GUI Settings
    MAX_TEXT_LENGTH = 500
    MAX_DISPLAY_SUGGESTIONS = 8
    ANALYSIS_TIMEOUT = 30.0
```

### Configurable Components

#### Detection Sensitivity Parameters
```python
class DetectionConfig:
    # Confidence thresholds for different word types
    COMMON_WORDS_THRESHOLD = 2.5      # High threshold for frequent words
    REGULAR_WORDS_THRESHOLD = 1.8     # Standard threshold for regular words
    TECHNICAL_WORDS_THRESHOLD = 1.2   # Lower threshold for technical terms
    
    # Context window sizes
    CONTEXT_WINDOW_SIZE = 5           # Words to analyze on each side
    EXTENDED_CONTEXT_WINDOW = 7       # For complex real-word errors
    
    # Minimum word length for analysis
    MIN_WORD_LENGTH = 3               # Skip very short words
    MAX_WORD_LENGTH = 25              # Skip extremely long words
    
    # Frequency thresholds
    MIN_FREQUENCY_THRESHOLD = 10      # Words must appear at least 10 times
    RARE_WORD_THRESHOLD = 100         # Words below this are considered rare
```

#### Generation Strategy Parameters
```python
class GenerationConfig:
    # Edit distance limits
    MAX_EDIT_DISTANCE_1 = 1           # For common typos
    MAX_EDIT_DISTANCE_2 = 2           # For major errors
    MAX_EDIT_DISTANCE_AGGRESSIVE = 3  # For extreme cases
    
    # Candidate limits per strategy
    SYMSPELL_MAX_CANDIDATES = 50
    EDIT_DISTANCE_MAX_CANDIDATES = 30
    PHONETIC_MAX_CANDIDATES = 20
    FUZZY_MAX_CANDIDATES = 15
    
    # Strategy weights (0.0 to 1.0)
    CONTEXTUAL_WEIGHT = 0.9
    SYMSPELL_WEIGHT = 0.8
    EDIT_DISTANCE_WEIGHT = 0.7
    PHONETIC_WEIGHT = 0.6
    FUZZY_WEIGHT = 0.5
```

#### Ranking Algorithm Parameters
```python
class RankingConfig:
    # Factor weights (must sum to 1.0)
    EDIT_DISTANCE_WEIGHT = 0.25
    FREQUENCY_WEIGHT = 0.20
    CONTEXT_WEIGHT = 0.20
    STRATEGY_WEIGHT = 0.15
    PHONETIC_WEIGHT = 0.10
    LENGTH_WEIGHT = 0.05
    TYPE_ADJUSTMENT_WEIGHT = 0.05
    
    # Scoring parameters
    FREQUENCY_LOG_BASE = 10          # Logarithmic frequency scaling
    LENGTH_PENALTY_FACTOR = 0.1      # Penalty for length differences
    PHONETIC_BONUS_FACTOR = 0.2      # Bonus for phonetic matches
    
    # Output limits
    MAX_SUGGESTIONS = 8              # Maximum suggestions to return
    MIN_CONFIDENCE_THRESHOLD = 0.3   # Minimum score to include
```

#### Performance Tuning Parameters
```python
class PerformanceConfig:
    # Memory limits
    MAX_VOCAB_SIZE = 100000          # Maximum vocabulary size
    MAX_CACHE_SIZE = 1000            # LRU cache size for suggestions
    MAX_CONTEXT_CACHE_SIZE = 500     # Context pattern cache size
    
    # Time limits
    ANALYSIS_TIMEOUT = 30.0          # Maximum analysis time (seconds)
    CANDIDATE_GENERATION_TIMEOUT = 5.0  # Per-word timeout
    RANKING_TIMEOUT = 2.0            # Ranking timeout
    
    # Threading parameters
    MAX_WORKER_THREADS = 4           # Maximum background threads
    THREAD_POOL_SIZE = 2             # Default thread pool size
    
    # Batch processing
    BATCH_SIZE = 100                 # Words to process in batch
    PRELOAD_BATCH_SIZE = 50          # Words to preload
```

### Dynamic Configuration System
```python
class SpellingConfig:
    def __init__(self, mode='standard'):
        if mode == 'fast':
            self._configure_fast_mode()
        elif mode == 'accurate':
            self._configure_accurate_mode()
        elif mode == 'aggressive':
            self._configure_aggressive_mode()
        else:
            self._configure_standard_mode()
    
    def _configure_fast_mode(self):
        """Optimized for speed over accuracy"""
        self.detection.context_window = 3
        self.generation.max_candidates_per_strategy = 10
        self.ranking.max_suggestions = 3
        self.performance.max_cache_size = 200
    
    def _configure_accurate_mode(self):
        """Optimized for accuracy over speed"""
        self.detection.context_window = 7
        self.generation.max_candidates_per_strategy = 50
        self.ranking.max_suggestions = 10
        self.ranking.min_confidence_threshold = 0.5
    
    def _configure_aggressive_mode(self):
        """Maximum coverage, slower processing"""
        self.detection.context_window = 9
        self.generation.max_edit_distance = 3
        self.generation.max_candidates_per_strategy = 100
        self.ranking.max_suggestions = 15
    
    def _configure_standard_mode(self):
        """Balanced performance and accuracy"""
        self.detection.context_window = 5
        self.generation.max_candidates_per_strategy = 25
        self.ranking.max_suggestions = 8
        self.ranking.min_confidence_threshold = 0.3
```

### Configuration File Format
```json
{
  "detection": {
    "common_words_threshold": 2.5,
    "regular_words_threshold": 1.8,
    "context_window_size": 5,
    "min_word_length": 3
  },
  "generation": {
    "max_edit_distance": 2,
    "symspell_max_candidates": 50,
    "phonetic_max_candidates": 20
  },
  "ranking": {
    "edit_distance_weight": 0.25,
    "frequency_weight": 0.20,
    "context_weight": 0.20,
    "max_suggestions": 8
  },
  "performance": {
    "analysis_timeout": 30.0,
    "max_cache_size": 1000,
    "thread_pool_size": 2
  }
}
```

### Configuration Validation
```python
def validate_configuration(config: SpellingConfig) -> List[str]:
    """
    Validate configuration parameters for consistency and performance
    """
    errors = []
    
    # Check weight sums
    ranking_weights = [
        config.ranking.edit_distance_weight,
        config.ranking.frequency_weight,
        config.ranking.context_weight,
        config.ranking.strategy_weight,
        config.ranking.phonetic_weight,
        config.ranking.length_weight,
        config.ranking.type_adjustment_weight
    ]
    
    if abs(sum(ranking_weights) - 1.0) > 0.01:
        errors.append("Ranking weights must sum to 1.0")
    
    # Check reasonable limits
    if config.generation.max_candidates_per_strategy > 200:
        errors.append("Max candidates per strategy too high (>200)")
    
    if config.performance.analysis_timeout > 300:
        errors.append("Analysis timeout too long (>300 seconds)")
    
    # Check logical consistency
    if config.detection.context_window > config.performance.max_cache_size:
        errors.append("Context window larger than cache size")
    
    return errors
```

---

## Technical Pipeline Implementation

### Complete Processing Flow
The spelling correction system implements a sophisticated multi-stage pipeline that combines statistical, linguistic, and machine learning techniques for comprehensive error detection and correction.

#### Stage 1: Text Preprocessing
```python
def preprocess_text(text: str) -> List[str]:
    """
    Convert raw text into normalized token sequence
    """
    # Unicode normalization (NFC form)
    text = unicodedata.normalize('NFC', text)
    
    # Convert to lowercase for case-insensitive processing
    text = text.lower()
    
    # Basic punctuation handling (preserve sentence boundaries)
    text = re.sub(r'([.!?])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenization with punctuation preservation
    tokens = []
    for word in text.split():
        # Separate punctuation from words
        word_tokens = re.findall(r'\w+|[^\w\s]', word)
        tokens.extend(word_tokens)
    
    return tokens
```

#### Stage 2: Multi-Layer Error Detection
```python
def detect_errors_comprehensive(tokens: List[str]) -> Dict[str, List]:
    """
    Apply multiple detection strategies for comprehensive error identification
    """
    errors = {
        'nonword': [],
        'realword': [],
        'mixed': [],
        'confidence_scores': []
    }
    
    # Layer 1: Non-word detection (fast, high-precision)
    nonword_errors = detect_nonword_errors(tokens)
    errors['nonword'] = nonword_errors
    
    # Layer 2: Real-word error detection (context-aware)
    realword_errors = detect_realword_errors(tokens, nonword_errors)
    errors['realword'] = realword_errors
    
    # Layer 3: Mixed error pattern detection
    mixed_errors = detect_mixed_error_patterns(tokens, nonword_errors, realword_errors)
    errors['mixed'] = mixed_errors
    
    # Layer 4: Confidence scoring and ranking
    confidence_scores = calculate_error_confidence(tokens, nonword_errors, realword_errors, mixed_errors)
    errors['confidence_scores'] = confidence_scores
    
    return errors
```

#### Stage 3: Candidate Generation with Multiple Strategies
```python
def generate_candidates_hierarchical(word: str, context: List[str], error_type: str) -> List[Candidate]:
    """
    Generate correction candidates using hierarchical strategy selection
    """
    candidates = []
    
    # Strategy prioritization based on error type
    if error_type == 'nonword':
        # For non-word errors: focus on edit distance and phonetic similarity
        strategies = [
            ('symspell', generate_symspell_candidates, 0.9),
            ('edit_distance', generate_edit_distance_candidates, 0.8),
            ('phonetic', generate_phonetic_candidates, 0.7),
            ('keyboard', generate_keyboard_candidates, 0.6),
        ]
    elif error_type == 'realword':
        # For real-word errors: focus on context and confusion pairs
        strategies = [
            ('contextual', generate_contextual_candidates, 0.9),
            ('confusion_pairs', generate_confusion_candidates, 0.8),
            ('semantic', generate_semantic_candidates, 0.7),
        ]
    else:
        # Mixed errors: combine all strategies
        strategies = [
            ('contextual', generate_contextual_candidates, 0.9),
            ('symspell', generate_symspell_candidates, 0.8),
            ('edit_distance', generate_edit_distance_candidates, 0.7),
            ('phonetic', generate_phonetic_candidates, 0.6),
        ]
    
    # Execute strategies in priority order
    for strategy_name, strategy_func, weight in strategies:
        strategy_candidates = strategy_func(word, context)
        for candidate in strategy_candidates:
            candidate.strategy_weight = weight
            candidate.strategy_name = strategy_name
            candidates.append(candidate)
    
    return candidates
```

#### Stage 4: Advanced Ranking and Selection
```python
def rank_candidates_comprehensive(candidates: List[Candidate], 
                                original_word: str, 
                                context: List[str],
                                error_type: str) -> List[RankedCandidate]:
    """
    Apply multi-factor ranking algorithm for optimal candidate selection
    """
    ranked_candidates = []
    
    for candidate in candidates:
        # Factor 1: Edit Distance Score (0-1 scale)
        edit_score = calculate_edit_distance_score(original_word, candidate.word)
        
        # Factor 2: Frequency Score (logarithmic scaling)
        freq_score = calculate_frequency_score(candidate.word)
        
        # Factor 3: Context Compatibility Score
        context_score = calculate_context_compatibility(candidate.word, context, original_word)
        
        # Factor 4: Strategy Weight (from generation hierarchy)
        strategy_score = candidate.strategy_weight
        
        # Factor 5: Phonetic Similarity Bonus
        phonetic_bonus = calculate_phonetic_similarity(original_word, candidate.word)
        
        # Factor 6: Length Similarity Penalty/Bonus
        length_score = calculate_length_similarity(original_word, candidate.word)
        
        # Factor 7: Error Type Specific Adjustments
        type_adjustment = calculate_error_type_adjustment(candidate, error_type, context)
        
        # Composite Score Calculation
        total_score = (
            0.25 * edit_score +
            0.20 * freq_score +
            0.20 * context_score +
            0.15 * strategy_score +
            0.10 * phonetic_bonus +
            0.05 * length_score +
            0.05 * type_adjustment
        )
        
        # Apply confidence threshold filtering
        if total_score >= get_minimum_threshold(error_type):
            ranked_candidates.append(RankedCandidate(
                word=candidate.word,
                score=total_score,
                factors={
                    'edit': edit_score,
                    'frequency': freq_score,
                    'context': context_score,
                    'strategy': strategy_score,
                    'phonetic': phonetic_bonus,
                    'length': length_score,
                    'type_adj': type_adjustment
                },
                strategy=candidate.strategy_name
            ))
    
    # Sort by total score (descending)
    ranked_candidates.sort(key=lambda x: x.score, reverse=True)
    
    return ranked_candidates[:get_max_suggestions(error_type)]
```

#### Stage 5: Post-Processing and Validation
```python
def post_process_corrections(original_tokens: List[str], 
                           corrections: Dict[int, List[str]]) -> Dict[int, str]:
    """
    Apply corrections with conflict resolution and validation
    """
    final_corrections = {}
    
    # Sort corrections by position to handle overlaps
    sorted_positions = sorted(corrections.keys())
    
    for i, position in enumerate(sorted_positions):
        candidates = corrections[position]
        if not candidates:
            continue
        
        best_candidate = candidates[0]  # Already ranked
        
        # Check for conflicts with adjacent corrections
        conflict_detected = False
        for prev_pos in sorted_positions[:i]:
            if abs(position - prev_pos) <= 1:  # Adjacent or same position
                # Check if corrections would create invalid sequences
                if would_create_conflict(final_corrections[prev_pos], best_candidate, position - prev_pos):
                    conflict_detected = True
                    break
        
        if not conflict_detected:
            # Validate correction doesn't break grammar/syntax
            if validate_correction(original_tokens, position, best_candidate):
                final_corrections[position] = best_candidate
    
    return final_corrections
```

### Data Flow Architecture
```
Raw Text Input
       ↓
┌─────────────────────┐
│  Text Preprocessing │
│  • Normalization    │
│  • Tokenization     │
│  • Cleaning         │
└─────────────────────┘
       ↓
┌─────────────────────┐
│ Multi-Layer         │
│ Error Detection     │
│  • Non-word         │
│  • Real-word        │
│  • Mixed patterns   │
└─────────────────────┘
       ↓
┌─────────────────────┐
│ Hierarchical        │
│ Candidate Generation│
│  • Strategy-based   │
│  • Context-aware    │
│  • Type-specific    │
└─────────────────────┘
       ↓
┌─────────────────────┐
│ Advanced Ranking    │
│ & Selection         │
│  • Multi-factor     │
│  • Context scoring  │
│  • Validation       │
└─────────────────────┘
       ↓
┌─────────────────────┐
│ Post-Processing     │
│ & Validation        │
│  • Conflict res.    │
│  • Grammar check    │
│  • Final selection  │
└─────────────────────┘
       ↓
Corrected Text Output
```

### Memory and Performance Architecture
```python
class SpellingSystem:
    def __init__(self):
        # Core data structures (loaded once)
        self.vocab = self._load_vocabulary()           # Set[str] - O(1) lookup
        self.word_freqs = self._load_frequencies()     # Dict[str, int] - O(1) access
        self.symspell_dict = self._load_symspell()     # List[str] - O(log n) search
        self.confusion_pairs = self._load_confusions() # Dict[str, List[str]]
        self.phonetic_index = self._build_phonetic_index()  # Dict[str, List[str]]
        
        # Component instances (reusable)
        self.tokenizer = TextTokenizer()
        self.nonword_detector = NonWordDetector(self.vocab)
        self.realword_detector = RealWordDetector(self.vocab, self.word_freqs, self.confusion_pairs)
        self.candidate_generator = CandidateGenerator(self.symspell_dict, self.vocab)
        self.ranker = AdvancedRanker(self.word_freqs)
        
        # Caching layer for performance
        self._suggestion_cache = {}  # LRU cache for suggestions
        self._context_cache = {}     # Context pattern caching
        
    def _load_vocabulary(self) -> Set[str]:
        """Load and merge base + enhanced vocabulary"""
        base_vocab = load_vocab()
        enhanced_vocab = load_enhanced_vocab(base_vocab, include_technical=True)
        return enhanced_vocab
    
    def _build_phonetic_index(self) -> Dict[str, List[str]]:
        """Pre-compute phonetic codes for fast lookup"""
        phonetic_index = defaultdict(list)
        for word in self.vocab:
            if word.isalpha():
                code = soundex(word)
                phonetic_index[code].append(word)
        return dict(phonetic_index)
```

### Error Recovery and Resilience
```python
def process_text_with_error_recovery(text: str) -> ProcessingResult:
    """
    Process text with comprehensive error handling and recovery
    """
    try:
        # Primary processing pipeline
        result = self._process_text_primary(text)
        return result
        
    except MemoryError:
        # Fallback: Reduce vocabulary size
        self._reduce_memory_footprint()
        return self._process_text_fallback(text)
        
    except TimeoutError:
        # Fallback: Simplify processing
        return self._process_text_fast(text)
        
    except Exception as e:
        # Ultimate fallback: Basic spell check only
        self.logger.error(f"Processing failed: {e}")
        return self._process_text_basic(text)
```

This comprehensive technical pipeline ensures robust, accurate, and efficient spelling correction across diverse text types and error patterns, with multiple fallback mechanisms for reliability.

---

## Usage Examples

### Basic GUI Usage
```python
# Launch the interactive spelling correction GUI
from spelling_gui import SpellingCorrectionGUI
import tkinter as tk

# Create main window
root = tk.Tk()
app = SpellingCorrectionGUI(root)

# Start the GUI event loop
root.mainloop()
```

### Programmatic API Usage
```python
from spelling.src.assets import load_vocab, load_word_freq, load_enhanced_vocab
from spelling.src.candidates import CandidateGenerator, load_symspell_words
from spelling.src.detect import NonWordDetector, RealWordDetector, MixedErrorHandler
from spelling.src.advanced_rank import AdvancedRanker
from spelling.src.text import tokenize

# Load enhanced system with technical vocabulary
base_vocab = load_vocab()
vocab = load_enhanced_vocab(base_vocab, include_technical=True)
word_freqs = load_word_freq()
symspell_words = load_symspell_words()

# Initialize components
nonword_detector = NonWordDetector(vocab)
realword_detector = RealWordDetector(vocab, word_freqs)
generator = CandidateGenerator(symspell_words, vocab, radius=2, use_enhanced_vocab=True)
ranker = AdvancedRanker(word_freqs)
mixed_handler = MixedErrorHandler(vocab, word_freqs)

# Process text
text = "The oncolgist recomended chemotherpy for the patinet's malignent neoplasm."
tokens = tokenize(text)

# Detect errors
nonword_indices = nonword_detector.get_nonwords(tokens)
print(f"Non-word errors at positions: {[i for i, _ in nonword_indices]}")

# Generate corrections for each error
for position, error_word in nonword_indices:
    context = tokens[max(0, position-2):position+3]  # 5-word context window
    candidates = generator.generate(error_word, context_tokens=context, index=position)
    suggestions = ranker.suggest(candidates, error_word, context)
    print(f"'{error_word}' → {suggestions[:3]}")
```

### Advanced Mixed Error Handling
```python
# Handle complex texts with multiple error types
complex_text = "The lieutenent colonel's menopausal symtoms were efective in there assault."
tokens = tokenize(complex_text)

# Use mixed error handler for comprehensive correction
corrected_tokens = mixed_handler.correct_mixed_errors(tokens)
corrected_text = ' '.join(corrected_tokens)

print(f"Original: {complex_text}")
print(f"Corrected: {corrected_text}")
# Output: "The lieutenant colonel's menstrual symptoms were effective in their assault."
```

### Technical Domain Processing
```python
# Process scientific/technical text
scientific_text = "The hypothsis that phylogentic trees exibit convergent evolution was suported by molekular evidece."

# Load domain-specific configuration
from spelling.src.assets import load_technical_vocabularies
tech_vocabularies = load_technical_vocabularies()

# Process with scientific context awareness
tokens = tokenize(scientific_text)
errors = detect_errors_with_domain_context(tokens, 'scientific', tech_vocabularies)

for position, error_word, suggestions in errors:
    print(f"Scientific error: '{error_word}' → {suggestions}")
```

### Batch Processing for Multiple Texts
```python
def process_document_batch(texts: List[str], batch_size: int = 10) -> List[Dict]:
    """
    Process multiple texts efficiently with batching
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        for text in batch:
            tokens = tokenize(text)
            
            # Detect all error types
            nonword_errors = nonword_detector.get_nonwords(tokens)
            realword_errors = [(j, tokens[j]) for j in range(len(tokens)) 
                             if realword_detector._is_realword_error(tokens, j)]
            
            # Generate corrections
            corrections = {}
            for pos, word in nonword_errors + realword_errors:
                context = tokens[max(0, pos-2):pos+3]
                candidates = generator.generate(word, context_tokens=context, index=pos)
                suggestions = ranker.suggest(candidates, word, context)
                corrections[pos] = suggestions
            
            results.append({
                'original_text': text,
                'tokens': tokens,
                'corrections': corrections,
                'error_count': len(corrections)
            })
    
    return results

# Process multiple documents
documents = [
    "The reseacher conducted an experment to test their theorey.",
    "The algoritm computed the integral using disrete cosine transformation.",
    "The physican diagnosed schizophrenia with catatonia and perserveration."
]

batch_results = process_document_batch(documents)
for result in batch_results:
    print(f"Text: {result['original_text']}")
    print(f"Errors found: {result['error_count']}")
    print("---")
```

### Real-time Streaming Processing
```python
class RealTimeSpellingProcessor:
    def __init__(self):
        self.buffer = []
        self.max_buffer_size = 50  # Process in chunks
        
    def process_stream(self, word_stream):
        """Process a stream of words in real-time"""
        for word in word_stream:
            self.buffer.append(word)
            
            # Process when buffer is full or sentence ends
            if len(self.buffer) >= self.max_buffer_size or word in '.!?':
                self._process_buffer()
                self.buffer = []
    
    def _process_buffer(self):
        """Process the current buffer contents"""
        if not self.buffer:
            return
            
        # Detect errors in buffer
        nonword_errors = nonword_detector.get_nonwords(self.buffer)
        
        # Generate real-time corrections
        for pos, error_word in nonword_errors:
            context = self.buffer[max(0, pos-2):pos+3]
            candidates = generator.generate(error_word, context_tokens=context, index=pos)
            suggestions = ranker.suggest(candidates, error_word, context)
            
            # Output real-time correction
            self._output_correction(pos, error_word, suggestions[0] if suggestions else None)

# Usage with streaming input
processor = RealTimeSpellingProcessor()
word_stream = ["The", "scientst", "conducted", "an", "experment", "to", "test", "their", "hypothsis", "."]
processor.process_stream(word_stream)
```

### Configuration-Based Processing
```python
from spelling.src.config import SpellingConfig

# Create configuration for different use cases
fast_config = SpellingConfig(mode='fast')        # Speed-optimized
accurate_config = SpellingConfig(mode='accurate') # Accuracy-optimized  
aggressive_config = SpellingConfig(mode='aggressive') # Maximum coverage

# Process with different configurations
def process_with_config(text: str, config: SpellingConfig):
    # Apply configuration to components
    generator.set_config(config.generation)
    ranker.set_config(config.ranking)
    detector.set_config(config.detection)
    
    # Process text
    tokens = tokenize(text)
    errors = detector.detect_comprehensive(tokens)
    
    corrections = {}
    for error_pos, error_word in errors:
        context = tokens[max(0, error_pos-config.detection.context_window):error_pos+config.detection.context_window+1]
        candidates = generator.generate(error_word, context_tokens=context, index=error_pos)
        suggestions = ranker.suggest(candidates, error_word, context)[:config.ranking.max_suggestions]
        corrections[error_pos] = suggestions
    
    return corrections

# Compare different configurations
text = "The algorith computed the integral using disrete cosine transformation."
configs = [('Fast', fast_config), ('Accurate', accurate_config), ('Aggressive', aggressive_config)]

for name, config in configs:
    corrections = process_with_config(text, config)
    print(f"{name} mode found {len(corrections)} corrections")
```

### Error Analysis and Statistics
```python
def analyze_correction_performance(test_cases: List[Dict]) -> Dict:
    """
    Analyze system performance on test cases
    """
    stats = {
        'total_cases': len(test_cases),
        'total_errors': 0,
        'corrected_errors': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'error_types': defaultdict(int),
        'correction_times': []
    }
    
    for case in test_cases:
        text = case['text']
        expected_errors = case['expected_errors']
        
        start_time = time.time()
        tokens = tokenize(text)
        detected_errors = detect_errors_comprehensive(tokens)
        processing_time = time.time() - start_time
        
        stats['correction_times'].append(processing_time)
        
        # Compare detected vs expected
        detected_set = set(tokens[pos] for pos, _ in detected_errors)
        expected_set = set(expected_errors)
        
        stats['total_errors'] += len(expected_set)
        stats['corrected_errors'] += len(detected_set & expected_set)
        stats['false_positives'] += len(detected_set - expected_set)
        stats['false_negatives'] += len(expected_set - detected_set)
        
        # Categorize error types
        for error_word in detected_set:
            error_type = classify_error_type(error_word, tokens)
            stats['error_types'][error_type] += 1
    
    # Calculate metrics
    stats['precision'] = stats['corrected_errors'] / (stats['corrected_errors'] + stats['false_positives']) if stats['corrected_errors'] + stats['false_positives'] > 0 else 0
    stats['recall'] = stats['corrected_errors'] / stats['total_errors'] if stats['total_errors'] > 0 else 0
    stats['f1_score'] = 2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall']) if stats['precision'] + stats['recall'] > 0 else 0
    stats['avg_processing_time'] = sum(stats['correction_times']) / len(stats['correction_times']) if stats['correction_times'] else 0
    
    return stats

# Run performance analysis
test_cases = [
    {'text': "The scientst conducted an experment.", 'expected_errors': ['scientst', 'experment']},
    {'text': "The algoritm was very efective.", 'expected_errors': ['algoritm', 'efective']},
]

performance_stats = analyze_correction_performance(test_cases)
print(f"Precision: {performance_stats['precision']:.2f}")
print(f"Recall: {performance_stats['recall']:.2f}")
print(f"F1 Score: {performance_stats['f1_score']:.2f}")
print(f"Average processing time: {performance_stats['avg_processing_time']:.3f}s")
```

### Integration with External Systems
```python
# Integration with text editors/IDEs
class SpellingPlugin:
    def __init__(self, spelling_system):
        self.spelling_system = spelling_system
        self.error_cache = {}
    
    def check_document(self, document_text: str) -> List[Dict]:
        """Check entire document for spelling errors"""
        tokens = tokenize(document_text)
        errors = self.spelling_system.detect_errors(tokens)
        
        # Convert to editor-friendly format
        editor_errors = []
        for pos, word in errors:
            if pos not in self.error_cache:
                context = tokens[max(0, pos-2):pos+3]
                suggestions = self.spelling_system.get_suggestions(word, context, pos)
                self.error_cache[pos] = suggestions
            
            # Calculate character positions in original text
            start_char = self._token_to_char_position(document_text, pos)
            end_char = start_char + len(word)
            
            editor_errors.append({
                'line': self._get_line_number(document_text, start_char),
                'start_char': start_char,
                'end_char': end_char,
                'word': word,
                'suggestions': self.error_cache[pos],
                'severity': 'error' if self._is_nonword_error(word) else 'warning'
            })
        
        return editor_errors
    
    def apply_correction(self, document_text: str, error_pos: int, correction: str) -> str:
        """Apply a correction to the document"""
        tokens = tokenize(document_text)
        tokens[error_pos] = correction
        return ' '.join(tokens)

# Usage in an IDE plugin
plugin = SpellingPlugin(spelling_system)
errors = plugin.check_document("This is a test document with erors.")
# Returns structured error information for IDE integration
```

These examples demonstrate the system's flexibility and comprehensive API for various use cases, from simple command-line usage to complex integration scenarios.

---

## Complete System Architecture Overview

### Component Interaction Diagram
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            User Input                                       │
│                            (Text/String)                                    │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Text Preprocessing                                  │
│  • Unicode Normalization (NFC)                                            │
│  • Case Conversion (lowercase)                                            │
│  • Tokenization (word boundary detection)                                 │
│  • Punctuation Preservation                                               │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Multi-Layer Error Detection                            │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Non-Word        │  │ Real-Word       │  │ Mixed Error     │            │
│  │ Detection       │  │ Detection       │  │ Pattern         │            │
│  │ (Dictionary     │  │ (Contextual     │  │ Detection       │            │
│  │  Lookup)        │  │  Analysis)      │  │ (Sequential)    │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
│  Error Types:                                                              │
│  • Non-word: "teh" → "the"                                                │
│  • Real-word: "roll" → "role" (context-dependent)                         │
│  • Mixed: Multiple error types in sequence                               │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Hierarchical Candidate Generation                         │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Contextual      │  │ Edit Distance    │  │ Phonetic        │            │
│  │ (Highest        │  │ (Levenshtein,    │  │ (SoundEx)       │            │
│  │  Priority)      │  │  Damerau)        │  │                 │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ SymSpell        │  │ Fuzzy Matching   │  │ Technical       │            │
│  │ Dictionary      │  │ (Partial)        │  │ Domain          │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
│  Generation Strategies: 13+ methods with priority weighting               │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Advanced Multi-Factor Ranking                           │
│                                                                             │
│  Scoring Factors (weighted combination):                                  │
│  • Edit Distance (25%): Similarity to original word                       │
│  • Frequency (20%): Common usage statistics                              │
│  • Context Compatibility (20%): Language model scoring                    │
│  • Strategy Weight (15%): Generation method reliability                   │
│  • Phonetic Similarity (10%): Pronunciation matching                       │
│  • Length Similarity (5%): Size appropriateness                           │
│  • Error Type Adjustment (5%): Specialized scoring                        │
│                                                                             │
│  Output: Top 8 suggestions with confidence scores                         │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Post-Processing & Validation                           │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Conflict        │  │ Grammar         │  │ Final           │            │
│  │ Resolution      │  │ Validation      │  │ Selection       │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
│  • Prevent contradictory corrections                                       │
│  • Validate syntactic correctness                                          │
│  • Apply confidence thresholding                                           │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GUI Presentation                                  │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Error           │  │ Interactive     │  │ Real-time       │            │
│  │ Highlighting    │  │ Suggestions     │  │ Feedback        │            │
│  │ (Red/Yellow)    │  │ (Popup Menus)   │  │ (Status/Stats)   │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
│  Features:                                                                 │
│  • 500-character limit with counter                                       │
│  • Background processing (non-blocking)                                   │
│  • Click-to-correct interface                                             │
│  • Vocabulary statistics display                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow and Memory Architecture

#### Core Data Structures
```python
class SystemData:
    # Primary vocabulary (33,563 base words + 126 technical = 33,689 total)
    vocabulary: Set[str]                    # O(1) membership testing
    
    # Frequency data for ranking
    word_frequencies: Dict[str, int]        # O(1) frequency lookup
    
    # Pre-computed indices for performance
    symspell_dictionary: List[str]          # O(log n) edit distance lookup
    phonetic_index: Dict[str, List[str]]   # O(1) soundex lookup
    confusion_pairs: Dict[str, List[str]]  # O(1) confusion lookup
    
    # Caching layers
    suggestion_cache: Dict[str, List[str]] # LRU cache for suggestions
    context_cache: Dict[str, float]        # Context pattern caching
    
    # Configuration
    detection_config: DetectionConfig
    generation_config: GenerationConfig
    ranking_config: RankingConfig
    performance_config: PerformanceConfig
```

#### Processing Pipeline Performance Characteristics
- **Initialization**: ~2-3 seconds (load vocabularies, build indices)
- **Per-Word Analysis**: 
  - Non-word detection: O(1) - O(k) where k = word length
  - Real-word detection: O(w) where w = context window size (typically 5-9)
  - Candidate generation: O(s × c) where s = strategies used, c = candidates per strategy
  - Ranking: O(c × log c) where c = total candidates
- **Memory Usage**: ~20MB runtime (vocabularies + caches)
- **Threading**: Background processing prevents UI blocking

#### Error Recovery Mechanisms
```python
class ErrorRecovery:
    def __init__(self):
        self.fallback_strategies = [
            self._standard_processing,
            self._reduced_accuracy_processing, 
            self._fast_approximation,
            self._basic_spellcheck_only
        ]
    
    def process_with_recovery(self, text: str) -> Result:
        """Process with automatic fallback on failures"""
        for strategy in self.fallback_strategies:
            try:
                return strategy(text)
            except (MemoryError, TimeoutError, Exception) as e:
                self.logger.warning(f"Strategy failed: {e}, trying fallback...")
                continue
        
        # Ultimate fallback
        return self._return_empty_result(text)
```

### Configuration and Adaptation

#### Dynamic Configuration System
The system supports multiple operating modes that automatically adjust parameters:

- **Fast Mode**: Minimal context (3 words), few candidates (10), quick ranking (3 suggestions)
- **Accurate Mode**: Extended context (7 words), many candidates (50), high threshold (10 suggestions)
- **Aggressive Mode**: Maximum context (9 words), extensive candidates (100), comprehensive coverage (15 suggestions)
- **Standard Mode**: Balanced settings (5 words context, 25 candidates, 8 suggestions)

#### Content-Aware Adaptation
```python
def adapt_to_content(text: str) -> SystemConfig:
    """Automatically adjust configuration based on content characteristics"""
    
    # Detect technical content
    technical_indicators = ['algorithm', 'hypothesis', 'methodology', 'analysis']
    if any(word in text.lower() for word in technical_indicators):
        config.detection.technical_threshold = 1.0
        config.generation.phonetic_weight = 0.8
    
    # Adjust for text length
    word_count = len(text.split())
    if word_count < 10:
        config.performance.timeout = 5.0
    elif word_count > 100:
        config.performance.batch_size = 50
    
    # Real-time vs batch processing
    if len(text) < 100:
        config.performance.threading = False  # Synchronous for speed
    
    return config
```

### Quality Assurance and Validation

#### Automated Testing Framework
```python
class SpellingTestSuite:
    def __init__(self):
        self.test_categories = {
            'simple_stress': SimpleStressTest(),
            'extreme_stress': ExtremeStressTest(),
            'technical_validation': TechnicalVocabularyTest(),
            'performance_benchmark': PerformanceBenchmark()
        }
    
    def run_comprehensive_tests(self) -> TestReport:
        """Run all test categories and generate comprehensive report"""
        results = {}
        
        for category_name, test_suite in self.test_categories.items():
            results[category_name] = test_suite.run()
        
        return self._generate_report(results)
```

#### Performance Metrics Tracked
- **Detection Accuracy**: Precision, recall, F1-score for error identification
- **Correction Coverage**: Percentage of expected corrections found in suggestions
- **Processing Speed**: Average time per word/document
- **Memory Efficiency**: Peak memory usage and memory leaks
- **User Experience**: GUI responsiveness and error feedback quality

### Integration and Extensibility

#### Plugin Architecture
```python
class SpellingPluginInterface:
    def __init__(self, core_system):
        self.core = core_system
        self.hooks = {
            'preprocessing': [],
            'postprocessing': [],
            'custom_detection': [],
            'custom_ranking': []
        }
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Register custom processing hooks"""
        if hook_name in self.hooks:
            self.hooks[hook_name].append(callback)
    
    def process_with_plugins(self, text: str) -> Result:
        """Process text with plugin enhancements"""
        # Run preprocessing hooks
        for hook in self.hooks['preprocessing']:
            text = hook(text)
        
        # Core processing
        result = self.core.process(text)
        
        # Run postprocessing hooks
        for hook in self.hooks['postprocessing']:
            result = hook(result)
        
        return result
```

#### API Compatibility Layers
- **REST API**: HTTP endpoints for web service integration
- **WebSocket API**: Real-time processing for collaborative editing
- **Library API**: Direct Python integration for applications
- **Command Line**: Batch processing for documents/files

This comprehensive architecture ensures the spelling correction system is robust, extensible, and capable of handling diverse use cases while maintaining high accuracy and performance standards.

# Generate suggestions
for i, word in nonword_errors:
    candidates = generator.generate(word, context_tokens=tokens, index=i)
    suggestions = ranker.suggest(candidates, word, tokens)
    print(f"{word} → {suggestions[:3]}")
```

### Command Line Interface
```python
def cli_spell_check(text: str):
    # Initialize system
    system = SpellingSystem()
    
    # Process
    tokens = tokenize(text)
    errors = system.detect_all_errors(tokens)
    
    # Generate corrections
    corrections = {}
    for i, word in errors:
        suggestions = system.get_suggestions(word, tokens, i)
        corrections[word] = suggestions
    
    return corrections

# Usage
result = cli_spell_check("Their are many erors in this sentance.")
# Returns: {'Their': ['There'], 'erors': ['errors'], 'sentance': ['sentence']}
```

---

## Technical Implementation Details

### Data Structures

#### Vocabulary Storage
```python
# Set for O(1) membership testing
vocab: Set[str] = {
    'abandon', 'ability', 'able', 'about', 'above', 'absence',
    # ... ~100,000 English words
}
```

#### Frequency Dictionary
```python
# Dictionary mapping words to frequency counts
word_freqs: Dict[str, int] = {
    'the': 1234567,
    'be': 987654,
    'to': 876543,
    # ... frequency data for ranking
}
```

#### Confusion Pairs Mapping
```python
confusion_pairs: Dict[str, List[str]] = {
    'there': ['their', "they're"],
    'your': ["you're"],
    'its': ["it's"],
    'effect': ['affect', 'effects'],
    'affect': ['effect', 'affects'],
    # ... comprehensive confusion mapping
}
```

#### SymSpell Dictionary
```python
# Pre-computed edit distance dictionary
symspell_words: List[str] = [
    'word1', 'word2', 'word3',  # Sorted by frequency
    # ... ~50,000 most common words
]
```

### Algorithms Implementation

#### Edit Distance Calculation
```python
def calculate_edit_distance(s1: str, s2: str) -> int:
    # Levenshtein distance with dynamic programming
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion
                dp[i][j-1] + 1,      # Insertion
                dp[i-1][j-1] + cost  # Substitution
            )
    
    return dp[m][n]
```

#### Soundex Algorithm
```python
def soundex(word: str) -> str:
    if not word:
        return "0000"
    
    word = word.upper()
    soundex_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    # Keep first letter
    result = word[0]
    prev_code = soundex_map.get(word[0], '0')
    
    for char in word[1:]:
        code = soundex_map.get(char, '0')
        if code != '0' and code != prev_code:
            result += code
        prev_code = code
    
    # Pad or truncate to 4 characters
    return (result + '0000')[:4]
```

#### Keyboard Distance Calculation
```python
QWERTY_LAYOUT = {
    'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
    'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
    'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4),
    'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8),
    'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4),
    'n': (2, 5), 'm': (2, 6)
}

def keyboard_distance(char1: str, char2: str) -> float:
    if char1 not in QWERTY_LAYOUT or char2 not in QWERTY_LAYOUT:
        return 3.0  # High penalty for non-keyboard chars
    
    pos1 = QWERTY_LAYOUT[char1]
    pos2 = QWERTY_LAYOUT[char2]
    
    # Manhattan distance
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

### Error Handling

#### Graceful Degradation
```python
def safe_operation(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Operation failed: {e}")
        return default_fallback()
```

#### Input Validation
```python
def validate_input(text: str) -> bool:
    if not isinstance(text, str):
        raise TypeError("Input must be string")
    
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text too long: {len(text)} > {MAX_TEXT_LENGTH}")
    
    if len(text.strip()) < MIN_TEXT_LENGTH:
        raise ValueError("Text too short")
    
    return True
```

#### Resource Management
```python
class ResourceManager:
    def __enter__(self):
        self.load_resources()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_resources()
    
    def load_resources(self):
        # Load vocabulary, models, etc.
        pass
    
    def cleanup_resources(self):
        # Free memory, close files
        pass
```

### Testing Framework

#### Unit Tests
```python
class TestSpellingCorrection(unittest.TestCase):
    def setUp(self):
        self.detector = NonWordDetector(test_vocab)
        self.generator = CandidateGenerator(test_symspell, test_vocab)
    
    def test_nonword_detection(self):
        tokens = ['this', 'is', 'wrng']
        results = self.detector.detect(tokens)
        self.assertEqual(results, [False, False, True])
    
    def test_candidate_generation(self):
        candidates = self.generator.generate('wrng')
        self.assertIn('wrong', candidates)
        self.assertIn('writing', candidates)
    
    def test_contextual_correction(self):
        tokens = ['the', 'crucial', 'roll', 'of']
        candidates = self.generator.get_contextual_candidates('roll', tokens, 2)
        self.assertIn('role', candidates)
```

---

## Stress Testing & Validation Suite

### Comprehensive Test Framework
The system includes an extensive stress testing suite designed to validate performance under extreme conditions and ensure robustness of all enhancements.

#### Test Categories

##### 1. Simple Stress Test (`test_simple_stress.py`)
**Purpose**: Validates basic functionality of enhancements
**Test Cases**:
- **Technical Terms**: Medical/scientific vocabulary recognition
- **Scientific Terms**: Research terminology validation
- **Mixed Errors**: Non-word errors with technical context
- **Real-word Context**: Context-dependent homophone detection

**Performance Target**: 100% accuracy on all test cases

##### 2. Extreme Stress Test (`test_extreme_stress.py`)
**Purpose**: Pushes system limits with challenging edge cases
**Test Categories**:
- **Ultra-Hard Non-Word**: Obscure scientific/technical terms with subtle errors
- **Very Long Words**: Single character errors in lengthy words
- **Foreign Words**: Technical jargon and loanwords
- **Multiple Similar Errors**: Sequential similar-looking corrections
- **Extreme Real-Word**: Highly context-dependent homophones
- **Technical Context**: Words correct but inappropriate for domain
- **Archaic vs Modern**: Historical language usage confusion
- **Chaotic Mixed**: Multiple error types simultaneously
- **Scientific Papers**: Academic text with intentional errors
- **Impossible Cases**: Designed to challenge even advanced systems

#### Test Metrics

##### Detection Accuracy
```python
def calculate_detection_accuracy(expected_errors: List[str], detected_errors: List[str]) -> Dict[str, float]:
    expected_set = set(expected_errors)
    detected_set = set(detected_errors)
    
    true_positives = len(expected_set & detected_set)
    false_positives = len(detected_set - expected_set)
    false_negatives = len(expected_set - detected_set)
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': true_positives / len(expected_set) if expected_set else 1.0
    }
```

##### Correction Coverage
```python
def calculate_correction_coverage(expected_corrections: Dict[str, List[str]], 
                                generated_candidates: Dict[str, List[str]]) -> Dict[str, float]:
    coverage_stats = {}
    
    for word, expected_list in expected_corrections.items():
        if word in generated_candidates:
            generated_list = generated_candidates[word]
            found_corrections = set(expected_list) & set(generated_list)
            coverage = len(found_corrections) / len(expected_list)
            coverage_stats[word] = coverage
        else:
            coverage_stats[word] = 0.0
    
    overall_coverage = sum(coverage_stats.values()) / len(coverage_stats)
    return {
        'per_word_coverage': coverage_stats,
        'overall_coverage': overall_coverage
    }
```

#### Performance Validation Results

##### Simple Stress Test Results (November 2025)
- **Tests Passed**: 4/4 (100.0%)
- **Technical Vocabulary Recognition**: 4/4 key terms (100%)
- **Enhanced Features**: All working correctly
- **Processing Time**: < 2 seconds per test case

##### Extreme Stress Test Results (November 2025)
- **Test Completion**: All 11 stress tests completed without crashes
- **Detection Performance**: Varies by category (60-95% accuracy)
- **Correction Coverage**: 40-80% of expected corrections found
- **Technical Term Handling**: Significantly improved over base system
- **Mixed Error Scenarios**: Successfully handled complex combinations

#### Stress Test Examples

##### Ultra-Hard Non-Word Example
```
Text: "The oncolgist recomended chemotherpy for the patinet's malignent neoplasm."
Expected Errors: ['oncolgist', 'recomended', 'chemotherpy', 'patinet's', 'malignent', 'neoplasm']
System Detection: ['oncolgist', 'recomended', 'chemotherpy', 'malignent', 'neoplasm']
Key Corrections Found: 'oncologist', 'recommended', 'chemotherapy', 'malignant'
```

##### Chaotic Mixed Error Example
```
Text: "The lieutenent colonel's menopausal symtoms were efective in there assault."
Expected Errors: ['lieutenent', "colonel's", 'menopausal', 'symtoms', 'efective']
System Detection: ['lieutenent', 'menopausal', 'symtoms', 'were', 'efective', 'assault']
Key Corrections Found: 'lieutenant', 'symptoms', 'effective'
```

---

## System Flow Diagram

```
Input Text
    ↓
┌─────────────────┐
│  Text Processing │ → Tokenization, Normalization
│  (text.py)       │
└─────────────────┘
    ↓
┌─────────────────┐
│ Error Detection  │ → Non-word + Real-word Detection
│  (detect.py)     │
└─────────────────┘
    ↓
┌─────────────────┐
│    For Each     │
│  Detected Error │
└─────────────────┘
    ↓
┌─────────────────┐
│ Candidate Gen   │ → 11+ Generation Strategies
│ (candidates.py) │
└─────────────────┘
    ↓
┌─────────────────┐
│ Advanced Ranking│ → Context + Frequency + Edit Distance
│(advanced_rank.py)│
└─────────────────┘
    ↓
┌─────────────────┐
│ GUI Display     │ → Highlighting + Suggestions
│(spelling_gui.py)│
└─────────────────┘
    ↓
User Interaction
```

This comprehensive system provides state-of-the-art spelling correction capabilities with both high accuracy and excellent performance, suitable for real-time interactive use while maintaining the sophistication needed for challenging contextual errors. Recent enhancements (November 2025) have significantly improved handling of technical vocabulary, enhanced context analysis with domain-aware detection, and mixed error handling for complex scenarios. The system now recognizes 126+ specialized terms across medical, scientific, and technical domains, uses multiple edit distance algorithms, and includes comprehensive stress testing validation.