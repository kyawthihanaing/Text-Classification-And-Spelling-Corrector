# spelling/src/advanced_rank.py
import math
from rapidfuzz.distance import Levenshtein, DamerauLevenshtein
from rapidfuzz import fuzz
from .lm import load_lm

class AdvancedRanker:
    """Advanced ensemble ranking system for spelling correction candidates"""
    
    def __init__(self, word_freqs, lm=None):
        self.word_freqs = word_freqs
        self.lm = lm or load_lm()
        
        # Precompute log frequencies for performance
        self.log_freqs = {}
        for word, freq in word_freqs.items():
            self.log_freqs[word] = math.log(freq + 1)
        
        # Ensemble weights (default for normal cases)
        self.weights = {
            'frequency': 0.35,
            'edit_distance': 0.25,
            'fuzzy_similarity': 0.20,
            'language_model': 0.15,
            'bonus': 0.05
        }
        
        # Alternative weights for synthetic/difficult cases
        self.synthetic_weights = {
            'frequency': 0.20,      # Reduce frequency bias
            'edit_distance': 0.40,  # Increase edit distance importance
            'fuzzy_similarity': 0.25, # Increase fuzzy matching
            'language_model': 0.10, # Reduce LM dependency
            'bonus': 0.05
        }
        
        # Ultra-aggressive weights for very difficult synthetic cases
        self.ultra_weights = {
            'frequency': 0.10,      # Minimal frequency bias
            'edit_distance': 0.50,  # Maximum edit distance importance
            'fuzzy_similarity': 0.30, # High fuzzy matching
            'language_model': 0.05, # Minimal LM dependency
            'bonus': 0.05
        }
    
    def rank(self, candidates, original, sent_tokens, token_index, synthetic_mode=False, ultra_mode=False):
        """Rank candidates using ensemble of multiple scoring methods"""
        if not candidates:
            return []
        
        # Choose weights based on mode
        if ultra_mode:
            weights = self.ultra_weights
        elif synthetic_mode:
            weights = self.synthetic_weights
        else:
            weights = self.weights
        
        scored_candidates = []
        
        for candidate in candidates:
            # Compute individual scores
            freq_score = self._frequency_score(candidate)
            edit_score = self._edit_distance_score(candidate, original)
            fuzzy_score = self._fuzzy_similarity_score(candidate, original)
            lm_score = self._language_model_score(candidate, original, sent_tokens, token_index)
            bonus_score = self._bonus_score(candidate, original, freq_score)
            
            # Ensemble combination with selected weights
            total_score = (
                weights['frequency'] * freq_score +
                weights['edit_distance'] * edit_score +
                weights['fuzzy_similarity'] * fuzzy_score +
                weights['language_model'] * lm_score +
                weights['bonus'] * bonus_score
            )
            
            scored_candidates.append((candidate, total_score))
        
        # Sort by total score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, score in scored_candidates]
    
    def _frequency_score(self, candidate):
        """Score based on word frequency in corpus"""
        if candidate in self.log_freqs:
            # Normalize log frequency to 0-10 scale
            return min(self.log_freqs[candidate] * 1.2, 10.0)
        return 0.5  # Small score for unknown words
    
    def _edit_distance_score(self, candidate, original):
        """Score based on edit distance (closer = higher score)"""
        lev_dist = Levenshtein.distance(original, candidate)
        dam_dist = DamerauLevenshtein.distance(original, candidate)
        min_dist = min(lev_dist, dam_dist)
        
        # Convert distance to similarity score (0-10 scale)
        max_len = max(len(original), len(candidate))
        if max_len == 0:
            return 10.0
        
        similarity = 1.0 - (min_dist / max_len)
        return similarity * 10.0
    
    def _fuzzy_similarity_score(self, candidate, original):
        """Score based on fuzzy string matching"""
        ratio = fuzz.ratio(original, candidate)
        token_ratio = fuzz.token_sort_ratio(original, candidate)
        partial_ratio = fuzz.partial_ratio(original, candidate)
        
        # Use best of the three ratios
        best_ratio = max(ratio, token_ratio, partial_ratio)
        return best_ratio / 10.0  # Scale to 0-10
    
    def _language_model_score(self, candidate, original, sent_tokens, token_index):
        """Score based on language model context"""
        if not self.lm or len(sent_tokens) <= 1:
            return 5.0  # Neutral score
        
        try:
            # Original sentence probability
            original_prob = self.lm.sent_logprob(sent_tokens)
            
            # New sentence with candidate
            new_tokens = sent_tokens[:]
            new_tokens[token_index] = candidate
            new_prob = self.lm.sent_logprob(new_tokens)
            
            # Score improvement (clamped to reasonable range)
            improvement = new_prob - original_prob
            return max(0, min(10.0, 5.0 + improvement * 2.0))
            
        except Exception:
            return 5.0  # Neutral score on error
    
    def _bonus_score(self, candidate, original, freq_score):
        """Additional bonuses for special patterns"""
        bonus = 0.0
        
        # Length match bonus
        if len(original) == len(candidate):
            bonus += 2.0
        
        # Single character error bonus
        lev_dist = Levenshtein.distance(original, candidate)
        if lev_dist == 1:
            bonus += 3.0
        elif lev_dist == 2:
            bonus += 1.5
        
        # High frequency word bonus
        if freq_score > 8.0:
            bonus += 2.0
        elif freq_score > 6.0:
            bonus += 1.0
        
        # Common prefix/suffix bonus
        if len(original) > 3 and len(candidate) > 3:
            # Check prefix
            common_prefix = 0
            for i in range(min(len(original), len(candidate))):
                if original[i] == candidate[i]:
                    common_prefix += 1
                else:
                    break
            
            # Check suffix
            common_suffix = 0
            for i in range(1, min(len(original), len(candidate)) + 1):
                if original[-i] == candidate[-i]:
                    common_suffix += 1
                else:
                    break
            
            if common_prefix >= 3 or common_suffix >= 3:
                bonus += 1.5
        
        # Character composition similarity
        original_chars = set(original.lower())
        candidate_chars = set(candidate.lower())
        char_overlap = len(original_chars & candidate_chars)
        char_total = len(original_chars | candidate_chars)
        
        if char_total > 0:
            char_similarity = char_overlap / char_total
            if char_similarity > 0.8:
                bonus += 1.0
        
        # Transposition bonus (common typo pattern)
        if self._is_transposition(original, candidate):
            bonus += 2.5
        
        return min(bonus, 10.0)  # Cap bonus at 10
    
    def _is_transposition(self, s1, s2):
        """Check if s2 is a simple transposition of s1"""
        if len(s1) != len(s2):
            return False
        
        diff_positions = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diff_positions.append(i)
        
        # Simple transposition: exactly 2 differences, adjacent positions
        if len(diff_positions) == 2:
            pos1, pos2 = diff_positions
            if abs(pos1 - pos2) == 1:
                return s1[pos1] == s2[pos2] and s1[pos2] == s2[pos1]
        
        return False
    
    def suggest(self, candidates, original, sent_tokens, token_index, top_k=5, synthetic_mode=False, ultra_mode=False):
        """Get top-k ranked suggestions"""
        ranked = self.rank(candidates, original, sent_tokens, token_index, synthetic_mode, ultra_mode)
        return ranked[:top_k]