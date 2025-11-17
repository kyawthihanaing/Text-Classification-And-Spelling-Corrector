# spelling/src/lm.py
import math
from collections import Counter, defaultdict
from .assets import load_trigrams
import json
from pathlib import Path

PROC = Path("spelling/data/processed")

class TrigramLM:
    """
    Trigram language model using the actual trigram data format.
    Data structure: {"w1 w2": {"w3": count, ...}}
    """
    def __init__(self, trigram_data: dict, discount=0.5, floor=1e-9):
        self.floor = floor
        self.d = discount
        self.trigram_data = trigram_data

        # Build bigram and unigram counts
        self.bi = Counter()
        self.uni = Counter()

        for context, words in trigram_data.items():
            w1, w2 = context.split(" ", 1)
            total_context_count = sum(words.values())
            self.bi[(w1, w2)] = total_context_count

            for w3, count in words.items():
                self.uni[w1] += count
                self.uni[w2] += count
                self.uni[w3] += count

        self.V = len(self.uni) or 1
        self.total_uni = sum(self.uni.values()) or 1

        # Enhanced: Load additional n-gram models for better context analysis
        self._load_enhanced_models()

    def _load_enhanced_models(self):
        """Load additional n-gram models for enhanced analysis"""
        try:
            # Load bigrams for better bigram analysis
            bigram_path = PROC / "bigrams.json"
            if bigram_path.exists():
                with open(bigram_path, 'r', encoding='utf-8') as f:
                    self.bigram_data = json.load(f)
            else:
                self.bigram_data = {}

            # Load fourgrams if available
            fourgram_path = PROC / "fourgrams.json"
            if fourgram_path.exists():
                with open(fourgram_path, 'r', encoding='utf-8') as f:
                    self.fourgram_data = json.load(f)
            else:
                self.fourgram_data = {}

            print(f"✅ Enhanced LM loaded: {len(self.bigram_data)} bigrams, {len(self.fourgram_data)} fourgrams")
        except Exception as e:
            print(f"⚠️  Could not load enhanced models: {e}")
            self.bigram_data = {}
            self.fourgram_data = {}

    def _p_uni(self, w):
        return max(self.uni[w], 0) / self.total_uni if self.total_uni else self.floor

    def _p_bi(self, w1, w2):
        """Probability of w2 given w1"""
        c12 = self.bi.get((w1, w2), 0)
        c1 = sum(count for (u, v), count in self.bi.items() if u == w1)
        if c12 > 0 and c1 > 0:
            return max((c12 - self.d), 0) / c1
        return self._p_uni(w2)

    def p_trigram(self, w1, w2, w3):
        """Probability of w3 given w1, w2"""
        context = f"{w1} {w2}"
        if context in self.trigram_data:
            c123 = self.trigram_data[context].get(w3, 0)
            c12 = self.bi.get((w1, w2), 0)
            if c123 > 0 and c12 > 0:
                return max((c123 - self.d), 0) / c12

        # Back off to bigram
        return self._p_bi(w2, w3)

    def p_fourgram(self, w1, w2, w3, w4):
        """Enhanced: Probability of w4 given w1, w2, w3"""
        context = f"{w1} {w2} {w3}"
        if context in self.fourgram_data:
            c1234 = self.fourgram_data[context].get(w4, 0)
            c123 = sum(self.fourgram_data[context].values())
            if c1234 > 0 and c123 > 0:
                return max((c1234 - self.d), 0) / c123

        # Back off to trigram
        return self.p_trigram(w2, w3, w4)

    def get_bigram_probability(self, w1, w2, smoothing="laplace"):
        """Enhanced bigram probability with better smoothing"""
        if w1 in self.bigram_data and w2 in self.bigram_data[w1]:
            count = self.bigram_data[w1][w2]
            total = sum(self.bigram_data[w1].values())
            if smoothing == "laplace":
                return (count + 1) / (total + self.V)
            else:
                return count / total if total > 0 else 0.0
        else:
            # Back off to unigram
            return self._p_uni(w2)

    def score_context_window(self, tokens: list, window_size=5):
        """Enhanced: Score a context window using optimal n-gram order"""
        if len(tokens) < 2:
            return 0.0

        tokens = [t.lower() for t in tokens]
        log_prob = 0.0

        # Use fourgrams where possible
        if len(tokens) >= 4:
            for i in range(3, len(tokens)):
                prob = self.p_fourgram(tokens[i-3], tokens[i-2], tokens[i-1], tokens[i])
                log_prob += math.log(prob if prob > 0 else self.floor)

        # Fill gaps with trigrams
        elif len(tokens) >= 3:
            for i in range(2, len(tokens)):
                prob = self.p_trigram(tokens[i-2], tokens[i-1], tokens[i])
                log_prob += math.log(prob if prob > 0 else self.floor)

        # Fallback to bigrams
        else:
            for i in range(1, len(tokens)):
                prob = self.get_bigram_probability(tokens[i-1], tokens[i])
                log_prob += math.log(prob if prob > 0 else self.floor)

        return log_prob

    def compare_candidates(self, original_context: list, candidate: str, position: int):
        """Enhanced: Compare how well different candidates fit in context"""
        if position >= len(original_context):
            return 0.0

        # Create test context with candidate
        test_context = original_context[:position] + [candidate] + original_context[position+1:]

        # Score the context window around the candidate
        window_start = max(0, position - 2)
        window_end = min(len(test_context), position + 3)
        context_window = test_context[window_start:window_end]

        return self.score_context_window(context_window)

    def sent_logprob(self, tokens):
        # add sentence markers
        toks = ["<s>", "<s>"] + tokens + ["</s>"]
        s = 0.0
        for i in range(2, len(toks)):
            p = self.p_trigram(toks[i-2], toks[i-1], toks[i])
            s += math.log(p if p > 0 else self.floor)
        return s

def load_lm():
    trigrams = load_trigrams()
    return TrigramLM(trigrams)