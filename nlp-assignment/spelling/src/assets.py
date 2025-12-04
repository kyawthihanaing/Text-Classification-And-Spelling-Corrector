from pathlib import Path
import json
import os

# Get the directory where this module is located, then navigate to data/processed
_MODULE_DIR = Path(__file__).parent  # spelling/src/
PROC = _MODULE_DIR.parent / "data" / "processed"  # spelling/data/processed/

def load_symspell_words(path=None):
    if path is None:
        path = PROC / "symspell_dict.txt"
    words = []
    
    if not os.path.exists(path):
        print(f"⚠️ Error: Could not find {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                w = parts[0]  # The word is the first part
                words.append(w)
    return words

def load_vocab(path=None):
    if path is None:
        path = PROC / "vocab_conservative.json"
    
    # Load base vocabulary
    with open(path, "r", encoding="utf-8") as f:
        vocab = set(json.load(f))

    # Merge symspell
    try:
        sym_words = load_symspell_words()
        if sym_words:
            vocab.update(sym_words)
            print(f"✅ Vocabulary loaded: {len(vocab)} words (merged sources)")
        else:
            print("⚠️ Warning: SymSpell dict was empty or not found.")
    except Exception as e:
        print(f"⚠️ Warning: Could not merge symspell dict: {e}")

    return vocab

def load_word_freq(path=None):
    if path is None:
        path = PROC / "word_freq.txt"
    freq = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:  # Check len to avoid errors
                a, b = parts[0], parts[1]
                if a.isdigit():
                    freq[b] = int(a)
                elif b.isdigit():
                    freq[a] = int(b)
    return freq

def load_trigrams(path=None):
    if path is None:
        path = PROC / "trigrams.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_technical_vocabularies():
    """
    Load domain-specific technical vocabularies for scientific/medical terms
    Returns a dictionary of domain -> set of words
    """
    tech_vocab = {}

    # Medical terms
    medical_terms = {
        "oncologist", "chemotherapy", "malignant", "neoplasm", "psychiatrist",
        "schizophrenia", "catatonia", "perseveration", "diagnosis", "symptoms",
        "pathology", "etiology", "prognosis", "therapy", "pharmacology",
        "anatomy", "physiology", "pathophysiology", "histology", "cytology",
        "hematology", "cardiology", "neurology", "psychiatry", "dermatology",
        "endocrinology", "gastroenterology", "nephrology", "pulmonology",
        "rheumatology", "oncology", "radiology", "surgery", "pediatrics",
        "geriatrics", "obstetrics", "gynecology", "urology", "ophthalmology"
    }
    tech_vocab["medical"] = medical_terms

    # Scientific terms
    scientific_terms = {
        "hypothesis", "phylogenetic", "convergent", "divergent", "evolution",
        "molecular", "genetic", "genomic", "proteomic", "transcriptomic",
        "metabolomic", "phenotypic", "genotypic", "heterozygous", "homozygous",
        "allele", "chromosome", "genome", "transcriptome", "proteome",
        "metabolome", "bioinformatics", "computational", "algorithmic",
        "statistical", "probabilistic", "stochastic", "deterministic",
        "heuristic", "optimization", "clustering", "classification",
        "regression", "correlation", "causation", "confounding", "bias",
        "variance", "precision", "accuracy", "sensitivity", "specificity"
    }
    tech_vocab["scientific"] = scientific_terms

    # Technical/CS terms
    technical_terms = {
        "algorithm", "computational", "optimization", "heuristic", "stochastic",
        "probabilistic", "deterministic", "polynomial", "exponential",
        "complexity", "asymptotic", "convergence", "divergence", "iteration",
        "recursion", "dynamic", "programming", "graph", "tree", "network",
        "database", "query", "index", "hash", "encryption", "decryption",
        "authentication", "authorization", "protocol", "interface",
        "abstraction", "encapsulation", "inheritance", "polymorphism",
        "overloading", "overriding", "template", "generic", "container",
        "iterator", "pointer", "reference", "memory", "allocation", "deallocation"
    }
    tech_vocab["technical"] = technical_terms
    
    # Common words that may be missing from base vocabulary
    common_missing = {
        "officer", "police", "detective", "sergeant", "captain", "lieutenant",
        "mayor", "governor", "president", "minister", "chancellor",
        "clam", "calm", "salmon", "shrimp", "lobster", "crab", "oyster",
        "attorney", "lawyer", "judge", "prosecutor", "defendant", "plaintiff",
        "museum", "gallery", "exhibit", "artifact", "sculpture", "painting",
        "cafe", "restaurant", "bistro", "diner", "bakery", "pizzeria",
        "smartphone", "laptop", "tablet", "desktop", "server", "router",
        "website", "webpage", "browser", "download", "upload", "streaming",
        # Words needed for real-word error detection (confusion pairs)
        "sing", "sign", "singing", "signing", "sings", "signs", "sang", "signed",
        "manger", "manager", "managers", "mangers",
        "massage", "message", "massages", "messages", "massaging", "messaging",
        "pubic", "public", "publicly",
        "trail", "trial", "trails", "trials",
        "dairy", "diary", "diaries", "dairies",
        "desert", "dessert", "deserts", "desserts",
        "quite", "quiet", "quietly",
        "lose", "loose", "losing", "loosing", "lost", "losses",
        "breath", "breathe", "breathing", "breaths",
        "advice", "advise", "advising", "advised",
        "principal", "principle", "principals", "principles",
        "stationary", "stationery",
        "complement", "compliment", "compliments", "complements",
        "affect", "effect", "affects", "effects", "affecting", "effecting",
        # Common words that might be missing from vocabulary
        "wagged", "wagging", "wag", "wags",
        "sunny", "rainy", "cloudy", "windy", "snowy", "foggy", "stormy",
        "beautiful", "wonderful", "terrible", "horrible", "amazing", "incredible",
        "quickly", "slowly", "quietly", "loudly", "softly", "gently",
        "arrived", "arriving", "arrives", "arrive",
        "contagious", "infectious", "delicious", "suspicious", "precious",
        "tomorrow", "yesterday", "today",
        "immediate", "immediately", "obvious", "obviously",
        "document", "documents", "documenting", "documented",
        "contract", "contracts", "contracting", "contracted",
        "petition", "petitions", "petitioning", "petitioned",
        "agreement", "agreements"
    }
    tech_vocab["common"] = common_missing

    return tech_vocab

def load_enhanced_vocab(base_vocab, include_technical=True):
    """
    Load enhanced vocabulary with optional technical terms
    """
    enhanced_vocab = base_vocab.copy()

    if include_technical:
        tech_vocabularies = load_technical_vocabularies()
        for domain, terms in tech_vocabularies.items():
            enhanced_vocab.update(terms)

    return enhanced_vocab