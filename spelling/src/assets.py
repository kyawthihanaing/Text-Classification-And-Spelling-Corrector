from pathlib import Path
import json
PROC = Path("spelling/data/processed")

def load_symspell_words(path=PROC/"symspell_dict.txt"):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().split()[0]
            if w:
                words.append(w)
    return words

def load_vocab(path=PROC/"vocab_conservative.json"):
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))

def load_word_freq(path=PROC/"word_freq.txt"):
    freq = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                a, b = parts
                if a.isdigit():
                    freq[b] = int(a)
                elif b.isdigit():
                    freq[a] = int(b)
    return freq

def load_trigrams(path=PROC/"trigrams.json"):
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

def load_symspell_words(path=PROC/"symspell_dict.txt"):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().split()[0]
            if w:
                words.append(w)
    return words