# --- preprocessor.py ---
import html as html_module
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Define Dependencies
# Keep negations to preserve sentiment
stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'neither', 'never'}
lemmatizer = WordNetLemmatizer()

# Contractions Map
CONTRACTIONS_MAP = {
    "don't": "do not", "can't": "cannot", "won't": "will not", "i'm": "i am",
    "it's": "it is", "that's": "that is", "there's": "there is", "they're": "they are",
    "we're": "we are", "you're": "you are", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "didn't": "did not", "doesn't": "does not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "shouldn't": "should not",
    "wouldn't": "would not", "couldn't": "could not", "mustn't": "must not", "let's": "let us"
}
pattern = re.compile('(%s)' % '|'.join(map(re.escape, CONTRACTIONS_MAP.keys())), flags=re.IGNORECASE)

def expand_contractions(text):
    def replace(match):
        key = match.group(0).lower()
        return CONTRACTIONS_MAP.get(key, key)
    return pattern.sub(replace, text)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
        
    # --- STAGE 1: AGGRESSIVE CLEANING (Leakage Removal) ---
    # Remove "Reuters" header (e.g. "WASHINGTON (Reuters) -")
    text = re.sub(r"^.*?\(Reuters\) - ", "", text)
    # Remove explicit "Reuters" mentions anywhere
    text = re.sub(r"\bReuters\b", "", text, flags=re.IGNORECASE)
    # Remove Clickbait Artifacts
    artifacts = r"\b(video|watch|image|pic|featured|breaking|hillary|getty|wow|just|via)\b"
    text = re.sub(artifacts, "", text, flags=re.IGNORECASE)

    # --- STAGE 2: NLP NORMALIZATION ---
    # Expand contractions
    text = expand_contractions(text)
    # Unescape HTML
    text = html_module.unescape(text) 
    # Lowercase
    text = text.lower()
    # Remove URLs, Emails, Numbers
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Remove Punctuation
    text = text.translate(str.maketrans('', '', punctuation))
    # Collapse Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess(text):
    # Apply the master cleaning function
    text = clean_text(text)
    tokens = word_tokenize(text)
    # Remove stopwords but keep negations
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1 and t not in stop_words]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmas)

# The Transformer Class for the Pipeline
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [preprocess(text) for text in X]