# spelling/src/text.py
import re  # Use standard re module instead of regex for better compatibility
TOK = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|[^\s]")

def normalize(s: str) -> str:
    return (s or "").strip().lower()

def tokenize(s: str):
    return TOK.findall(s or "")