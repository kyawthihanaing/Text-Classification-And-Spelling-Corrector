# spelling/src/text.py
import regex as re
TOK = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|[^\s]")

def normalize(s: str) -> str:
    return (s or "").strip().lower()

def tokenize(s: str):
    return TOK.findall(s or "")