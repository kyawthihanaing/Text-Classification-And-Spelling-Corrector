from pathlib import Path
import regex as re

path = Path("spelling/data/raw/arxiv_abstracts.txt")
tok = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?")
n = 0
for line in path.open(encoding="utf-8"):
    n += len(tok.findall(line))
print("Tokens:", n)