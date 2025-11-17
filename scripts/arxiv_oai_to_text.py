# scripts/arxiv_oai_to_text.py
# Fetch arXiv abstracts via OAI-PMH and save one abstract per line.
# Usage examples (run from repo root):
#   python scripts/arxiv_oai_to_text.py --sets cs physics --max-records 12000 --out spelling/data/raw/arxiv_abstracts.txt
#   python scripts/arxiv_oai_to_text.py --sets cs --max-records 6000

import argparse, time, re, sys, xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
import requests

BASE = "https://export.arxiv.org/oai2"

def fetch(oai_set, max_records=None, sleep_s=3.0, session=None):
    ns = {"oai":"http://www.openarchives.org/OAI/2.0/","dc":"http://purl.org/dc/elements/1.1/"}
    s = session or requests.Session()
    s.headers.update({"User-Agent":"NLP_Assignment/1.0 (contact: youremail@example.com)"})

    params = {"verb":"ListRecords","metadataPrefix":"oai_dc","set":oai_set}
    token = None; n = 0
    while True:
        q = {"verb":"ListRecords","resumptionToken":token} if token else params
        r = s.get(f"{BASE}?{urlencode(q)}", timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)

        for rec in root.findall(".//oai:record", ns):
            descs = [d.text or "" for d in rec.findall(".//dc:description", ns)]
            abstract = re.sub(r"\s+"," "," ".join(descs)).strip()
            if abstract:
                yield abstract
                n += 1
                if max_records and n >= max_records:
                    return

        t = root.find(".//oai:resumptionToken", ns)
        if t is None or not (t.text or "").strip(): break
        token = t.text.strip()
        time.sleep(sleep_s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["cs"], help="arXiv sets like: cs physics math stat")
    ap.add_argument("--out", default="spelling/data/raw/arxiv_abstracts.txt", help="Output text file")
    ap.add_argument("--max-records", type=int, default=12000, help="Cap total abstracts written")
    ap.add_argument("--sleep", type=float, default=3.0, help="Seconds between paginated requests")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent":"NLP_Assignment/1.0 (contact: youremail@example.com)"})

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for oai_set in args.sets:
            print(f"[INFO] Harvesting set='{oai_set}' …", file=sys.stderr)
            for abs_ in fetch(oai_set, args.max_records, args.sleep, s):
                f.write(abs_ + "\n")
                n += 1
    print(f"[DONE] Wrote {n} abstracts → {out}")

if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"[ERROR] HTTP: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr); sys.exit(1)