# main.py - Main CLI entry point for V3 candidate ranker pipeline
import json, zipfile, csv, gzip, argparse
from pathlib import Path
from ranker import rank_candidates
from reasoning import generate_reasoning

def clean_data(obj):
    if isinstance(obj, dict):   return {str(k).strip(): clean_data(v) for k,v in obj.items()}
    elif isinstance(obj, list): return [clean_data(i) for i in obj]
    elif isinstance(obj, str):  return obj.strip()
    return obj

def load_candidates(filepath):
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath,'r') as zf:
            jsonl_files = [f for f in zf.namelist() if f.endswith('.jsonl')]
            if not jsonl_files: raise FileNotFoundError("No .jsonl file found in zip.")
            with zf.open(jsonl_files[0],'r') as f:
                content = f.read().decode('utf-8')
    elif filepath.endswith('.gz'):
        with gzip.open(filepath,'rt',encoding='utf-8') as f: content=f.read().strip()
    else:
        with open(filepath,'r',encoding='utf-8') as f: content=f.read().strip()
    if not content: return []
    raw = json.loads(content) if content.startswith('[') else \
          [json.loads(line) for line in content.split('\n') if line.strip()]
    return [clean_data(c) for c in raw]

def main():
    parser = argparse.ArgumentParser(description="Redrob AI Recruiter Ranker — V3")
    parser.add_argument("--candidates", default="data1/candidates.zip")
    parser.add_argument("--out", default="Team_recruiters.csv")
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates.")

    print("Scoring and ranking...")
    top_candidates = rank_candidates(candidates, top_n=args.top_n)
    print(f"Ranked top {len(top_candidates)} candidates.")

    rows = []
    for rank, sc in enumerate(top_candidates, start=1):
        score = round(max(0.10, min(0.99, sc["score"])), 4)
        rows.append({"candidate_id":sc["candidate_id"],"rank":rank,
                     "score":score,"reasoning":generate_reasoning(sc)})

    # Enforce monotonic + tie-break by candidate_id ascending
    for i in range(len(rows)-1):
        if rows[i]["score"] < rows[i+1]["score"]:
            rows[i+1]["score"] = rows[i]["score"]
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    for i,row in enumerate(rows): row["rank"] = i+1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id","rank","score","reasoning"])
        w.writeheader(); w.writerows(rows)

    print(f"\nWritten to {args.out}")
    print(f"  Rows: {len(rows)}  Top: {rows[0]['score']}  100th: {rows[-1]['score']}")

if __name__ == "__main__":
    main()
