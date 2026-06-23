# main.py - Main CLI entry point for candidate ranker pipeline
# Version tracked in git repository.
import json
import zipfile
import csv
import gzip
import argparse
from ranker import rank_candidates
from reasoning import generate_reasoning

def clean_data(obj):
    """Recursively strips whitespace from all string keys and values.
    This prevents KeyError crashes from poorly formatted dataset keys like 'candidate_id '."""
    if isinstance(obj, dict):
        return {str(k).strip(): clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    elif isinstance(obj, str):
        return obj.strip()
    return obj

def load_candidates(filepath: str) -> list[dict]:
    """Loads candidates from .json, .jsonl, .jsonl.gz, or .zip"""
    content = ""
    
    # --- NEW: Handle .zip files ---
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Find the first .jsonl file inside the zip
            jsonl_files = [f for f in zf.namelist() if f.endswith('.jsonl')]
            if not jsonl_files:
                raise FileNotFoundError("No .jsonl file found inside the zip.")
            with zf.open(jsonl_files[0], 'r') as f:
                content = f.read().decode('utf-8')
    # --- END NEW ---
    
    elif filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            content = f.read().strip()
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    
    if not content:
        return []
        
    if content.startswith('['):
        raw_data = json.loads(content)
    else:
        raw_data = [json.loads(line) for line in content.split('\n') if line.strip()]
        
    return [clean_data(c) for c in raw_data]

def main():
    parser = argparse.ArgumentParser(description="Redrob AI Recruiter Ranker")
    parser.add_argument("--candidates", type=str, default="data/candidates.zip", help="Path to candidates file (.jsonl, .jsonl.gz, or .zip)")
    parser.add_argument("--out", type=str, default="Team_recruiters_submission.csv", help="Output CSV path")
    parser.add_argument("--top_n", type=int, default=100, help="Number of candidates to rank")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.")

    print("Scoring and ranking candidates...")
    top_candidates = rank_candidates(candidates, top_n=args.top_n)
    print(f"Ranked top {len(top_candidates)} candidates.")

    print("Generating diversified reasoning...")
    results = []
    last_template_type = ""
    
    for rank, scored_c in enumerate(top_candidates, start=1):
        reasoning, last_template_type = generate_reasoning(scored_c, last_template_type)
        
        results.append({
            "candidate_id": scored_c["candidate_id"],
            "rank": rank,
            "score": scored_c["score"],
            "reasoning": reasoning
        })

    print(f"Writing results to {args.out}...")
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)

    print("Submission generated successfully!")
    print(f"   - Total rows: {len(results)}")
    print(f"   - Top score: {results[0]['score']}")
    print(f"   - {args.top_n}th score: {results[-1]['score']}")

if __name__ == "__main__":
    main()