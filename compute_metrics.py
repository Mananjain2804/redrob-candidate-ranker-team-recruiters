# compute_metrics.py - Verification metrics helper script (Pearson/Spearman check)
# Version tracked in git repository.
import csv
import math

candidate_ids = ['CAND_0066791','CAND_0053605','CAND_0061257','CAND_0007009','CAND_0073007','CAND_0096104','CAND_0082086','CAND_0071939','CAND_0048558','CAND_0009332','CAND_0064904','CAND_0072688']
# User labels provided in order 1..12
user_labels = [1,1,1,0,2,2,1,1,2,1,1,2]

rows = {}
with open('submission.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows[r['candidate_id']] = r

data = []
missing = []
for cid, lbl in zip(candidate_ids, user_labels):
    r = rows.get(cid)
    if not r:
        missing.append(cid)
        continue
    score = float(r['score'])
    data.append((cid, score, lbl, r['rank']))

if missing:
    print('missing_candidates', missing)

if len(data) == 0:
    raise SystemExit('No candidates found in submission.csv to evaluate')

scores = [d[1] for d in data]
labels = [d[2] for d in data]

# basic stats
mean_label = sum(labels)/len(labels)
mean_score = sum(scores)/len(scores)

# mean model score by label
by_label = {}
for s, l in zip(scores, labels):
    by_label.setdefault(l, []).append(s)
avg_by_label = {l: (sum(v)/len(v)) for l,v in by_label.items()}

# Pearson r
def pearson(x,y):
    mx = sum(x)/len(x)
    my = sum(y)/len(y)
    num = sum((a-mx)*(b-my) for a,b in zip(x,y))
    den = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return num/den if den!=0 else 0.0

r = pearson(scores, labels)

# top disagreements: high model score but human low (label 0 or 1)
diffs = []
for cid, s, l, rk in data:
    diff = s - (l/2.0)  # scale labels 0..2 to 0..1
    diffs.append((diff, cid, s, l, rk))

high_model_low_human = sorted([d for d in diffs if d[3] < 2], reverse=True)[:5]
low_model_high_human = sorted([d for d in diffs if d[3] == 2], key=lambda x: x[0])[:5]

# print concise report
print('num_samples', len(data))
print('mean_user_label', round(mean_label,3))
print('mean_model_score', round(mean_score,3))
print('pearson_score_vs_label', round(r,3))
print('avg_model_score_by_label')
for l in sorted(avg_by_label.keys()):
    print(f'  label_{l}:', round(avg_by_label[l],3))

print('\nTop disagreements: HIGH model score but lower human label')
for diff,cid,s,l,rk in high_model_low_human:
    print(f'  {cid} rank={rk} model={s:.3f} label={l} diff={diff:.3f}')

print('\nTop disagreements: HUMAN labeled 2 but model low')
for diff,cid,s,l,rk in low_model_high_human:
    print(f'  {cid} rank={rk} model={s:.3f} label={l} diff={diff:.3f}')
