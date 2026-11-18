import csv

import matplotlib
import numpy as np
from sentence_transformers import SentenceTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "redis/langcache-embed-v1"  # same model the gateway's cache uses

with open("eval/pairs.csv", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

model = SentenceTransformer(MODEL)
a = model.encode([r["prompt_a"] for r in rows], normalize_embeddings=True)
b = model.encode([r["prompt_b"] for r in rows], normalize_embeddings=True)

dist = 1 - np.sum(a * b, axis=1)  # cosine distance, same scale as RedisVL
y = np.array([int(r["should_match"]) for r in rows])
P, N = int((y == 1).sum()), int((y == 0).sum())
print(f"{len(rows)} pairs: {P} paraphrases, {N} near-misses\n")

thresholds = np.round(np.arange(0.02, 0.62, 0.02), 2)
results = []
print(f"{'thr':>5} {'recall':>8} {'false-hit':>10} {'precision':>10}")
for t in thresholds:
    hit = dist <= t
    tp = int((hit & (y == 1)).sum())
    fp = int((hit & (y == 0)).sum())
    recall = tp / P
    fpr = fp / N
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    results.append((float(t), recall, fpr, prec))
    print(f"{t:>5.2f} {recall:>8.0%} {fpr:>10.0%} {prec:>10.0%}")

with open("eval/results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "paraphrase_hit_rate", "false_hit_rate", "precision"])
    w.writerows(results)

# Inspect the pairs that decide where the threshold can sit
print("\nClosest near-misses (these cause false hits):")
for i in np.argsort(np.where(y == 0, dist, np.inf))[:5]:
    print(f"  {dist[i]:.3f} | {rows[i]['prompt_a']}  <->  {rows[i]['prompt_b']}")

print("\nFurthest paraphrases (these get missed first):")
for i in np.argsort(np.where(y == 1, -dist, np.inf))[:5]:
    print(f"  {dist[i]:.3f} | {rows[i]['prompt_a']}  <->  {rows[i]['prompt_b']}")

ts = [r[0] for r in results]
plt.figure(figsize=(8, 5))
plt.plot(ts, [r[1] for r in results], label="Paraphrase hit rate (want high)")
plt.plot(ts, [r[2] for r in results], label="False-hit rate (want low)")
plt.axvline(0.15, color="gray", linestyle="--", label="Current default (0.15)")
plt.xlabel("Distance threshold (cosine distance, lower = stricter)")
plt.ylabel("Rate")
plt.title("Semantic cache: threshold trade-off")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eval/threshold_curve.png", dpi=150)
print("\nSaved eval/results.csv and eval/threshold_curve.png")