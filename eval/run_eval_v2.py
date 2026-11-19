import csv

import matplotlib
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMBED_MODEL = "redis/langcache-embed-v1"
VERIFIER = "cross-encoder/quora-roberta-base"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


rows = load("eval/pairs.csv") + load("eval/pairs_extra.csv")
y = np.array([int(r["should_match"]) for r in rows])
P, N = int((y == 1).sum()), int((y == 0).sum())
print(f"{len(rows)} pairs: {P} paraphrases, {N} near-misses (one false hit = {1 / N:.0%})\n")

emb = SentenceTransformer(EMBED_MODEL)
a = emb.encode([r["prompt_a"] for r in rows], normalize_embeddings=True)
b = emb.encode([r["prompt_b"] for r in rows], normalize_embeddings=True)
dist = 1 - np.sum(a * b, axis=1)

ce = CrossEncoder(VERIFIER)
score = np.asarray(
    ce.predict([(r["prompt_a"], r["prompt_b"]) for r in rows]), dtype=float
)
print(f"verifier score range: {score.min():.3f} to {score.max():.3f} (expect within 0..1)")

with open("eval/pair_scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["prompt_a", "prompt_b", "should_match", "distance", "verifier_score"])
    for r, d, s in zip(rows, dist, score):
        w.writerow([r["prompt_a"], r["prompt_b"], r["should_match"], f"{d:.4f}", f"{s:.4f}"])

# How well does each signal separate the classes? 0.5 = random, 1.0 = perfect
print(f"AUC, distance only: {roc_auc_score(y, -dist):.3f}")
print(f"AUC, verifier only: {roc_auc_score(y, score):.3f}\n")


def pt(hit):
    """Return (false-hit rate, paraphrase hit rate) for a boolean hit mask."""
    return (hit & (y == 0)).sum() / N, (hit & (y == 1)).sum() / P


def best_recall(points, max_fpr):
    ok = [rec for fpr, rec in points if fpr <= max_fpr]
    return max(ok) if ok else 0.0


curves = {"Distance only": [pt(dist <= t) for t in np.arange(0.0, 0.80, 0.01)]}
for gate in (0.20, 0.30, 0.40):
    curves[f"Distance <= {gate:.2f} + verifier"] = [
        pt((dist <= gate) & (score >= v)) for v in np.linspace(0, 1, 101)
    ]

targets = (0.0, 0.05, 0.10)
labels = [f"FPR<={m:.0%}" for m in targets]
print("Best paraphrase hit rate while keeping false hits at or below each target:")
print(f"{'method':34}" + "".join(f"{l:>10}" for l in labels))
for name, points in curves.items():
    print(f"{name:34}" + "".join(f"{best_recall(points, m):>10.0%}" for m in targets))

print("\nNear-misses the verifier is most fooled by:")
for i in np.argsort(np.where(y == 0, -score, np.inf))[:5]:
    print(f"  verifier {score[i]:.2f} | dist {dist[i]:.3f} | {rows[i]['prompt_a']}  <->  {rows[i]['prompt_b']}")

print("\nParaphrases the verifier scores lowest:")
for i in np.argsort(np.where(y == 1, score, np.inf))[:5]:
    print(f"  verifier {score[i]:.2f} | dist {dist[i]:.3f} | {rows[i]['prompt_a']}  <->  {rows[i]['prompt_b']}")

plt.figure(figsize=(8, 6))
for name, points in curves.items():
    xs, ys = zip(*points)
    plt.plot(xs, ys, label=name)
plt.xlabel("False-hit rate (lower is better)")
plt.ylabel("Paraphrase hit rate (higher is better)")
plt.title("Does a verifier beat a distance cutoff alone?")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eval/verifier_curves.png", dpi=150)
print("\nSaved eval/pair_scores.csv and eval/verifier_curves.png")
