import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path); plt.close(fig); print("saved", path)


def plot_rqvae_training(log, path):
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ep, [r["recon"] for r in log], label="reconstruction")
    ax[0].plot(ep, [r["vq"] for r in log], label="quantisation")
    ax[0].set(title="RQ-VAE loss", xlabel="epoch"); ax[0].legend()
    L = len(log[0]["used"])
    for l in range(L):
        ax[1].plot(ep, [r["used"][l] for r in log], label=f"level {l+1}")
        ax[2].plot(ep, [r["perplexity"][l] for r in log], label=f"level {l+1}")
    ax[1].set(title="Codes in use", xlabel="epoch"); ax[1].legend()
    ax[2].set(title="Codebook usage perplexity", xlabel="epoch"); ax[2].legend()
    _save(fig, path)


def plot_codebook_usage(codes, K, path):
    L = codes.shape[1]
    fig, ax = plt.subplots(1, L, figsize=(4.5 * L, 3.5), sharey=True)
    for l in range(L):
        c = np.sort(np.bincount(codes[:, l], minlength=K))[::-1]
        ax[l].bar(range(K), c, width=1.0)
        ax[l].set(title=f"Level {l+1} code frequency (sorted)", xlabel="code rank")
    ax[0].set_ylabel("items")
    _save(fig, path)


def plot_semantic_space(emb, codes, labels, path, n=4000, seed=0):
    """t-SNE of item text embeddings, coloured by first semantic-ID token and by true category."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb), min(n, len(emb)), replace=False)
    X = PCA(min(50, emb.shape[1]), random_state=seed).fit_transform(emb[idx])
    Y = TSNE(2, init="pca", perplexity=30, random_state=seed).fit_transform(X)

    def scatter(ax, values, title):
        top = [v for v, _ in zip(*np.unique(values, return_counts=True))]
        counts = {v: (values == v).sum() for v in top}
        top = sorted(top, key=lambda v: -counts[v])[:20]
        cmap = plt.get_cmap("tab20")
        ax.scatter(Y[:, 0], Y[:, 1], s=3, c="#cccccc")
        for i, v in enumerate(top):
            m = values == v
            ax.scatter(Y[m, 0], Y[m, 1], s=4, color=cmap(i), label=str(v)[:18])
        ax.set(title=title, xticks=[], yticks=[])
        ax.legend(markerscale=3, fontsize=6, ncol=2, loc="best")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    scatter(ax[0], codes[idx, 0], "Coloured by semantic-ID level-1 code (top 20)")
    scatter(ax[1], np.asarray(labels)[idx], "Coloured by true product group")
    _save(fig, path)


def plot_training_curves(log, path):
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ep, [r["train_loss"] for r in log], label="train")
    ax[0].plot(ep, [r["val_loss"] for r in log], label="val (target item)")
    ax[0].set(title="Cross-entropy", xlabel="epoch"); ax[0].legend()
    ax[1].plot(ep, [r["val_ndcg10"] for r in log], marker="o", label="NDCG@10")
    ax[1].plot(ep, [r["val_hr10"] for r in log], marker="o", label="HR@10")
    ax[1].set(title="Validation ranking (beam search)", xlabel="epoch"); ax[1].legend()
    ax[2].plot(ep, [r["lr"] for r in log]); ax[2].set(title="Learning rate (end of epoch)", xlabel="epoch")
    _save(fig, path)


def plot_model_comparison(results, ks, path):
    models = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, metric in zip(ax, ["HR", "NDCG", "MRR"]):
        w = 0.8 / len(models)
        for j, m in enumerate(models):
            a.bar(np.arange(len(ks)) + j * w, [results[m][f"{metric}@{k}"] for k in ks], w, label=m)
        a.set_xticks(np.arange(len(ks)) + 0.4 - w / 2); a.set_xticklabels([f"@{k}" for k in ks])
        a.set(title=metric)
    ax[0].legend()
    _save(fig, path)


def plot_beyond_accuracy(results, k, path):
    models = list(results)
    names = [f"Coverage@{k}", f"Novelty@{k}", f"ILD@{k}"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for a, n in zip(ax, names):
        a.bar(models, [results[m].get(n, 0) for m in models], color=plt.get_cmap("tab10").colors[:len(models)])
        a.set_title(n); a.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_level_accuracy(tf, path):
    L = len(tf["token_acc"]); x = np.arange(L)
    names = [f"level {i+1}" for i in range(L - 1)] + ["dedup"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - 0.2, tf["token_acc"], 0.4, label="token accuracy (teacher-forced)")
    ax.bar(x + 0.2, tf["prefix_acc"], 0.4, label="prefix accuracy (greedy)")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title("Accuracy per semantic-ID level"); ax.legend()
    _save(fig, path)


def plot_breakdowns(rank, hist_len, target_pop, k, path):
    ndcg = np.where(rank <= k, 1 / np.log2(rank + 1), 0.0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    edges = [0, 5, 10, 20, 35, 10 ** 6]
    labs = ["<5", "5-9", "10-19", "20-34", "35+"]
    b = np.digitize(hist_len, edges[1:-1])
    ax[0].bar(labs, [ndcg[b == i].mean() if (b == i).any() else 0 for i in range(len(labs))])
    ax[0].set(title=f"NDCG@{k} by history length (items)", xlabel="history length")
    q = np.quantile(target_pop, np.linspace(0, 1, 11)); q[-1] += 1
    d = np.clip(np.digitize(target_pop, q[1:-1]), 0, 9)
    ax[1].bar(range(1, 11), [ndcg[d == i].mean() if (d == i).any() else 0 for i in range(10)])
    ax[1].set(title=f"NDCG@{k} by target-item popularity decile", xlabel="decile (1 = rarest)")
    _save(fig, path)