import numpy as np
import torch
import torch.nn.functional as F
from ..data import seq_to_tokens, collate_left


def compute_metrics(ranked, target, ks, n_items, item_pop, item_emb=None, ild_users=2000):
    """
    ranked: [U, >=max(k)] recommended item ids (-1 = invalid), best first
    target: [U] the single held-out item per user (leave-one-out)
    item_pop: [N] training popularity counts; item_emb: [N,D] L2-normalised text embeddings
    Returns (metrics dict, per-user 1-based rank of the target; 1e9 if not retrieved).
    """
    ranked, target = np.asarray(ranked), np.asarray(target)
    hit = ranked == target[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, 10 ** 9).astype(np.int64)
    self_info = -np.log2(np.maximum(item_pop, 1) / item_pop.sum())     # rarer item -> higher novelty
    out = {}
    for k in ks:
        in_k = rank <= k
        out[f"HR@{k}"] = float(in_k.mean())                              # == Recall@k with one target
        out[f"NDCG@{k}"] = float(np.where(in_k, 1 / np.log2(rank + 1), 0).mean())
        out[f"MRR@{k}"] = float(np.where(in_k, 1 / rank, 0).mean())
        top = ranked[:, :k]; valid = top >= 0
        out[f"Coverage@{k}"] = float(len(np.unique(top[valid])) / n_items)   # catalogue share ever recommended
        out[f"Novelty@{k}"] = float(self_info[top[valid]].mean())
        if item_emb is not None:                                         # intra-list diversity
            ok = np.flatnonzero(valid.all(1))[:ild_users]
            if len(ok):
                E = item_emb[top[ok]]                                    # [u,k,D]
                sim = np.einsum("ukd,ujd->ukj", E, E)
                out[f"ILD@{k}"] = float(((1 - sim).sum((1, 2)) / (k * (k - 1))).mean())
    return out, rank


@torch.no_grad()
def teacher_forced_eval(model, hists, targets, item_tokens, max_items, device, bs=256):
    """
    Feed history + the TRUE target tokens, and check each next-token prediction.
    token_acc[l]  - P(level-l token correct | true previous tokens)
    prefix_acc[l] - P(tokens 0..l all correct) when the model decodes greedily
    loss          - mean cross-entropy over the target item's tokens
    """
    model.eval()
    TI, V = item_tokens.shape[1], model.vocab_size
    correct, prefix, nll, n = np.zeros(TI), np.zeros(TI), 0.0, 0
    for i in range(0, len(hists), bs):
        h = [seq_to_tokens(x, item_tokens, max_items - 1) for x in hists[i:i + bs]]
        t = item_tokens[targets[i:i + bs]]
        seq = collate_left([np.concatenate([a, b]) for a, b in zip(h, t)]).to(device)
        logits = model(seq[:, :-1])[:, -TI:].float()                     # predictions for the TI target tokens
        y = seq[:, -TI:]
        ok = logits.argmax(-1) == y
        correct += ok.sum(0).cpu().numpy()
        prefix += ok.long().cumprod(1).sum(0).cpu().numpy()
        nll += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum").item()
        n += len(y)
    return {"loss": nll / (n * TI), "token_acc": (correct / n).tolist(), "prefix_acc": (prefix / n).tolist()}