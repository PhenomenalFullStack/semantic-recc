import numpy as np
import scipy.sparse as sp


def item_popularity(train_seqs, n_items):
    return np.bincount(np.concatenate(train_seqs), minlength=n_items)


class PopularityRecommender:
    """Same top-K list for everyone: the globally most purchased items."""
    def __init__(self, pop):
        self.order = np.argsort(-pop)

    def recommend(self, hists, k):
        return np.tile(self.order[:k], (len(hists), 1))


class BigramRecommender:
    """Item->next-item co-occurrence counts (a first-order Markov chain), backed off to popularity."""
    def __init__(self, train_seqs, pop):
        n = len(pop)
        src = np.concatenate([s[:-1] for s in train_seqs]); dst = np.concatenate([s[1:] for s in train_seqs])
        self.C = sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n))
        self.pop_order = np.argsort(-pop)

    def recommend(self, hists, k):
        out = np.empty((len(hists), k), dtype=np.int64)
        for u, h in enumerate(hists):
            row = self.C[int(h[-1])]
            top = list(row.indices[np.argsort(-row.data)[:k]])
            if len(top) < k:                                             # back off to popularity
                have = set(top)
                top += [i for i in self.pop_order[:k + len(top)] if i not in have][:k - len(top)]
            out[u] = top
        return out