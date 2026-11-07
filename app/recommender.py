"""Loads the trained model once and turns a user's raw shop activity into recommendations."""
import threading
from pathlib import Path
import numpy as np
import torch

from src.utils import load_config, get_device
from src.data import load_processed, split_sequences, seq_to_tokens, collate_left
from src.model.transformer import SemanticIDTransformer
from src.model.decoding import ConstrainedTrie, beam_search
from src.metrics.baselines import item_popularity


def build_history(events, include_views=False):
    """
    Turn a user's time-ordered events into the item sequence the model reads.
    The model was trained on PURCHASE sequences, so the signals are:
      add_to_cart  -> counts when the item is added (the intent to buy)
      purchase     -> counts only if the item was NOT already counted via the bag ("buy now")
      remove_from_cart -> the user changed their mind: drop that item from the history again
      view         -> counted only if include_views (NOT what the model was trained on)
    Wishlist, search and filter events are logged but not fed to the model.
    """
    hist, bag = [], set()
    for e in events:
        t, i = e["type"], e["item"]
        if i is None:
            continue
        if t == "add_to_cart":
            bag.add(i); hist.append(i)
        elif t == "remove_from_cart":
            bag.discard(i)
            if i in hist:                                   # remove the most recent occurrence
                hist.pop(len(hist) - 1 - hist[::-1].index(i))
        elif t == "purchase":
            if i in bag:
                bag.discard(i)                              # already counted when it was added
            else:
                hist.append(i)                              # "buy now" without a prior add
        elif t == "view" and include_views:
            hist.append(i)
    out = []                                                # collapse immediate repeats, as in training
    for i in hist:
        if not out or out[-1] != i:
            out.append(i)
    return out


class Recommender:
    BEAM = 50              # candidates kept by beam search (must be <= vocab size)
    MAX_PER_NAME = 1       # presentation filter: at most this many colour variants of one product

    def __init__(self, cfg_path="config.yaml"):
        self.cfg = load_config(cfg_path)
        self.dev = get_device()
        out = Path(self.cfg["paths"]["out_dir"])
        seqs, _, self.items = load_processed(self.cfg)
        sid = np.load(out / "semantic_ids.npz")
        self.item_tokens, vocab = sid["item_tokens"], int(sid["vocab_size"])
        self.N, self.max_items = len(self.items), self.cfg["data"]["max_items_per_seq"]
        ti, t = self.item_tokens.shape[1], self.cfg["transformer"]

        ckpt = torch.load(out / "transformer_best.pt", map_location=self.dev, weights_only=False)
        self.model = SemanticIDTransformer(vocab, ti, 1 + self.max_items * ti, t["d_model"], t["n_layers"],
                                           t["n_heads"], t["d_ff"], t["dropout"]).to(self.dev)
        self.model.load_state_dict(ckpt["model"]); self.model.eval()
        self.trie = ConstrainedTrie(torch.from_numpy(self.item_tokens), vocab).to(self.dev)
        self.beam = min(self.BEAM, vocab)

        # popularity from TRAINING data only (cold-start fallback + bestseller ordering of the shop)
        self.pop = item_popularity(split_sequences(seqs)[0], self.N)
        self.pop_order = np.argsort(-self.pop)
        del seqs
        self.lock = threading.Lock()                        # one inference at a time on the model

        self.names = self.items["prod_name"].fillna("").tolist()
        self.cards = [{"item": int(i), "name": n, "type": str(ty), "group": str(g), "colour": str(c)}
                      for i, n, ty, g, c in zip(self.items["item"], self.names,
                                                self.items["product_type_name"].fillna(""),
                                                self.items["product_group_name"].fillna(""),
                                                self.items["colour_group_name"].fillna(""))]
        print(f"recommender ready: {self.N} items | device {self.dev} | beam {self.beam}")

    @torch.no_grad()
    def recommend(self, history, k=12, exclude=(), raw=False):
        """
        Returns (items, source, n_filled). items = [{card..., 'tag': 'model'|'popular'}].
        raw=True returns the model's top-k untouched (may include items the user already has,
        and several colours of one product). Otherwise: drop seen items, cap colour variants,
        and top up with popular items if the model's candidates run out.
        """
        if history:
            h = collate_left([seq_to_tokens(history, self.item_tokens, self.max_items - 1)]).to(self.dev)
            with self.lock:
                items, _ = beam_search(self.model, self.trie, h, self.beam)
            cand = [int(i) for i in items[0].cpu().tolist() if i >= 0]
            source = "model"
        else:
            cand, source = [], "popular"                    # cold start: nothing to condition on
        if raw and cand:
            return [dict(self.cards[i], tag="model") for i in cand[:k]], source, 0

        exclude, seen_names, out = set(exclude), {}, []
        def take(pool, tag):
            for i in pool:
                if len(out) >= k:
                    return
                n = self.names[i]
                if i in exclude or seen_names.get(n, 0) >= self.MAX_PER_NAME or any(o["item"] == i for o in out):
                    continue
                seen_names[n] = seen_names.get(n, 0) + 1
                out.append(dict(self.cards[i], tag=tag))
        take(cand, "model")
        n_model = len(out)
        take(self.pop_order[:500].tolist(), "popular")      # fill any gap with bestsellers
        return out, source, len(out) - n_model if source == "model" else len(out)