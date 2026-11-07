"""Constrained beam search: the model may only emit token sequences that are real items."""
import numpy as np
import torch
import torch.nn.functional as F
from ..data import seq_to_tokens, collate_left


class ConstrainedTrie:
    """
      keys[d]  - sorted unique integer keys of every length-d prefix
      masks[d] - [num_prefixes, V] bool, which tokens may follow that prefix
    """
    def __init__(self, item_tokens, vocab_size):
        self.V = vocab_size
        self.N, self.L = item_tokens.shape
        assert vocab_size ** self.L < 2 ** 62, "prefix key would overflow int64"
        self.keys, self.masks = [], []
        for d in range(self.L):
            uniq, inv = torch.unique(self._key(item_tokens[:, :d]), return_inverse=True)
            m = torch.zeros(len(uniq), vocab_size, dtype=torch.bool)
            m[inv, item_tokens[:, d]] = True
            self.keys.append(uniq); self.masks.append(m)
        full = self._key(item_tokens)                              # unique key for every whole item
        self.item_keys, self.item_ids = torch.sort(full)

    def _key(self, prefix):
        key = torch.zeros(prefix.shape[:-1], dtype=torch.long, device=prefix.device)
        for i in range(prefix.shape[-1]):
            key = key * self.V + prefix[..., i]
        return key

    def to(self, device):
        self.keys = [k.to(device) for k in self.keys]
        self.masks = [m.to(device) for m in self.masks]
        self.item_keys, self.item_ids = self.item_keys.to(device), self.item_ids.to(device)
        return self

    def allowed(self, prefix, d):
        """prefix [...,d] -> bool [...,V] of legal next tokens."""
        key = self._key(prefix)
        idx = torch.searchsorted(self.keys[d], key.reshape(-1)).clamp(max=len(self.keys[d]) - 1)
        return self.masks[d][idx].view(*key.shape, self.V)

    def tokens_to_items(self, toks):
        """[...,L] token sequences -> item ids, or -1 if the sequence is not a real item."""
        key = self._key(toks)
        idx = torch.searchsorted(self.item_keys, key.reshape(-1)).clamp(max=len(self.item_keys) - 1)
        found = self.item_keys[idx] == key.reshape(-1)
        return torch.where(found, self.item_ids[idx], torch.full_like(idx, -1)).view(key.shape)


@torch.no_grad()
def beam_search(model, trie, hist, beam):
    """hist: [B,T] left-padded token histories. Returns (item ids [B,beam], log-probs [B,beam])."""
    B, dev, V, L = hist.size(0), hist.device, trie.V, trie.L
    assert beam <= V, "beam size cannot exceed the token vocabulary"
    # step 0: first token of the next item
    logp = F.log_softmax(model(hist, last_only=True).float(), -1)
    logp = logp.masked_fill(~trie.allowed(torch.zeros(B, 0, dtype=torch.long, device=dev), 0), float("-inf"))
    scores, tok = logp.topk(beam, -1)
    gen = tok.unsqueeze(-1)                                        # [B,beam,1]
    hist_e = hist.unsqueeze(1).expand(-1, beam, -1).reshape(B * beam, -1)
    for d in range(1, L):
        inp = torch.cat([hist_e, gen.reshape(B * beam, d)], 1)
        logp = F.log_softmax(model(inp, last_only=True).float(), -1).view(B, beam, V)
        logp = logp.masked_fill(~trie.allowed(gen, d), float("-inf"))
        cand = (scores.unsqueeze(-1) + logp).view(B, beam * V)
        scores, idx = cand.topk(beam, -1)
        src, tok = idx // V, idx % V                               # which beam, which token
        gen = torch.cat([gen.gather(1, src.unsqueeze(-1).expand(-1, -1, d)), tok.unsqueeze(-1)], -1)
    items = trie.tokens_to_items(gen)
    return items.masked_fill(scores == float("-inf"), -1), scores


def generate_rankings(model, trie, hists, item_tokens, max_items, beam, batch_size, device):
    """List of item-id histories -> ranked item ids [n, beam]. Histories are cut to max_items-1
    so history + the generated item never exceeds the position range seen in training."""
    model.eval()
    out = []
    for i in range(0, len(hists), batch_size):
        h = collate_left([seq_to_tokens(x, item_tokens, max_items - 1) for x in hists[i:i + batch_size]])
        items, _ = beam_search(model, trie, h.to(device), beam)
        out.append(items.cpu().numpy())
    return np.concatenate(out)