"""Residual-quantised VAE: item text embedding -> L discrete codes (the item's 'semantic ID')."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


# Text -> dense embedding
def embed_texts(texts, model_name, batch_size=256):
    if model_name == "tfidf":                                   # offline fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        X = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2).fit_transform(texts)
        emb = TruncatedSVD(min(256, X.shape[1] - 1), random_state=0).fit_transform(X)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        return emb.astype(np.float32)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    return m.encode(texts, batch_size=batch_size, show_progress_bar=True,
                    normalize_embeddings=True).astype(np.float32)


# RQ-VAE
class RQVAE(nn.Module):
    def __init__(self, in_dim, hidden, latent, levels, K, beta):
        super().__init__()
        self.levels, self.K, self.beta = levels, K, beta
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, in_dim))
        self.codebooks = nn.ModuleList([nn.Embedding(K, latent) for _ in range(levels)])

    def quantize(self, z):
        """Level 1 quantises z; level 2 quantises what level 1 missed (the residual); etc."""
        residual, z_q, codes = z, torch.zeros_like(z), []
        vq_loss = z.new_zeros(())
        for cb in self.codebooks:
            idx = torch.cdist(residual.detach(), cb.weight).argmin(1)    # nearest code
            e = cb(idx)
            # codebook loss pulls the code to the residual; commitment pulls the encoder to the code
            vq_loss = vq_loss + F.mse_loss(e, residual.detach()) \
                              + self.beta * F.mse_loss(residual, e.detach())
            z_q = z_q + e
            residual = residual - e.detach()
            codes.append(idx)
        z_q = z + (z_q - z).detach()          # straight-through estimator: decoder grads flow to encoder
        return z_q, torch.stack(codes, 1), vq_loss

    def forward(self, x):
        z = self.encoder(x)
        z_q, codes, vq = self.quantize(z)
        x_hat = self.decoder(z_q)
        return x_hat, codes, F.mse_loss(x_hat, x), vq

    @torch.no_grad()
    def encode_codes(self, x, bs=8192):
        self.eval()
        return torch.cat([self.quantize(self.encoder(x[i:i + bs]))[1] for i in range(0, len(x), bs)])

    @torch.no_grad()
    def init_codebooks(self, x):
        """K-means on the (untrained) encoder's residuals avoids codebook collapse at the start."""
        residual = self.encoder(x)
        for cb in self.codebooks:
            km = KMeans(self.K, n_init=1, max_iter=50, random_state=0).fit(residual.cpu().numpy())
            cb.weight.copy_(torch.from_numpy(km.cluster_centers_).float())
            residual = residual - cb.weight[torch.from_numpy(km.labels_).to(x.device)]

    @torch.no_grad()
    def revive_dead_codes(self, x):
        """Re-seed codes that no item currently uses with random residuals from the data."""
        residual = self.encoder(x)
        for cb in self.codebooks:
            idx = torch.cdist(residual, cb.weight).argmin(1)
            dead = torch.bincount(idx, minlength=self.K) == 0
            if dead.any():
                pick = torch.randint(0, len(x), (int(dead.sum()),), device=x.device)
                cb.weight[dead] = residual[pick] + 0.01 * torch.randn_like(residual[pick])
            residual = residual - cb.weight[torch.cdist(residual, cb.weight).argmin(1)]


def train_rqvae(x, cfg, device):
    c = cfg["semantic_ids"]
    model = RQVAE(x.shape[1], c["hidden_dim"], c["latent_dim"], c["levels"],
                  c["codebook_size"], c["commitment"]).to(device)
    x = x.to(device)
    model.init_codebooks(x)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"])
    N, bs, log = len(x), c["batch_size"], []

    for ep in range(c["epochs"]):
        model.train()
        perm = torch.randperm(N, device=device)
        tot_r = tot_v = 0.0
        for i in range(0, N, bs):
            xb = x[perm[i:i + bs]]
            _, _, recon, vq = model(xb)
            loss = recon + vq
            opt.zero_grad(); loss.backward(); opt.step()
            tot_r += recon.item() * len(xb); tot_v += vq.item() * len(xb)
        if (ep + 1) % 10 == 0 and ep + 1 < c["epochs"]:
            model.revive_dead_codes(x)

        # codebook health: how many codes are used, and the perplexity of their usage
        codes = model.encode_codes(x)
        used, perp = [], []
        for l in range(c["levels"]):
            p = torch.bincount(codes[:, l], minlength=c["codebook_size"]).float() / N
            used.append(int((p > 0).sum())); perp.append(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())))
        log.append({"epoch": ep + 1, "recon": tot_r / N, "vq": tot_v / N, "used": used, "perplexity": perp})
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[rqvae] ep {ep+1:3d} recon {tot_r/N:.4f} vq {tot_v/N:.4f} used {used}")
    return model, model.encode_codes(x).cpu().numpy(), log


# codes -> token ids
def build_item_tokens(codes, K):
    """
    Each item gets L code tokens + 1 de-duplication token, so items whose first L codes collide
    still get unique IDs. Token id layout: 0=PAD, 1=BOS, then one disjoint id range per level.
    """
    N, L = codes.shape
    dup = pd.DataFrame(codes).groupby(list(range(L))).cumcount().to_numpy()   # 0,1,2.. inside a collision group
    Kd = int(dup.max()) + 1
    toks = np.empty((N, L + 1), dtype=np.int64)
    toks[:, :L] = 2 + codes + (np.arange(L) * K)[None, :]
    toks[:, L] = 2 + L * K + dup
    return toks, 2 + L * K + Kd