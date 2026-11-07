import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import PAD, BOS

# Columns of articles.csv that are concatenated into one text description per item
TEXT_COLS = ["prod_name", "product_type_name", "product_group_name", "colour_group_name",
             "department_name", "index_name", "section_name", "garment_group_name", "detail_desc"]


# 1. Preprocessing: raw Kaggle CSVs -> user sequences + item table
def _filter(tx, min_item, min_user, rounds=5):
    """Iteratively drop rare items / short users (each removal can affect the other)."""
    for _ in range(rounds):
        n0 = len(tx)
        ic = tx["article_id"].value_counts()
        tx = tx[tx["article_id"].map(ic) >= min_item]
        uc = tx["user"].value_counts()
        tx = tx[tx["user"].map(uc) >= min_user]
        if len(tx) == n0:
            break
    return tx


def preprocess(cfg):
    raw, proc = Path(cfg["paths"]["raw_dir"]), Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]

    # Stream the (~3.5GB) transactions file in chunks and keep only recent rows.
    parts = []
    for ch in pd.read_csv(raw / "transactions_train.csv",
                          usecols=["t_dat", "customer_id", "article_id"],
                          dtype={"article_id": "int64"}, chunksize=2_000_000):
        ch = ch[ch["t_dat"] >= d["start_date"]]          # ISO dates compare correctly as strings
        if len(ch):
            parts.append(ch)
    tx = pd.concat(parts, ignore_index=True)
    print(f"transactions after date filter: {len(tx):,}")

    tx["day"] = (pd.to_datetime(tx["t_dat"]) - pd.to_datetime(tx["t_dat"]).min()).dt.days.astype("int32")
    tx["user"] = pd.factorize(tx["customer_id"])[0].astype("int32")   # long hex id -> small int
    tx = tx.drop(columns=["customer_id", "t_dat"])

    articles = pd.read_csv(raw / "articles.csv")
    tx = tx[tx["article_id"].isin(articles["article_id"])]

    tx = _filter(tx, d["min_item_count"], d["min_user_len"])

    # Subsample users for tractability, then re-filter.
    users = tx["user"].unique()
    if d["max_users"] and len(users) > d["max_users"]:
        keep = np.random.default_rng(cfg["seed"]).choice(users, d["max_users"], replace=False)
        tx = tx[tx["user"].isin(keep)]
        tx = _filter(tx, d["min_item_count"], d["min_user_len"])

    # Chronological order per user. The file only has dates (no times), so the original file
    # order is the tie-break inside a day; a stable sort preserves it.
    tx = tx.reset_index(drop=True).sort_values(["user", "day"], kind="stable")

    if d["dedup_consecutive"]:
        same = (tx["user"] == tx["user"].shift()) & (tx["article_id"] == tx["article_id"].shift())
        tx = tx[~same]
    uc = tx["user"].value_counts()
    tx = tx[tx["user"].map(uc) >= d["min_user_len"]]

    # Contiguous item ids 0..N-1 (rows of the embedding table / semantic-ID table)
    art_ids = np.sort(tx["article_id"].unique())
    item_of = pd.Series(np.arange(len(art_ids)), index=art_ids)
    tx["item"] = tx["article_id"].map(item_of).to_numpy()

    u, it, dy = tx["user"].to_numpy(), tx["item"].to_numpy().astype(np.int32), tx["day"].to_numpy()
    bounds = np.flatnonzero(np.diff(u)) + 1               # tx is sorted by user -> contiguous blocks
    seqs, days = np.split(it, bounds), np.split(dy, bounds)

    items = articles.set_index("article_id").loc[art_ids].reset_index()
    items["text"] = items[TEXT_COLS].fillna("").astype(str).agg(" | ".join, axis=1)
    items.insert(0, "item", np.arange(len(items)))
    items[["item", "article_id", "prod_name", "product_type_name", "product_group_name",
           "colour_group_name", "text"]].to_csv(proc / "items.csv", index=False)
    with open(proc / "sequences.pkl", "wb") as f:
        pickle.dump({"seqs": seqs, "days": days}, f)

    lens = np.array([len(s) for s in seqs])
    print(f"users={len(seqs):,} items={len(items):,} interactions={lens.sum():,} "
          f"avg_len={lens.mean():.1f} median_len={np.median(lens):.0f}")


def load_processed(cfg):
    proc = Path(cfg["paths"]["processed_dir"])
    with open(proc / "sequences.pkl", "rb") as f:
        d = pickle.load(f)
    items = pd.read_csv(proc / "items.csv")
    return d["seqs"], d["days"], items


def split_sequences(seqs):
    """Leave-one-out: last item = test, second-last = validation, the rest = training."""
    train = [s[:-2] for s in seqs]
    val_h, val_t = [s[:-2] for s in seqs], np.array([s[-2] for s in seqs])
    test_h, test_t = [s[:-1] for s in seqs], np.array([s[-1] for s in seqs])
    return train, val_h, val_t, test_h, test_t


# 2. Tokenisation: item sequence -> flat semantic-ID token sequence
def seq_to_tokens(seq, item_tokens, max_items):
    """[BOS, item1 tokens..., item2 tokens..., ...] using only the most recent `max_items` items."""
    s = np.asarray(seq)[-max_items:]
    return np.concatenate([[BOS], item_tokens[s].reshape(-1)]).astype(np.int64)


class TrainDataset(Dataset):
    def __init__(self, seqs, item_tokens, max_items):
        self.seqs, self.item_tokens, self.max_items = seqs, item_tokens, max_items

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return seq_to_tokens(self.seqs[i], self.item_tokens, self.max_items)


def collate_right(batch):
    """Right-pad (training). Causal attention means real tokens never see the trailing pads."""
    T = max(len(b) for b in batch)
    out = np.full((len(batch), T), PAD, dtype=np.int64)
    for i, b in enumerate(batch):
        out[i, :len(b)] = b
    return torch.from_numpy(out)


def collate_left(batch):
    """Left-pad (inference), so the LAST position of every row is a real token."""
    T = max(len(b) for b in batch)
    out = np.full((len(batch), T), PAD, dtype=np.int64)
    for i, b in enumerate(batch):
        out[i, T - len(b):] = b
    return torch.from_numpy(out)