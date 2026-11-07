"""item text -> embedding -> RQ-VAE -> semantic-ID token table."""
import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import torch
from src.utils import load_config, set_seed, get_device, save_json
from src.data import load_processed
from src.model.rqvae import embed_texts, train_rqvae, build_item_tokens


def main():
    cfg = load_config(); set_seed(cfg["seed"]); dev = get_device()
    out, proc = Path(cfg["paths"]["out_dir"]), Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    c = cfg["semantic_ids"]
    _, _, items = load_processed(cfg)

    emb = embed_texts(items["text"].tolist(), c["text_model"])
    np.save(proc / "item_emb.npy", emb.astype(np.float16))
    print("embeddings:", emb.shape)

    # z-score each dimension so the reconstruction loss is not vanishingly small (unit-norm inputs)
    x = torch.from_numpy((emb - emb.mean(0)) / (emb.std(0) + 1e-6))
    model, codes, log = train_rqvae(x, cfg, dev)
    item_tokens, vocab = build_item_tokens(codes, c["codebook_size"])

    n_unique = len({tuple(r) for r in codes.tolist()})
    stats = {"n_items": len(codes), "unique_code_tuples": n_unique,
             "collision_rate": 1 - n_unique / len(codes),
             "max_dedup_index": int(item_tokens[:, -1].max() - (2 + c["levels"] * c["codebook_size"])),
             "vocab_size": int(vocab)}
    print(stats)
    np.savez(out / "semantic_ids.npz", codes=codes, item_tokens=item_tokens, vocab_size=vocab)
    torch.save(model.state_dict(), out / "rqvae.pt")
    save_json(log, out / "rqvae_log.json"); save_json(stats, out / "semantic_id_stats.json")


if __name__ == "__main__":
    main()