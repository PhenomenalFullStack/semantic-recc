import sys; sys.path.insert(0, ".")
import math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import load_config, set_seed, get_device, save_json, PAD
from src.data import load_processed, split_sequences, TrainDataset, collate_right
from src.model.transformer import SemanticIDTransformer
from src.model.decoding import ConstrainedTrie, generate_rankings
from src.metrics.ranking import compute_metrics, teacher_forced_eval
from src.metrics.baselines import item_popularity


def main():
    cfg = load_config(); set_seed(cfg["seed"]); dev = get_device()
    out, t = Path(cfg["paths"]["out_dir"]), cfg["transformer"]
    seqs, _, items = load_processed(cfg)
    sid = np.load(out / "semantic_ids.npz")
    item_tokens, vocab = sid["item_tokens"], int(sid["vocab_size"])
    TI, max_items, N = item_tokens.shape[1], cfg["data"]["max_items_per_seq"], len(items)

    train, val_h, val_t, _, _ = split_sequences(seqs)
    pop = item_popularity(train, N)
    rng = np.random.default_rng(cfg["seed"])
    vi = rng.choice(len(val_h), min(t["val_users"], len(val_h)), replace=False)   # fixed validation subset
    vh, vt = [val_h[i] for i in vi], val_t[vi]

    model = SemanticIDTransformer(vocab, TI, 1 + max_items * TI, t["d_model"], t["n_layers"],
                                  t["n_heads"], t["d_ff"], t["dropout"]).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | vocab {vocab} | device {dev}")
    trie = ConstrainedTrie(torch.from_numpy(item_tokens), vocab).to(dev)

    dl = DataLoader(TrainDataset(train, item_tokens, max_items), batch_size=t["batch_size"],
                    shuffle=True, collate_fn=collate_right, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"], betas=(0.9, 0.95))
    total = t["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / t["warmup_steps"]) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    amp = dev.type == "cuda"                                                # bf16 autocast on GPU only

    log, best, bad = [], -1.0, 0
    for ep in range(1, t["epochs"] + 1):
        model.train(); tot, n, t0 = 0.0, 0, time.time()
        for batch in dl:
            batch = batch.to(dev)
            x, y = batch[:, :-1], batch[:, 1:]                              # next-token prediction
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab), y.reshape(-1), ignore_index=PAD)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1

        if dev.type == "cuda":
            torch.cuda.empty_cache()  # free up memory for validation
            
        tf = teacher_forced_eval(model, vh, vt, item_tokens, max_items, dev)
        ranked = generate_rankings(model, trie, vh, item_tokens, max_items, cfg["eval"]["beam_size"],
                                   cfg["eval"]["eval_batch_size"], dev)
        m, _ = compute_metrics(ranked, vt, [10], N, pop)
        log.append({"epoch": ep, "train_loss": tot / n, "val_loss": tf["loss"], "val_ndcg10": m["NDCG@10"],
                    "val_hr10": m["HR@10"], "lr": sched.get_last_lr()[0]})
        print(f"ep {ep:2d} train {tot/n:.4f} val {tf['loss']:.4f} NDCG@10 {m['NDCG@10']:.4f} "
              f"HR@10 {m['HR@10']:.4f} ({time.time()-t0:.0f}s)")
        save_json(log, out / "train_log.json")

        if m["NDCG@10"] > best:
            best, bad = m["NDCG@10"], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "vocab": vocab, "ti": TI}, out / "transformer_best.pt")
        else:
            bad += 1
            if bad >= t["patience"]:
                print("early stopping"); break


if __name__ == "__main__":
    main()