import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import torch
from src.utils import load_config, set_seed, get_device, save_json
from src.data import load_processed, split_sequences
from src.model.transformer import SemanticIDTransformer
from src.model.decoding import ConstrainedTrie, generate_rankings
from src.metrics.ranking import compute_metrics, teacher_forced_eval
from src.metrics.baselines import item_popularity, PopularityRecommender, BigramRecommender


def main():
    cfg = load_config(); set_seed(cfg["seed"]); dev = get_device()
    out, proc, e = Path(cfg["paths"]["out_dir"]), Path(cfg["paths"]["processed_dir"]), cfg["eval"]
    ks, max_items = e["ks"], cfg["data"]["max_items_per_seq"]
    assert e["beam_size"] >= max(ks)
    seqs, _, items = load_processed(cfg)
    N = len(items)
    sid = np.load(out / "semantic_ids.npz"); item_tokens, vocab = sid["item_tokens"], int(sid["vocab_size"])
    emb = np.load(proc / "item_emb.npy").astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

    train, _, _, test_h, test_t = split_sequences(seqs)
    pop = item_popularity(train, N)
    ui = np.random.default_rng(cfg["seed"]).choice(len(test_h), min(e["max_eval_users"], len(test_h)), replace=False)
    th, tt = [test_h[i] for i in ui], test_t[ui]

    ckpt = torch.load(out / "transformer_best.pt", map_location=dev, weights_only=False)
    t = cfg["transformer"]
    model = SemanticIDTransformer(vocab, ckpt["ti"], 1 + max_items * ckpt["ti"], t["d_model"], t["n_layers"],
                                  t["n_heads"], t["d_ff"], t["dropout"]).to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()
    trie = ConstrainedTrie(torch.from_numpy(item_tokens), vocab).to(dev)

    # Every model gets the same users, the same histories, the same metric code
    recs = {"Popularity": PopularityRecommender(pop).recommend(th, e["beam_size"]),
            "Bigram": BigramRecommender(train, pop).recommend(th, e["beam_size"]),
            "SemanticID-Transformer": generate_rankings(model, trie, th, item_tokens, max_items,
                                                        e["beam_size"], e["eval_batch_size"], dev)}
    results, ranks = {}, {}
    for name, r in recs.items():
        results[name], ranks[name] = compute_metrics(r, tt, ks, N, pop, emb)
        print(name, {k: round(v, 4) for k, v in results[name].items() if k.endswith(f"@{ks[1]}")})
    save_json(results, out / "results.json")
    save_json(teacher_forced_eval(model, th, tt, item_tokens, max_items, dev), out / "teacher_forced.json")
    np.savez(out / "test_ranks.npz", rank=ranks["SemanticID-Transformer"],
             hist_len=np.array([len(h) for h in th]), target_pop=pop[tt])

    # human-readable case studies
    desc = lambda i: f"{items.prod_name[i]} ({items.product_type_name[i]}, {items.colour_group_name[i]})"
    ranked = recs["SemanticID-Transformer"]
    lines = ["# Case studies (SemanticID-Transformer)\n"]
    for u in range(8):
        lines += [f"## User {u}", "**Last purchases:** " + "; ".join(desc(i) for i in th[u][-4:]),
                  f"**True next item:** {desc(tt[u])}", "**Top-5 recommendations:**"]
        lines += [f"{j+1}. {desc(i)} {'  <-- HIT' if i == tt[u] else ''}" for j, i in enumerate(ranked[u][:5])] + [""]
    (out / "case_studies.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()