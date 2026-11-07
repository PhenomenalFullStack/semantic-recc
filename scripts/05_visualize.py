import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
from src.utils import load_config, load_json
from src.data import load_processed
from src.viz import plots as P


def main():
    cfg = load_config()
    out, proc = Path(cfg["paths"]["out_dir"]), Path(cfg["paths"]["processed_dir"])
    fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    ks = cfg["eval"]["ks"]; K = cfg["semantic_ids"]["codebook_size"]
    _, _, items = load_processed(cfg)
    sid = np.load(out / "semantic_ids.npz"); codes = sid["codes"]
    emb = np.load(proc / "item_emb.npy").astype(np.float32)

    P.plot_rqvae_training(load_json(out / "rqvae_log.json"), fig / "01_rqvae_training.png")
    P.plot_codebook_usage(codes, K, fig / "02_codebook_usage.png")
    P.plot_semantic_space(emb, codes, items["product_group_name"].fillna("?").to_numpy(), fig / "03_semantic_space.png")
    P.plot_training_curves(load_json(out / "train_log.json"), fig / "04_training_curves.png")
    res = load_json(out / "results.json")
    P.plot_model_comparison(res, ks, fig / "05_model_comparison.png")
    P.plot_beyond_accuracy(res, ks[1], fig / "06_beyond_accuracy.png")
    P.plot_level_accuracy(load_json(out / "teacher_forced.json"), fig / "07_level_accuracy.png")
    r = np.load(out / "test_ranks.npz")
    P.plot_breakdowns(r["rank"], r["hist_len"], r["target_pop"], ks[1], fig / "08_breakdowns.png")


if __name__ == "__main__":
    main()