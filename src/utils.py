import json, random
from pathlib import Path
import numpy as np
import torch
import yaml

PAD, BOS = 0, 1          # reserved token ids; semantic-ID tokens start at 2


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)