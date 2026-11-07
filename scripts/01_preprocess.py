import sys; sys.path.insert(0, ".")
from src.utils import load_config
from src.data import preprocess

if __name__ == "__main__":
    preprocess(load_config())