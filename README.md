Generative Recommender with Semantic IDs

I built a generative recommender system on the H&M fashion dataset (about 2.3M purchases from 187K shoppers). Product text is compressed into short discrete “semantic IDs” with an RQ-VAE. A small decoder-only transformer (3.4M parameters) then predicts a shopper’s next item by generating its ID token by token, using beam search constrained to real products. In leave-one-out testing on 20,000 users, it reached a 14.1% Hit Rate@10, about 44% higher than a strong sequential baseline (9.8%). It ties that baseline at the very top of the ranking and wins in the lower part of the top-10/top-20 lists. I also built a working online shop with 10 simulated users. It records their activity (views, bag, purchases) and updates a “Recommended” section live from the model.

Tech: Python, PyTorch, FastAPI, SQLite, JavaScript




semantic-rec/
|
├── app/
│   ├── __init__.py            # empty file
│   ├── recommender.py
│   ├── server.py
│   └── static/
│       └── index.html
│
├── config.yaml
├── requirements.txt
│
├── data/
│   ├── raw/                      # articles.csv, transactions_train.csv
│   └── processed/                # created by script 01 / 02
|
├── src/
│   ├── __init__.py
│   ├── utils.py                  # config, seeding, device, json helpers
│   ├── data.py                   # preprocessing, splits, tokenisation, datasets
|   |
│   ├── model/                    # PART 1: THE MODEL
│   │   ├── __init__.py
│   │   ├── rqvae.py              # text embeddings -> RQ-VAE -> semantic IDs
│   │   ├── transformer.py        # decoder-only transformer
│   │   └── decoding.py           # trie-constrained beam search
│   │
│   ├── metrics/                  # PART 2: METRICS
│   │   ├── __init__.py
│   │   ├── ranking.py            # HR/NDCG/MRR/coverage/novelty/ILD, token accuracy
│   │   └── baselines.py   
│   │
│   │
│   └── viz/                      # PART 3: VISUALISATION
│       ├── __init__.py
│       └── plots.py
│
├── scripts/
│   ├── 01_preprocess.py
│   ├── 02_build_semantic_ids.py
│   ├── 03_train_transformer.py
│   ├── 04_evaluate.py
│   └── 05_visualize.py
│
└── outputs/                      # checkpoints, logs, results, figures

# Training and Metrics
python scripts/01_preprocess.py
python scripts/02_build_semantic_ids.py
python scripts/03_train_transformer.py
python scripts/04_evaluate.py
python scripts/05_visualize.py

# Frontend
python -m uvicorn app.server:app --port 8000