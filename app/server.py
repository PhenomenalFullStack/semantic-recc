"""FastAPI backend: serves the shop, records every user action, and asks the model for recommendations."""
import sqlite3, time
from contextlib import closing
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.recommender import Recommender, build_history

STATIC = Path(__file__).resolve().parent / "static"
rec = Recommender()                                          # loads the checkpoint once at startup
cfg = rec.cfg
IMG_DIR = Path(cfg["paths"].get("images_dir", "data/raw/images"))
DB_PATH = Path(cfg["paths"]["out_dir"]) / "demo.db"
N_USERS = 10
ITEM_EVENTS = {"view", "wishlist", "unwishlist", "add_to_cart", "remove_from_cart", "purchase"}
ALL_EVENTS = ITEM_EVENTS | {"search", "filter"}

app = FastAPI(title="Semantic-ID recommender demo")


def db():
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    return con

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
with closing(db()) as con, con:
    con.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, type TEXT NOT NULL,
        item INTEGER, meta TEXT, ts REAL NOT NULL)""")


def load_events(user):
    with closing(db()) as con:
        return [dict(r) for r in con.execute(
            "SELECT type, item, meta, ts FROM events WHERE user_id=? ORDER BY id", (user,))]


def check_user(user):
    if not 1 <= user <= N_USERS:
        raise HTTPException(404, f"user must be 1..{N_USERS}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    groups = rec.items["product_group_name"].fillna("Other").value_counts()
    return {"groups": [{"name": g, "count": int(c)} for g, c in groups.items()], "n_items": rec.N, "n_users": N_USERS}


@app.get("/api/catalog")
def catalog(group: str = "All", q: str = "", page: int = Query(1, ge=1), per_page: int = Query(24, ge=1, le=60)):
    df = rec.items.iloc[rec.pop_order]                       # bestsellers first
    if group != "All":
        df = df[df["product_group_name"] == group]
    if q.strip():
        s = q.strip().lower()
        hay = (df["prod_name"].fillna("") + " " + df["product_type_name"].fillna("") + " " +
               df["colour_group_name"].fillna("")).str.lower()
        df = df[hay.str.contains(s, regex=False)]
    ids = df["item"].tolist()
    a = (page - 1) * per_page
    return {"total": len(ids), "page": page, "per_page": per_page, "items": [rec.cards[i] for i in ids[a:a + per_page]]}


@app.get("/api/item/{item}")
def item(item: int):
    if not 0 <= item < rec.N:
        raise HTTPException(404, "no such item")
    return rec.cards[item]


@app.get("/img/{item}")
def image(item: int):
    if not 0 <= item < rec.N:
        raise HTTPException(404)
    aid = int(rec.items["article_id"].iloc[item])
    s = f"{aid:010d}"
    p = IMG_DIR / s[:3] / f"{s}.jpg"                         # H&M layout: images/010/0108775015.jpg
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "public, max-age=86400"})


class Event(BaseModel):
    user: int
    type: str
    item: Optional[int] = None
    meta: Optional[str] = None


@app.post("/api/event")
def post_event(e: Event):
    check_user(e.user)
    if e.type not in ALL_EVENTS:
        raise HTTPException(400, "unknown event type")
    if e.type in ITEM_EVENTS and (e.item is None or not 0 <= e.item < rec.N):
        raise HTTPException(400, "item event needs a valid item id")
    with closing(db()) as con, con:
        con.execute("INSERT INTO events (user_id, type, item, meta, ts) VALUES (?,?,?,?,?)",
                    (e.user, e.type, e.item, e.meta, time.time()))
    return {"ok": True}


@app.post("/api/user/{user}/reset")
def reset(user: int):
    check_user(user)
    with closing(db()) as con, con:
        con.execute("DELETE FROM events WHERE user_id=?", (user,))
    return {"ok": True}


def replay(events):
    """Derive the visible shop state (bag, wishlist) from the raw event log."""
    bag, wish = [], []
    for e in events:
        t, i = e["type"], e["item"]
        if t == "add_to_cart" and i not in bag: bag.append(i)
        elif t in ("remove_from_cart", "purchase") and i in bag: bag.remove(i)
        elif t == "wishlist" and i not in wish: wish.append(i)
        elif t == "unwishlist" and i in wish: wish.remove(i)
    return bag, wish


@app.get("/api/user/{user}/state")
def state(user: int):
    check_user(user)
    ev = load_events(user)
    bag, wish = replay(ev)
    act = [{"type": e["type"], "meta": e["meta"], "ts": e["ts"],
            "item": rec.cards[e["item"]] if e["item"] is not None else None} for e in ev[-40:]][::-1]
    return {"bag": [rec.cards[i] for i in bag], "wishlist": wish, "n_events": len(ev), "activity": act}


@app.get("/api/recommend/{user}")
def recommend(user: int, k: int = Query(12, ge=1, le=30), views: bool = False, raw: bool = False):
    check_user(user)
    ev = load_events(user)
    history = build_history(ev, include_views=views)
    bag, _ = replay(ev)
    purchased = {e["item"] for e in ev if e["type"] == "purchase"}
    exclude = set(history) | set(bag) | purchased            # don't recommend what they already have
    items, source, filled = rec.recommend(history, k, exclude, raw)
    return {"source": source, "n_history": len(history), "filled": filled,
            "history": [rec.cards[i] for i in history[-6:]], "items": items}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)