import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
DATA = Path(os.getenv("DATA_DIR", "/data"))
DATA.mkdir(parents=True, exist_ok=True)
CONFIG, CACHE = DATA / "watchlist.json", DATA / "cache.json"
if not CONFIG.exists():
    shutil.copy(BASE / "watchlist.default.json", CONFIG)
REFRESH_MIN = int(os.getenv("REFRESH_MINUTES", "30"))
HISTORY = os.getenv("HISTORY_PERIOD", "max")
DEFAULTS = {"pivot_window": 5, "lookback": 252, "cluster_pct": 1.5,
            "min_touches": 2, "breach_buffer_pct": 1.0, "breach_recent_days": 20,
            "structure": "swing", "vix_bands": [15, 20, 30]}


# ---- Signals --------------------------------------------------------------
# Each takes (daily OHLC DataFrame, settings) and returns
# {"level": "good" | "warn" | "bad" | "na", "label": str, "detail": str}.
# To add a column, write another function and register it in SIGNALS below.

def sma200(df, s):
    c = df["Close"]
    sma = c.rolling(200).mean()
    if sma.notna().sum() < 2:
        return {"level": "na", "label": "No data", "detail": "needs 200 days of history"}
    above = (c > sma)[sma.notna()]
    cur = bool(above.iloc[-1])
    diff = np.where(above.values != cur)[0]
    days = len(above) - 1 - (diff[-1] if len(diff) else -1)  # bars on this side of the SMA
    pct = (c.iloc[-1] / sma.iloc[-1] - 1) * 100
    prev = sma.iloc[-21]
    slope = "rising" if sma.iloc[-1] > prev else "falling" if sma.iloc[-1] < prev else "flat"
    fresh = days <= 5
    if cur:
        label, level = ("Just reclaimed", "warn") if fresh else ("Above", "good")
    else:
        label, level = ("Just broke", "bad") if fresh else ("Below", "bad")
    cross = (f"crossed {days - 1}d ago" if len(diff) and days > 1
             else "crossed today" if len(diff) else f"no cross in {days}d")
    return {"level": level, "label": label,
            "detail": f"{pct:+.1f}% from {sma.iloc[-1]:.2f}, {cross}, SMA {slope}"}


def _tag(t):
    return "manual" if t == 0 else f"{t} touches"


def support(df, s):
    w, lb = s["pivot_window"], s["lookback"]
    lows, closes = df["Low"].values[-lb:], df["Close"].values[-lb:]
    n = len(lows)
    if n < 4 * w:
        return {"level": "na", "label": "No data", "detail": "not enough history"}
    # Swing lows, then cluster nearby ones into levels; touches = swing lows in the cluster
    pivots = sorted(lows[i] for i in range(w, n - w) if lows[i] == lows[i - w:i + w + 1].min())
    clusters = []
    for p in pivots:
        if clusters and p <= clusters[-1][0] * (1 + s["cluster_pct"] / 100):
            clusters[-1].append(p)
        else:
            clusters.append([p])
    if s.get("support"):  # manual override replaces auto-detection for this ticker
        levels = [(float(x), 0) for x in s["support"]]
    else:
        levels = [(float(np.mean(c)), len(c)) for c in clusters if len(c) >= s["min_touches"]]
    levels += [(float(x), 0) for x in s.get("support_add", [])]  # manual extras on top of auto
    px, buf = closes[-1], 1 - s["breach_buffer_pct"] / 100
    broken = [l for l in levels if px < l[0] * buf]
    intact = [l for l in levels if px >= l[0] * buf]
    if broken:
        lvl, t = min(broken)  # nearest level above price
        days = 0
        for x in closes[::-1]:
            if x >= lvl * buf:
                break
            days += 1
        if days <= s["breach_recent_days"]:
            return {"level": "bad", "label": "Breached",
                    "detail": f"{lvl:.2f} ({_tag(t)}), closed below {days}d"}
    if intact:
        lvl, t = max(intact)  # nearest level below price
        d = (px / lvl - 1) * 100
        return {"level": "good" if d > 3 else "warn", "label": "Holding",
                "detail": f"{lvl:.2f} ({_tag(t)}), {d:.1f}% above"}
    return {"level": "na", "label": "No level", "detail": "no confirmed support"}


def _swings(v, w, hi):
    fn = np.max if hi else np.min
    pts = []
    for i in range(w, len(v) - w):
        if v[i] == fn(v[i - w:i + w + 1]) and not (pts and i - pts[-1][0] <= w):
            pts.append((i, v[i]))  # equal highs or lows inside one window count once
    return pts


def _structure(df, s, hi):
    v = df["High" if hi else "Low"].values
    if s.get("structure") == "swing":
        v = v[-s["lookback"]:]
        pts = _swings(v, s["pivot_window"], hi)
        if len(pts) < 2:
            return {"level": "na", "label": "No data", "detail": "not enough swings"}
        (_, a), (i, b) = pts[-2], pts[-1]
        if hi:  # latest swing high is below the one before, and price has not taken it out
            yes = bool(b < a and v[i:].max() <= b)
            detail = f"swing high {b:.2f} vs {a:.2f} before"
        else:  # latest swing low is below the one before, or price is undercutting it now
            yes = bool(b < a or v[-1] < b)
            detail = f"swing low {b:.2f} vs {a:.2f} before" + (f", now {v[-1]:.2f}" if v[-1] < b else "")
    else:  # latest daily bar vs the previous one
        a, b = v[-2], v[-1]
        yes = bool(b < a)
        detail = f"{b:.2f} vs {a:.2f} prior bar"
    return {"level": "warn" if yes else "na", "label": "Yes" if yes else "No", "flag": yes,
            "detail": detail}


SIGNALS = {"200d SMA": sma200, "Support": support,
           "Lower high": lambda df, s: _structure(df, s, True),
           "Lower low": lambda df, s: _structure(df, s, False)}


# ---- Data + refresh ---------------------------------------------------------
# Market-wide tiles (shown above the table). Each takes settings and returns
# {"name", "value", "label", "level", "detail"}. Register new ones in MARKET.
def vix(s, frames):
    df = frames.get("^VIX")
    if df is None or len(df) < 2:
        return {"name": "VIX", "value": "n/a", "label": "No data", "level": "na", "detail": ""}
    c = df["Close"]
    v, chg = float(c.iloc[-1]), float(c.iloc[-1] / c.iloc[-2] - 1) * 100
    lo, mid, hi = s["vix_bands"]
    label, level = (("Complacent", "warn") if v < lo else ("Normal", "na") if v < mid
                    else ("Elevated", "warn") if v < hi else ("Fear", "bad"))
    return {"name": "VIX", "value": f"{v:.2f}", "label": label, "level": level,
            "detail": f"{chg:+.1f}% on the day, {df.index[-1].date()}"}


MARKET = [vix]
state = {"updated": None, "columns": list(SIGNALS), "market": [], "rows": []}
lock = threading.Lock()
FRAMES = {}  # last fetched price history per symbol, used for the as-of view


def fetch(sym):
    try:
        t = yf.Ticker(sym)
        df = t.history(period=HISTORY, interval="1d", auto_adjust=False).dropna(subset=["Close"])
        try:  # the long request can lag by a bar; a short one often has the newest
            r = t.history(period="5d", interval="1d", auto_adjust=False).dropna(subset=["Close"])
            if len(df) and len(r):
                df = pd.concat([df, r[r.index > df.index[-1]]])
        except Exception:
            pass
        return sym, (df if len(df) else None)
    except Exception as e:
        print(f"fetch {sym} failed: {e}", flush=True)
        return sym, None


def build_row(sym, name, df, settings):
    c = df["Close"]
    row = {"symbol": sym, "name": name, "price": round(float(c.iloc[-1]), 2),
           "chg": round(float(c.iloc[-1] / c.iloc[-2] - 1) * 100, 2) if len(c) > 1 else 0,
           "asof": str(df.index[-1].date()), "signals": {}}
    for key, fn in SIGNALS.items():
        try:
            res = fn(df, settings)
            res.setdefault("flag", res["level"] == "bad")
            row["signals"][key] = res
        except Exception as e:
            row["signals"][key] = {"level": "na", "label": "Error", "detail": str(e)}
    return row


def make_state(frames, items, settings, asof=None):
    if asof:  # historical view: pretend the data ends on that date
        frames = {k: v.loc[:asof] for k, v in frames.items()}
    rows = []
    for i in items:
        df, name = frames.get(i["symbol"]), i.get("name", "")
        if df is None or len(df) < 30:
            rows.append({"symbol": i["symbol"], "name": name, "price": None, "chg": None, "signals": {}})
        else:
            manual = {k: i[k] for k in ("support", "support_add") if k in i}
            rows.append(build_row(i["symbol"], name, df, {**settings, **manual}))
    market = []
    for fn in MARKET:
        try:
            market.append(fn(settings, frames))
        except Exception as e:
            market.append({"name": fn.__name__.upper(), "value": "n/a", "label": "Error",
                           "level": "na", "detail": str(e)})
    return {"columns": list(SIGNALS), "market": market, "rows": rows}


def load_config():
    cfg = json.loads(CONFIG.read_text())
    settings = {**DEFAULTS, **cfg.get("settings", {})}
    items = [t if isinstance(t, dict) else {"symbol": t} for t in cfg["tickers"]]
    return settings, items


def refresh():
    settings, items = load_config()
    symbols = list(dict.fromkeys([i["symbol"] for i in items] + ["^VIX"]))
    with ThreadPoolExecutor(8) as ex:
        data = {k: v for k, v in ex.map(fetch, symbols) if v is not None}
    with lock:
        FRAMES.clear()
        FRAMES.update(data)
        state.update(updated=datetime.now(timezone.utc).isoformat(), **make_state(FRAMES, items, settings))
        CACHE.write_text(json.dumps(state))


def loop():
    while True:
        try:
            refresh()
        except Exception as e:
            print(f"refresh failed: {e}", flush=True)
        time.sleep(REFRESH_MIN * 60)


app = FastAPI()


@app.on_event("startup")
def start():
    try:
        state.update(json.loads(CACHE.read_text()))
    except Exception:
        pass
    threading.Thread(target=loop, daemon=True).start()


@app.get("/api/data")
def get_data(asof: Optional[str] = None):
    if not asof:
        with lock:
            return state
    try:
        datetime.strptime(asof, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "asof must look like 2020-03-23")
    settings, items = load_config()
    with lock:
        frames, updated = dict(FRAMES), state["updated"]
    if not frames:  # still loading after a restart
        return {"updated": None, "columns": list(SIGNALS), "market": [], "rows": []}
    return {"updated": updated, "asof": asof, **make_state(frames, items, settings, asof)}


@app.post("/api/refresh")
def force_refresh():
    threading.Thread(target=refresh, daemon=True).start()
    return {"ok": True}


@app.get("/api/debug/{symbol}")
def debug(symbol: str):
    """Raw last bars for both requests, to see exactly what Yahoo returned."""
    t, out = yf.Ticker(symbol), {"yfinance": yf.__version__}
    for p in (HISTORY, "5d"):
        try:
            out[p] = t.history(period=p, interval="1d", auto_adjust=False).tail(4).reset_index().astype(str).to_dict("records")
        except Exception as e:
            out[p] = str(e)
    return out


NAMES = {"ES=F": "E-mini S&P 500", "NQ=F": "E-mini Nasdaq 100"}


class TickerIn(BaseModel):
    symbol: str = Field(pattern=r"^[A-Za-z0-9.\-^=/]{1,15}$")
    name: str = Field("", max_length=60)
    support: Optional[List[float]] = None
    support_add: Optional[List[float]] = None


class SettingsIn(BaseModel):
    pivot_window: int = Field(5, ge=1, le=30)
    lookback: int = Field(252, ge=60, le=1000)
    cluster_pct: float = Field(1.5, gt=0, le=10)
    min_touches: int = Field(2, ge=1, le=10)
    breach_buffer_pct: float = Field(1.0, ge=0, le=10)
    breach_recent_days: int = Field(20, ge=1, le=250)
    structure: Literal["swing", "bar"] = "swing"
    vix_bands: List[float] = Field([15, 20, 30], min_length=3, max_length=3)


class ConfigIn(BaseModel):
    settings: SettingsIn = SettingsIn()
    tickers: List[TickerIn] = Field(max_length=100)


@app.get("/api/config")
def get_config():
    cfg = json.loads(CONFIG.read_text())
    return {"settings": {**DEFAULTS, **cfg.get("settings", {})}, "defaults": DEFAULTS,
            "tickers": [t if isinstance(t, dict) else {"symbol": t} for t in cfg["tickers"]]}


@app.put("/api/config")
def put_config(cfg: ConfigIn):
    if cfg.settings.vix_bands != sorted(cfg.settings.vix_bands):
        raise HTTPException(422, "VIX bands must be in ascending order")
    out, seen = {"settings": cfg.settings.model_dump(), "tickers": []}, set()
    for t in cfg.tickers:
        sym = t.symbol.upper()
        if sym.startswith("/"):  # thinkorswim style futures, e.g. /ES becomes ES=F
            sym = sym[1:] + "=F"
        if sym in seen:
            continue
        seen.add(sym)
        d = {"symbol": sym, "name": t.name or NAMES.get(sym, "")}
        for k in ("support", "support_add"):
            if getattr(t, k):
                d[k] = getattr(t, k)
        out["tickers"].append(d)
    tmp = CONFIG.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(CONFIG)
    threading.Thread(target=refresh, daemon=True).start()
    return {"ok": True}


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
