#!/usr/bin/env python3
"""
HFT Quantum Edge - SOXA + VWAP signal bot (Telegram, text signals only)

Works WITHOUT a paid TradingView plan. It downloads free price data,
runs the same signal logic as your Pine strategy on every CLOSED candle,
and sends a text signal with ENTRY, STOP LOSS and TARGETS to Telegram.

Install once:   pip install yfinance pandas numpy requests
Run:            python hft_quantum_edge_bot.py
Keep it running (computer on, internet on) to receive signals.
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# =====================================================================
# SETTINGS  (edit this part only)
# =====================================================================
BOT_TOKEN = os.getenv("BOT_TOKEN", "PASTE_YOUR_NEW_BOT_TOKEN_HERE")
CHAT_ID   = os.getenv("CHAT_ID", "PASTE_YOUR_CHAT_ID_HERE")

SYMBOL    = "GC=F"        # Yahoo symbol. Gold futures = GC=F. Gold spot = XAUUSD=X. Forex e.g. EURUSD=X
LABEL     = "XAUUSD"      # name shown in Telegram
INTERVAL  = "15m"         # 5m, 15m, 30m or 1h
POLL_SECONDS = 30         # how often to check for a new closed candle

# Forex/gold daily session starts 17:00 New York time (same as TradingView forex "D" bar)
SESSION_TZ = "America/New_York"
SESSION_START_HOUR = 17

SEND_EXIT_ALERTS = False     # False = only the ENTRY signal (with SL + targets). True = also message when T1/T2/T3/SL/BE is hit

# --- model (same as the Pine script) ---
RSI_LEN, CCI_LEN, ATR_LEN, ROC_LEN = 14, 20, 14, 12
WT_EMA, WT_CH = 10, 21
OPT_WIN = 5
REGIME_MODE = "chart"     # "chart" = same as your indicator live | "daily" = true daily close vs daily EMA | "off"
REG_EMA = 50

# --- exits ---
SL_MULT, TP1_MULT, TP2_MULT, TP3_MULT = 2.0, 1.5, 3.0, 5.0
USE_BE = True

# --- VWAP filter ---
USE_VWAP_FILTER = True
MIN_DIST, MAX_DIST = 0.0, 3.09
WAIT_BARS = 10
DEV1, DEV2 = 1.28, 2.01

# =====================================================================
BAR_MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}[INTERVAL]
PERIOD = {"5m": "55d", "15m": "55d", "30m": "55d", "1h": "700d"}[INTERVAL]
WARMUP = 60
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


# ---------------------------------------------------------------- data
def fetch():
    import yfinance as yf
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL,
                     progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError("no data returned")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(SESSION_TZ)
    now = pd.Timestamp.now(tz=SESSION_TZ)
    bar_end = df.index + pd.Timedelta(minutes=BAR_MIN)
    df = df[bar_end <= now + pd.Timedelta(seconds=2)]      # drop the forming candle
    if len(df) < 200:
        raise RuntimeError("not enough candles")
    return df


# ---------------------------------------------------------- indicators
def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def compute(df):
    d = df.copy()
    o, h, l, c = d["Open"], d["High"], d["Low"], d["Close"]

    vol = d["Volume"].astype(float)
    if vol.sum() == 0 or (vol == 0).mean() > 0.5:
        vol = pd.Series(1.0, index=d.index)       # forex has no real volume -> equal weight

    local = d.index.tz_localize(None)
    session = (local - pd.Timedelta(hours=SESSION_START_HOUR)).floor("D")
    sid = pd.Series(session, index=d.index)

    # VWAP + stdev bands (daily reset, hl2)
    hl2 = (h + l) / 2
    cs_v = vol.groupby(sid).cumsum()
    cs_pv = (hl2 * vol).groupby(sid).cumsum()
    cs_p2 = (vol * hl2 * hl2).groupby(sid).cumsum()
    vwap = (cs_pv / cs_v).where(cs_v > 0, hl2)
    var = (cs_p2 / cs_v - vwap * vwap).clip(lower=0)
    vdev = np.sqrt(var).where(cs_v > 0, 0.0)

    # features
    delta = c.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = rma(up, RSI_LEN) / rma(dn, RSI_LEN)
    rsi = 100 - 100 / (1 + rs)
    tp = (h + l + c) / 3
    sma_t = tp.rolling(CCI_LEN).mean()
    md = (tp - sma_t).abs().rolling(CCI_LEN).mean()
    cci = ((tp - sma_t) / (0.015 * md)).where(md != 0, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = rma(tr, ATR_LEN)
    norm_atr = atr / c * 100
    roc = (c - c.shift(ROC_LEN)) / c.shift(ROC_LEN) * 100
    esa = ema(tp, WT_EMA)
    dd = ema((tp - esa).abs(), WT_EMA)
    ci = ((tp - esa) / (0.015 * dd)).where(dd != 0, 0.0)
    wt1 = ema(ci, WT_CH)

    rsin = (rsi - 50) / 50
    ccin = cci.clip(-200, 200) / 200
    atrn = (norm_atr / 5).clip(upper=1)
    rocn = roc.clip(-5, 5) / 5
    wtn = wt1.clip(-60, 60) / 60

    pchg = c - c.shift(1)

    def weight(f):
        r = f.rolling(OPT_WIN).corr(pchg)
        return r.replace([np.inf, -np.inf], np.nan).fillna(0.2)

    w_rsi, w_cci, w_roc, w_wt = weight(rsin), weight(ccin), weight(rocn), weight(wtn)
    w_atr = -weight(atrn)
    tot = w_rsi.abs() + w_cci.abs() + w_atr.abs() + w_roc.abs() + w_wt.abs()
    comp = (w_rsi * rsin + w_cci * ccin + w_atr * atrn + w_roc * rocn + w_wt * wtn) / tot.replace(0, 1)

    # regime filter
    if REGIME_MODE == "off":
        final = comp.copy()
    else:
        if REGIME_MODE == "daily":
            daily = c.groupby(sid).last()
            rp = sid.map(daily.shift(1))
            re_ = sid.map(ema(daily, REG_EMA).shift(1))
        else:
            rp, re_ = c, ema(c, REG_EMA)
        reg = np.sign(rp - re_).fillna(0)
        final = pd.Series(np.where(reg > 0, comp.clip(lower=0),
                                   np.where(reg < 0, comp.clip(upper=0), 0.0)), index=d.index)
    final.iloc[:WARMUP] = np.nan

    d["vwap"], d["vdev"] = vwap, vdev
    d["up1"], d["dn1"] = vwap + DEV1 * vdev, vwap - DEV1 * vdev
    d["up2"], d["dn2"] = vwap + DEV2 * vdev, vwap - DEV2 * vdev
    d["atr"], d["final"], d["sid"] = atr, final, sid
    return d


# ---------------------------------------------------- strategy simulation
def simulate(d):
    n = len(d)
    fo, close = d["final"].values, d["Close"].values
    high, low, atr = d["High"].values, d["Low"].values, d["atr"].values
    vwap, vdev = d["vwap"].values, d["vdev"].values
    idx = d.index

    cur_pos = 0
    pend_l = pend_s = False
    pend_age = 0
    pos = None
    events = []

    for i in range(n):
        # 1) manage open trade (levels as of previous bar)
        if pos is not None and i > pos["bar"]:
            dr = pos["dir"]
            stop_hit = low[i] <= pos["stop"] if dr == 1 else high[i] >= pos["stop"]
            if stop_hit:
                kind = "BE" if (pos["hit"] > 0 and USE_BE) else "SL"
                events.append({"i": i, "time": idx[i], "type": kind, "trade": dict(pos)})
                pos = None
            else:
                while pos["hit"] < 3:
                    lvl = pos["t"][pos["hit"]]
                    reached = high[i] >= lvl if dr == 1 else low[i] <= lvl
                    if not reached:
                        break
                    pos["hit"] += 1
                    events.append({"i": i, "time": idx[i], "type": f"T{pos['hit']}", "trade": dict(pos)})
                    if pos["hit"] == 1 and USE_BE:
                        pos["stop"] = pos["entry"]
                if pos["hit"] >= 3:
                    pos = None

        # 2) raw model signal
        raw_l = raw_s = False
        if i > 0 and not np.isnan(fo[i]) and not np.isnan(fo[i - 1]):
            if fo[i] > 0 and fo[i - 1] <= 0 and cur_pos <= 0:
                raw_l, cur_pos = True, 1
            elif fo[i] < 0 and fo[i - 1] >= 0 and cur_pos >= 0:
                raw_s, cur_pos = True, -1

        # 3) pending signal waits for the VWAP zone
        if raw_l:
            pend_l, pend_s, pend_age = True, False, 0
        elif raw_s:
            pend_s, pend_l, pend_age = True, False, 0
        elif pend_l or pend_s:
            pend_age += 1
            if pend_age > WAIT_BARS:
                pend_l = pend_s = False

        # 4) VWAP zone check
        dist = (close[i] - vwap[i]) / vdev[i] if vdev[i] > 0 else 0.0
        in_disc = dist <= -MIN_DIST and (MAX_DIST == 0 or dist >= -MAX_DIST)
        in_prem = dist >= MIN_DIST and (MAX_DIST == 0 or dist <= MAX_DIST)
        ok_l = (not USE_VWAP_FILTER) or in_disc
        ok_s = (not USE_VWAP_FILTER) or in_prem

        # 5) one trade at a time
        can_enter = pos is None
        long_sig = pend_l and ok_l and can_enter
        short_sig = pend_s and ok_s and can_enter
        if pend_l and ok_l and not can_enter:
            pend_l = False
        if pend_s and ok_s and not can_enter:
            pend_s = False
        if long_sig:
            pend_l = False
        if short_sig:
            pend_s = False

        # 6) open the trade at this bar's close
        if (long_sig or short_sig) and not np.isnan(atr[i]):
            dr = 1 if long_sig else -1
            e, a = close[i], atr[i]
            pos = {"dir": dr, "bar": i, "entry": e, "atr": a,
                   "stop": e - dr * a * SL_MULT, "stop0": e - dr * a * SL_MULT,
                   "t": [e + dr * a * TP1_MULT, e + dr * a * TP2_MULT, e + dr * a * TP3_MULT],
                   "hit": 0, "dist": dist, "entry_time": idx[i]}
            events.append({"i": i, "time": idx[i], "type": "LONG" if dr == 1 else "SHORT",
                           "trade": dict(pos)})
    return events


# ------------------------------------------------------------- formatting
def fp(x):
    return f"{x:,.2f}" if x >= 1000 else (f"{x:.3f}" if x >= 50 else f"{x:.5f}")


# ----------------------------------------------------------------- telegram
def caption(ev):
    tr, t = ev["trade"], ev["type"]
    when = ev["time"].strftime("%d %b %Y %H:%M")
    head = {
        "LONG": "🟢 LONG ENTRY", "SHORT": "🔴 SHORT ENTRY",
        "T1": "🎯 TARGET 1 HIT", "T2": "🎯🎯 TARGET 2 HIT", "T3": "🏆 TARGET 3 HIT",
        "SL": "🛑 STOP LOSS HIT", "BE": "⚪ BREAKEVEN STOP HIT",
    }[t]
    lines = [f"HFT Quantum Edge | {LABEL} {INTERVAL}", head, f"Candle closed: {when} ({SESSION_TZ})"]
    late = (pd.Timestamp.now(tz=SESSION_TZ) - (ev["time"] + pd.Timedelta(minutes=BAR_MIN))).total_seconds() / 60
    if late > 2:
        lines.append(f"Sent {late:.0f} min after candle close - check the live price before entering")
    if t in ("LONG", "SHORT"):
        lines += [f"Entry: {fp(tr['entry'])}", f"Stop:  {fp(tr['stop0'])}",
                  f"T1: {fp(tr['t'][0])}", f"T2: {fp(tr['t'][1])}", f"T3: {fp(tr['t'][2])}",
                  f"VWAP zone: {tr['dist']:+.2f} stdev", "Move stop to entry after T1." if USE_BE else ""]
    else:
        lines += [f"Entry was {fp(tr['entry'])} ({'long' if tr['dir'] == 1 else 'short'})"]
    return "\n".join(x for x in lines if x)


def tg_text(text):
    r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=30)
    r.raise_for_status()


def send_event(d, ev):
    is_entry = ev["type"] in ("LONG", "SHORT")
    if not is_entry and not SEND_EXIT_ALERTS:
        return
    tg_text(caption(ev))


# ---------------------------------------------------------------- main loop
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


def fetch_retry():
    err = None
    for _ in range(3):
        try:
            return fetch()
        except Exception as e:
            err = e
            time.sleep(5)
    raise err


def run_check(state):
    """One full check. Sends any NEW signal since the last check."""
    d = compute(fetch_retry())
    events = simulate(d)
    if "last_time" not in state:
        state["last_time"] = events[-1]["time"].isoformat() if events else d.index[-1].isoformat()
        save_state(state)
        print("First run: old signals ignored. Watching for new ones...")
        return
    last = pd.Timestamp(state["last_time"])
    for ev in events:
        if ev["time"] > last:
            send_event(d, ev)
            state["last_time"] = ev["time"].isoformat()
            save_state(state)
            print(datetime.now().strftime("%H:%M:%S"), "sent", ev["type"])
    print(datetime.now().strftime("%H:%M:%S"), "checked, last candle", d.index[-1].strftime("%d %b %H:%M"))


def main():
    if "PASTE" in BOT_TOKEN or "PASTE" in CHAT_ID:
        sys.exit("BOT_TOKEN / CHAT_ID missing. Set them in the file (PC) or as GitHub secrets (cloud).")
    state = load_state()

    if "--once" in sys.argv:                      # cloud mode: one check, then exit
        try:
            if "last_time" not in state:
                tg_text(f"✅ HFT Quantum Edge connected (cloud mode)\n{LABEL} {INTERVAL}\nWaiting for the next signal...")
            run_check(state)
        except Exception as e:
            print("error:", e)
            sys.exit(1)
        return

    try:                                          # PC mode: keep running
        tg_text(f"✅ HFT Quantum Edge bot started\n{LABEL} {INTERVAL}\nWaiting for the next signal...")
    except Exception as e:
        sys.exit(f"Telegram test failed (check token / chat id / press Start on the bot): {e}")
    while True:
        try:
            run_check(state)
        except Exception as e:
            print(datetime.now().strftime("%H:%M:%S"), "error:", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
