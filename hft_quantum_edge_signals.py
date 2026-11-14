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
BOT_TOKEN = os.getenv("BOT_TOKEN", "9073353197:AAGRUwWjRji6s__a6ITHP0b4VUcRb8kSEcI")  # Set as a GitHub Actions repository secret
CHAT_ID   = os.getenv("CHAT_ID", "9073353197")

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
