#!/usr/bin/env python3
"""Watchlist-based congressional trade disclosure backtest.

Tracks the 10 known alpha politicians and back-tests their disclosed
stock purchases. Entry 7 trading days after disclosure date, exit after
30 calendar days or 10% stop loss.

Data: Politician Trade Tracker via RapidAPI (free tier, 60 req/mo).
Cache: data/cache/ — fetched once, reused until deleted.

Usage:
  export RAPIDAPI_KEY=YOUR_KEY
  python backtest.py --save          # full backtest + save CSVs
  python backtest.py --from 2023     # limit to 2023 onwards
  python backtest.py --who pelosi    # single politician
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
ENTRY_LAG_DAYS = 7      # trading days after disclosure before entry
HOLD_DAYS      = 30     # calendar days to hold
STOP_PCT       = 0.10   # stop loss: 10% below entry
MIN_TRADE_USD  = 1_000  # ignore tiny trades (minimum disclosed amount)
SIM_POSITION   = 10_000  # $ per position — 10 politicians × $10k = $100k total capital

RAPIDAPI_HOST    = "politician-trade-tracker1.p.rapidapi.com"
TRADES_BY_TYPE_URL = f"https://{RAPIDAPI_HOST}/trades/type"
POLITICIAN_PROFILE_URL = f"https://{RAPIDAPI_HOST}/politicians/profile"
CACHE_DIR = Path("data/cache")
HEADERS = {"User-Agent": "senator-trader/1.0", "Accept": "application/json, */*"}

KNOWN_ETFS = frozenset({
    "SPY","QQQ","IVV","VOO","VTI","GLD","SLV","TLT","IEF","HYG","LQD",
    "XLF","XLE","XLK","XLV","XLP","XLI","XLU","ARKK","DIA","MDY","IWM",
    "EEM","EFA","VEA","VWO","AGG","BND","SCHD","VUG","VTV","VB","VO",
    "IJH","IJR","IAU","GDX","GDXJ","SH","PSQ","TQQQ","SQQQ","UPRO",
})

# ── Watchlist ─────────────────────────────────────────────────────────────────
# The 10 known high-alpha politicians. Names must loosely match what the
# API returns — matching is case-insensitive substring.
WATCHLIST = [
    {"name": "Nancy Pelosi",             "chamber": "House", "party": "D",
     "focus": "LEAPS options on mega-cap tech (NVDA, AVGO)"},
    {"name": "Tommy Tuberville",         "chamber": "Senate", "party": "R",
     "focus": "Highly active: tech, agriculture, infrastructure, biotech"},
    {"name": "Michael McCaul",           "chamber": "House", "party": "R",
     "focus": "High-volume: tech, finance, industrials"},
    {"name": "Ro Khanna",                "chamber": "House", "party": "D",
     "focus": "Family trust, thousands of trades, broad tech"},
    {"name": "Markwayne Mullin",         "chamber": "Senate", "party": "R",
     "focus": "Energy, defense, industrial manufacturing"},
    {"name": "Josh Gottheimer",          "chamber": "House", "party": "D",
     "focus": "Active short-term: financials, tech, consumer defensive"},
    {"name": "Dan Crenshaw",             "chamber": "House", "party": "R",
     "focus": "Specialized tech, medical devices, defense"},
    {"name": "Debbie Wasserman Schultz", "chamber": "House", "party": "D",
     "focus": "Concentrated high-growth tech and healthcare"},
    {"name": "Kevin Hern",               "chamber": "House", "party": "R",
     "focus": "Energy, industrial manufacturing, commercial real estate"},
    {"name": "Ron Wyden",                "chamber": "Senate", "party": "D",
     "focus": "Healthcare, clean energy, domestic manufacturing"},
]

WATCHLIST_NAMES = [w["name"].lower() for w in WATCHLIST]


def _on_watchlist(name: str) -> bool:
    n = name.lower()
    return any(w in n or n in w for w in WATCHLIST_NAMES)


# ── API helpers ───────────────────────────────────────────────────────────────

def _rapidapi_headers(api_key: str) -> dict:
    return {**HEADERS, "x-rapidapi-key": api_key, "x-rapidapi-host": RAPIDAPI_HOST}


def _fetch_by_type(api_key: str, trade_type: str = "buy", max_pages: int = 10) -> list:
    """Fetch paginated trades of a given type. Cached after first successful fetch."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"trades_type_{trade_type}.json"
    if cache_file.exists() and cache_file.stat().st_size > 5_000:
        try:
            data = json.loads(cache_file.read_text())
            if data:
                print(f"  Using cached {trade_type} data ({len(data)} records)", flush=True)
                return data
        except Exception:
            pass

    all_records: list = []
    page_size = 96
    print(f"  Fetching {trade_type} trades (up to {max_pages} pages) …", flush=True)
    for page in range(1, max_pages + 1):
        resp = requests.get(
            TRADES_BY_TYPE_URL,
            headers=_rapidapi_headers(api_key),
            params={"trade_type": trade_type, "page": page},
            timeout=60,
        )
        if resp.status_code == 401:
            raise RuntimeError("API key invalid — set RAPIDAPI_KEY or pass --api-key KEY")
        if resp.status_code == 429:
            print(f"  Rate limit on page {page} — saving {len(all_records)} records so far")
            break
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data = data.get("trades", data.get("data", data.get("results", [])))
        if not isinstance(data, list) or len(data) == 0:
            break
        all_records.extend(data)
        print(f"  Page {page}: {len(data)} records (running total: {len(all_records)})", flush=True)
        if len(data) < page_size:
            break

    print(f"  ✓ {len(all_records)} total {trade_type} records", flush=True)
    if all_records:
        cache_file.write_text(json.dumps(all_records))
    return all_records


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_amount(s: str) -> float:
    """Parse trade amount ranges to midpoint USD.
    Handles: '1K-15K', '15K-50K', '500K-1M', '1M+', '$15,001-$50,000'.
    """
    if not s:
        return 0.0

    def _expand(tok: str) -> float:
        tok = tok.replace("$", "").replace(",", "").strip().rstrip("+")
        if tok.upper().endswith("M"):
            return float(tok[:-1]) * 1_000_000
        if tok.upper().endswith("K"):
            return float(tok[:-1]) * 1_000
        return float(tok)

    parts = [p.strip() for p in s.split("-") if p.strip()]
    nums = []
    for p in parts:
        try:
            nums.append(_expand(p))
        except (ValueError, IndexError):
            pass
    return sum(nums) / len(nums) if nums else 0.0


def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(s, format=fmt).date()
        except Exception:
            pass
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


# ── Load & filter ─────────────────────────────────────────────────────────────

def load_disclosures(api_key: str, who: Optional[str] = None) -> pd.DataFrame:
    raw = _fetch_by_type(api_key, trade_type="buy")

    rows = []
    for r in raw:
        name = str(r.get("name", "") or r.get("politician", "")).strip()
        if not name:
            continue

        # Filter to watchlist (or specific --who search)
        if who:
            if who.lower() not in name.lower():
                continue
        elif not _on_watchlist(name):
            continue

        sym = str(r.get("ticker", "") or r.get("symbol", "")).strip().upper()
        if not sym or sym in ("N/A", "--", "NONE") or len(sym) > 5:
            continue
        if sym in KNOWN_ETFS:
            continue

        txn = str(r.get("trade_type", "") or r.get("type", "")).lower()
        if not any(kw in txn for kw in ("purchase", "buy")):
            continue

        disc_dt = _parse_date(str(r.get("trade_date", "") or r.get("disclosure_date", "")))
        if not disc_dt:
            continue

        ch = str(r.get("chamber", "")).lower()
        amount_mid = _parse_amount(str(r.get("trade_amount", "") or r.get("amount", "")))

        rows.append({
            "symbol":          sym,
            "member":          name,
            "chamber":         "senate" if "senate" in ch else "house",
            "disclosure_date": disc_dt,
            "amount_mid":      amount_mid,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol", "member", "chamber", "disclosure_date", "amount_mid"])
    df = df[df["amount_mid"] >= MIN_TRADE_USD].copy() if not df.empty else df
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    df = df.sort_values("disclosure_date")

    print(f"\n  Watchlist trades found: {len(df)}")
    if not df.empty:
        print(df.groupby("member").size().sort_values(ascending=False).to_string())
    return df


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_prices(symbols: list, start: str, end: str) -> pd.DataFrame:
    all_syms = list(set(symbols) | {"SPY"})
    print(f"  Downloading prices for {len(all_syms)} symbols …", flush=True)
    raw = yf.download(all_syms, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"]
    return raw[["Close"]].rename(columns={"Close": all_syms[0]})


def _nearest_price(prices: pd.DataFrame, symbol: str,
                   target: date, tolerance_days: int = 5) -> Optional[float]:
    if symbol not in prices.columns:
        return None
    col = prices[symbol].dropna()
    if col.empty:
        return None
    for delta in range(tolerance_days + 1):
        ts = pd.Timestamp(target) + pd.offsets.BDay(delta)
        if ts in col.index:
            return float(col.loc[ts])
    return None


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, prices: pd.DataFrame,
                 from_year: int = 2020) -> pd.DataFrame:
    cutoff = pd.Timestamp(f"{from_year}-01-01")
    df = df[df["disclosure_date"] >= cutoff].copy()
    results = []
    for _, row in df.iterrows():
        sym  = row["symbol"]
        disc = row["disclosure_date"].date()
        entry_date = (pd.Timestamp(disc) + pd.offsets.BDay(ENTRY_LAG_DAYS)).date()
        exit_date  = entry_date + timedelta(days=HOLD_DAYS)
        if exit_date > date.today() - timedelta(days=1):
            continue
        entry_px   = _nearest_price(prices, sym,   entry_date)
        exit_px    = _nearest_price(prices, sym,   exit_date)
        spy_entry  = _nearest_price(prices, "SPY", entry_date)
        spy_exit   = _nearest_price(prices, "SPY", exit_date)
        if not all([entry_px, exit_px, spy_entry, spy_exit]):
            continue
        raw_ret = (exit_px - entry_px) / entry_px
        if raw_ret < -STOP_PCT:
            exit_px = entry_px * (1 - STOP_PCT)
            raw_ret = -STOP_PCT
            exit_reason = "stop"
        else:
            exit_reason = "hold"
        spy_ret = (spy_exit - spy_entry) / spy_entry
        results.append({
            "member":        row["member"],
            "symbol":        sym,
            "disclosure_date": disc,
            "entry_date":    entry_date,
            "exit_date":     exit_date,
            "entry_px":      round(entry_px, 2),
            "exit_px":       round(exit_px,  2),
            "return_pct":    round(raw_ret * 100, 2),
            "spy_return_pct": round(spy_ret * 100, 2),
            "alpha_pct":     round((raw_ret - spy_ret) * 100, 2),
            "exit_reason":   exit_reason,
        })
    return pd.DataFrame(results)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_portfolio(results: pd.DataFrame) -> dict:
    total_pnl = 0.0
    n_trades  = wins = 0
    for _, row in results.iterrows():
        qty = int(SIM_POSITION // row["entry_px"])
        if qty < 1:
            continue
        pnl = qty * (row["exit_px"] - row["entry_px"])
        total_pnl += pnl
        n_trades  += 1
        if pnl > 0:
            wins += 1
    return {
        "n_trades":       n_trades,
        "win_rate":       round(wins / n_trades * 100, 1) if n_trades else 0,
        "total_pnl":      round(total_pnl, 2),
        "avg_pnl_trade":  round(total_pnl / n_trades, 2) if n_trades else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",    default=os.environ.get("RAPIDAPI_KEY", ""))
    parser.add_argument("--from",       dest="from_year", type=int, default=2020)
    parser.add_argument("--who",        default=None, help="Filter to one politician by name")
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: set RAPIDAPI_KEY env var or pass --api-key KEY")
        print("Free signup: https://www.politiciantradetracker.us/")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Watchlist Congressional Trade Backtest")
    print(f"  Entry lag: {ENTRY_LAG_DAYS} days | Hold: {HOLD_DAYS} days | Stop: {STOP_PCT*100:.0f}%")
    print(f"{'='*65}")
    print(f"\nWATCHLIST:")
    for w in WATCHLIST:
        print(f"  {w['name']:<30} {w['chamber']:<8} ({w['party']})  {w['focus']}")

    print(f"\nStep 1: Loading disclosures…")
    df = load_disclosures(args.api_key, who=args.who)
    if df.empty:
        print("No watchlist trades found in data. Check API data range.")
        return

    print(f"\nStep 2: Fetching prices…")
    start_str = f"{args.from_year - 1}-06-01"
    prices    = fetch_prices(df["symbol"].unique().tolist(), start_str, date.today().isoformat())

    print(f"\nStep 3: Running backtest…")
    results = run_backtest(df, prices, from_year=args.from_year)
    print(f"  {len(results)} completed trades")

    if results.empty:
        print("No completed trades — data may only cover recent weeks.")
        print("The API free tier may not have full historical depth.")
        return

    print(f"\n{'='*65}")
    print(f"  RESULTS BY POLITICIAN")
    print(f"{'='*65}")
    by_member = (results.groupby("member")
                 .agg(trades=("return_pct","count"),
                      win_rate=("return_pct", lambda x: f"{(x>0).mean()*100:.0f}%"),
                      avg_return=("return_pct","mean"),
                      avg_alpha=("alpha_pct","mean"),
                      total_alpha=("alpha_pct","sum"))
                 .sort_values("avg_alpha", ascending=False)
                 .round(2))
    print(by_member.to_string())

    sim = simulate_portfolio(results)
    print(f"\n{'='*65}")
    print(f"  PORTFOLIO SIMULATION (${SIM_POSITION:,}/position, all watchlist)")
    print(f"{'='*65}")
    print(f"  Trades:        {sim['n_trades']}")
    print(f"  Win rate:      {sim['win_rate']}%")
    print(f"  Total PnL:     ${sim['total_pnl']:,.2f}")
    print(f"  Avg per trade: ${sim['avg_pnl_trade']:,.2f}")

    print(f"\n{'='*65}")
    print(f"  TOP 15 SYMBOLS TRADED BY WATCHLIST (by avg alpha)")
    print(f"{'='*65}")
    top_syms = (results.groupby("symbol")
                .agg(n=("return_pct","count"),
                     avg_ret=("return_pct","mean"),
                     avg_alpha=("alpha_pct","mean"))
                .sort_values("avg_alpha", ascending=False)
                .head(15))
    print(top_syms.round(2).to_string())

    print(f"\n{'='*65}")
    print(f"  OVERALL STATS")
    print(f"{'='*65}")
    print(f"  Avg return:      {results['return_pct'].mean():.2f}%")
    print(f"  Avg alpha:       {results['alpha_pct'].mean():.2f}%")
    print(f"  Win rate:        {(results['return_pct']>0).mean()*100:.1f}%")
    print(f"  Stopped out:     {(results['exit_reason']=='stop').sum()} "
          f"({(results['exit_reason']=='stop').mean()*100:.1f}%)")

    if args.save:
        results.to_csv("backtest_results.csv", index=False)
        print("\nSaved: backtest_results.csv")


if __name__ == "__main__":
    main()
