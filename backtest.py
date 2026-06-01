#!/usr/bin/env python3
"""Senator trade disclosure backtest.

Answers two questions:
  1. Which senators/representatives have the strongest post-disclosure edge?
  2. If you followed the top N performers, what would your return have been?

Method:
  - Entry:  disclosure_date + ENTRY_LAG_DAYS trading days (simulates realistic reaction time)
  - Exit:   entry + HOLD_DAYS calendar days, or stop loss at -STOP_PCT%, whichever first
  - Benchmark: SPY return over the same window
  - Alpha:  trade return minus SPY return (risk-adjusted edge)

Data source:
  Senate / House stock watcher S3 (free, no API key)

Usage:
  python backtest.py                   # full analysis, all years
  python backtest.py --top 20          # show top 20 members
  python backtest.py --from 2022       # limit to 2022 onwards
  python backtest.py --chamber senate  # senate only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
ENTRY_LAG_DAYS = 7      # trading days after disclosure before we enter
HOLD_DAYS      = 30     # calendar days to hold
STOP_PCT       = 0.10   # stop loss at 10% below entry
MIN_TRADE_USD  = 15_001 # ignore < $15k disclosed trades (senators report in ranges)
MIN_PRICE      = 5.0    # ignore penny stocks
TOP_N_DEFAULT  = 15     # senators to include in portfolio simulation
SIM_POSITION   = 5_000  # $ per position in portfolio simulation

# Politician Trade Tracker via RapidAPI
# Sign up free (60 req/mo) at https://www.politiciantradetracker.us/
# Set via --api-key or RAPIDAPI_KEY env var.
RAPIDAPI_HOST = "politician-trade-tracker1.p.rapidapi.com"
TRADES_BY_CHAMBER_URL = f"https://{RAPIDAPI_HOST}/trades/chamber"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; senator-backtest/1.0)",
    "Accept": "application/json, */*",
}
CACHE_DIR = Path("data/cache")

KNOWN_ETFS = frozenset({
    "SPY","QQQ","IVV","VOO","VTI","GLD","SLV","TLT","IEF","HYG","LQD",
    "XLF","XLE","XLK","XLV","XLP","XLI","XLU","ARKK","DIA","MDY","IWM",
    "EEM","EFA","VEA","VWO","AGG","BND","SCHD","TQQQ","SQQQ","UPRO",
    "VUG","VTV","VB","VO","IJH","IJR","IAU","GDX","GDXJ","SH","PSQ",
})


# ── Data fetching ─────────────────────────────────────────────────────────────

def _rapidapi_headers(api_key: str) -> dict:
    return {
        **HEADERS,
        "x-rapidapi-key":  api_key,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }


def _fetch_chamber(api_key: str, chamber: str) -> list:
    """Fetch all trades for one chamber, with cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"trades_{chamber}.json"
    if cache_file.exists() and cache_file.stat().st_size > 5_000:
        try:
            data = json.loads(cache_file.read_text())
            if data:
                print(f"  Using cached {chamber} data ({len(data)} records)", flush=True)
                return data
        except Exception:
            pass
    print(f"  Fetching {chamber} trades from Politician Trade Tracker …", flush=True)
    resp = requests.get(
        TRADES_BY_CHAMBER_URL,
        headers=_rapidapi_headers(api_key),
        params={"chamber": chamber, "limit": 10000},
        timeout=60,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "API key invalid — check your RapidAPI key and re-run with --api-key KEY"
        )
    if resp.status_code == 429:
        raise RuntimeError("Rate limit hit — 60 requests/month on free tier. Try again tomorrow.")
    resp.raise_for_status()
    data = resp.json()
    # API may return a dict with a list inside, or a bare list
    if isinstance(data, dict):
        data = data.get("trades", data.get("data", data.get("results", [])))
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response format: {type(data)} — {str(data)[:200]}")
    print(f"  ✓ {len(data)} {chamber} records", flush=True)
    cache_file.write_text(json.dumps(data))
    return data


def _parse_amount(s: str) -> float:
    if not s:
        return 0.0
    clean = s.replace("$","").replace(",","").strip()
    parts = [p.strip() for p in clean.split("-") if p.strip()]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass
    return sum(nums) / len(nums) if nums else 0.0


def _parse_date(s: str) -> date | None:
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


def _normalise_record(r: dict, chamber_label: str) -> dict | None:
    """Normalise one trade record from Politician Trade Tracker.

    The API field names are discovered from the first real response.
    Common patterns tried: ticker/symbol, politician/name/senator,
    transaction_date/date/filed_at, type/transaction, amount/range.
    """
    # Symbol
    sym = str(r.get("ticker") or r.get("symbol") or r.get("asset_ticker") or "").strip().upper()
    if not sym or sym in ("N/A", "--", "NONE") or len(sym) > 5:
        return None
    if sym in KNOWN_ETFS:
        return None

    # Transaction type — keep only purchases
    txn = str(r.get("trade_type") or r.get("type") or r.get("transaction") or "").lower()
    if "purchase" not in txn and "buy" not in txn:
        return None

    # Politician name
    name = str(
        r.get("name") or r.get("politician") or r.get("senator") or
        r.get("representative") or r.get("member") or ""
    ).strip()

    # Date — use trade_date; days_until_disclosure field tells us the lag
    date_str = str(
        r.get("trade_date") or r.get("disclosure_date") or
        r.get("transaction_date") or r.get("date") or ""
    )
    disc_dt = _parse_date(date_str)
    if not disc_dt:
        return None

    # Amount
    amount_str = str(r.get("trade_amount") or r.get("amount") or r.get("range") or "")
    amount_mid = _parse_amount(amount_str)

    return {
        "symbol":          sym,
        "member":          name,
        "chamber":         chamber_label,
        "transaction_date": disc_dt,
        "disclosure_date": disc_dt,
        "amount_mid":      amount_mid,
    }


def load_disclosures(api_key: str, chamber: str = "both") -> pd.DataFrame:
    chambers = []
    if chamber in ("senate", "both"):
        chambers.append(("Senate", "senate"))
    if chamber in ("house", "both"):
        chambers.append(("House", "house"))

    rows = []
    for api_chamber, label in chambers:
        raw = _fetch_chamber(api_key, api_chamber)
        if raw:
            # Print first record so we can see the actual field names
            print(f"  Sample {label} record keys: {list(raw[0].keys())[:12]}", flush=True)
        for r in raw:
            norm = _normalise_record(r, label)
            if norm:
                rows.append(norm)

    if not rows:
        raise RuntimeError(
            "No buy disclosures found. Check the sample record keys above "
            "and update _normalise_record() if field names differ."
        )
    df = pd.DataFrame(rows)
    df = df[df["amount_mid"] >= MIN_TRADE_USD].copy()
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    df = df.sort_values("disclosure_date")
    senate_n = (df["chamber"] == "senate").sum()
    house_n  = (df["chamber"] == "house").sum()
    print(f"  Senate buys: {senate_n}  House buys: {house_n}", flush=True)
    return df


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Return daily close prices for all symbols plus SPY."""
    all_syms = list(set(symbols) | {"SPY"})
    print(f"  Downloading prices for {len(all_syms)} symbols ({start} → {end}) …", flush=True)
    raw = yf.download(
        all_syms,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": all_syms[0]})
    return closes


def _nearest_price(prices: pd.DataFrame, symbol: str, target: date, tolerance_days: int = 5) -> float | None:
    if symbol not in prices.columns:
        return None
    col = prices[symbol].dropna()
    if col.empty:
        return None
    target_ts = pd.Timestamp(target)
    # Forward-fill up to tolerance_days (skip weekends/holidays)
    for delta in range(tolerance_days + 1):
        ts = target_ts + pd.offsets.BDay(delta)
        if ts in col.index:
            return float(col.loc[ts])
    return None


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, prices: pd.DataFrame, from_year: int = 2019) -> pd.DataFrame:
    cutoff = pd.Timestamp(f"{from_year}-01-01")
    df = df[df["disclosure_date"] >= cutoff].copy()

    results = []
    for _, row in df.iterrows():
        sym   = row["symbol"]
        disc  = row["disclosure_date"].date()

        # Entry: disclosure_date + ENTRY_LAG_DAYS business days
        entry_date = (pd.Timestamp(disc) + pd.offsets.BDay(ENTRY_LAG_DAYS)).date()
        exit_date  = entry_date + timedelta(days=HOLD_DAYS)

        # Skip if too recent (no exit price yet)
        if exit_date > date.today() - timedelta(days=1):
            continue

        entry_px = _nearest_price(prices, sym,   entry_date)
        exit_px  = _nearest_price(prices, sym,   exit_date)
        spy_entry = _nearest_price(prices, "SPY", entry_date)
        spy_exit  = _nearest_price(prices, "SPY", exit_date)

        if not all([entry_px, exit_px, spy_entry, spy_exit]):
            continue
        if entry_px < MIN_PRICE:
            continue

        # Apply stop loss
        exit_reason = "hold"
        # Simple stop: if stock fell more than STOP_PCT from entry, cap loss
        raw_ret = (exit_px - entry_px) / entry_px
        if raw_ret < -STOP_PCT:
            exit_px    = entry_px * (1 - STOP_PCT)
            exit_reason = "stop"
            raw_ret    = -STOP_PCT

        spy_ret  = (spy_exit  - spy_entry)  / spy_entry
        alpha    = raw_ret - spy_ret

        results.append({
            "member":        row["member"],
            "chamber":       row["chamber"],
            "symbol":        sym,
            "disclosure_date": disc,
            "entry_date":    entry_date,
            "exit_date":     exit_date,
            "entry_px":      round(entry_px, 2),
            "exit_px":       round(exit_px,  2),
            "return_pct":    round(raw_ret * 100, 2),
            "spy_return_pct": round(spy_ret * 100, 2),
            "alpha_pct":     round(alpha * 100, 2),
            "exit_reason":   exit_reason,
            "amount_mid":    row["amount_mid"],
        })

    return pd.DataFrame(results)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_members(results: pd.DataFrame, min_trades: int = 5) -> pd.DataFrame:
    rows = []
    for member, grp in results.groupby("member"):
        if len(grp) < min_trades:
            continue
        wins    = (grp["return_pct"] > 0).sum()
        losses  = (grp["return_pct"] < 0).sum()
        rows.append({
            "member":       member,
            "chamber":      grp["chamber"].iloc[0],
            "n_trades":     len(grp),
            "win_rate":     round(wins / len(grp) * 100, 1),
            "avg_return":   round(grp["return_pct"].mean(), 2),
            "avg_alpha":    round(grp["alpha_pct"].mean(),  2),
            "median_alpha": round(grp["alpha_pct"].median(), 2),
            "total_alpha":  round(grp["alpha_pct"].sum(), 2),
            "stops_pct":    round((grp["exit_reason"] == "stop").sum() / len(grp) * 100, 1),
            "best_trade":   round(grp["return_pct"].max(), 1),
            "worst_trade":  round(grp["return_pct"].min(), 1),
        })
    return (pd.DataFrame(rows)
            .sort_values("avg_alpha", ascending=False)
            .reset_index(drop=True))


# ── Portfolio simulation ──────────────────────────────────────────────────────

def simulate_portfolio(results: pd.DataFrame, top_members: list[str],
                        position_size: float = SIM_POSITION) -> dict:
    filtered = results[results["member"].isin(top_members)].copy()
    filtered = filtered.sort_values("entry_date")

    total_invested = 0.0
    total_pnl      = 0.0
    n_trades       = 0
    wins           = 0

    for _, row in filtered.iterrows():
        qty  = int(position_size // row["entry_px"])
        if qty < 1:
            continue
        pnl  = qty * (row["exit_px"] - row["entry_px"])
        total_invested += qty * row["entry_px"]
        total_pnl      += pnl
        n_trades       += 1
        if pnl > 0:
            wins += 1

    return {
        "n_trades":      n_trades,
        "win_rate":      round(wins / n_trades * 100, 1) if n_trades else 0,
        "total_pnl":     round(total_pnl, 2),
        "avg_pnl_trade": round(total_pnl / n_trades, 2) if n_trades else 0,
        "total_invested": round(total_invested, 2),
        "total_return_pct": round(total_pnl / total_invested * 100, 2) if total_invested else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Senator trade disclosure backtest")
    parser.add_argument("--api-key", default=os.environ.get("RAPIDAPI_KEY", ""),
                        help="RapidAPI key for Politician Trade Tracker "
                             "(or set RAPIDAPI_KEY env var). "
                             "Free tier at https://www.politiciantradetracker.us/")
    parser.add_argument("--top",     type=int, default=TOP_N_DEFAULT, help="Show top N members")
    parser.add_argument("--from",    dest="from_year", type=int, default=2020)
    parser.add_argument("--chamber", choices=["senate","house","both"], default="both")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--save",    action="store_true", help="Save results to CSV")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Senator/Rep Trade Disclosure Backtest")
    print(f"  Entry lag: {ENTRY_LAG_DAYS} trading days after disclosure")
    print(f"  Hold: {HOLD_DAYS} days  |  Stop: {STOP_PCT*100:.0f}%  |  From: {args.from_year}")
    print(f"{'='*65}\n")

    # ── 1. Load disclosures ───────────────────────────────────────────────────
    if not args.api_key:
        print("\nERROR: RapidAPI key required.")
        print("  1. Sign up free at https://www.politiciantradetracker.us/")
        print("  2. Run:  venv/bin/python backtest.py --api-key YOUR_KEY --save")
        print("  Or set env:  export RAPIDAPI_KEY=YOUR_KEY")
        sys.exit(1)

    print("Step 1: Loading congressional disclosures…")
    df = load_disclosures(args.api_key, chamber=args.chamber)
    print(f"  {len(df)} qualifying buy disclosures (≥${MIN_TRADE_USD:,})\n")

    # ── 2. Fetch prices ───────────────────────────────────────────────────────
    print("Step 2: Fetching historical prices…")
    start_str = f"{args.from_year - 1}-06-01"
    end_str   = date.today().isoformat()
    symbols   = df["symbol"].unique().tolist()
    prices    = fetch_prices(symbols, start_str, end_str)
    print(f"  {prices.shape[1]} symbols, {len(prices)} days\n")

    # ── 3. Run backtest ───────────────────────────────────────────────────────
    print("Step 3: Running backtest…")
    results = run_backtest(df, prices, from_year=args.from_year)
    print(f"  {len(results)} completed trades\n")

    if results.empty:
        print("No results — check data or date range.")
        return

    # ── 4. Score members ──────────────────────────────────────────────────────
    print("Step 4: Scoring members (min {} trades)…\n".format(args.min_trades))
    scores = score_members(results, min_trades=args.min_trades)

    print(f"{'='*65}")
    print(f"  TOP {args.top} MEMBERS BY AVERAGE ALPHA (vs SPY)")
    print(f"{'='*65}")
    top_scores = scores.head(args.top)
    print(top_scores[[
        "member","chamber","n_trades","win_rate",
        "avg_return","avg_alpha","median_alpha","stops_pct"
    ]].to_string(index=True))

    print(f"\n{'='*65}")
    print(f"  BOTTOM 10 (avoid these)")
    print(f"{'='*65}")
    print(scores.tail(10)[[
        "member","chamber","n_trades","avg_alpha"
    ]].to_string(index=True))

    # ── 5. Portfolio simulation ───────────────────────────────────────────────
    top_members = top_scores.head(args.top)["member"].tolist()
    sim = simulate_portfolio(results, top_members, SIM_POSITION)

    print(f"\n{'='*65}")
    print(f"  PORTFOLIO SIMULATION: Top {args.top} members, ${SIM_POSITION:,}/position")
    print(f"{'='*65}")
    print(f"  Trades:          {sim['n_trades']}")
    print(f"  Win rate:        {sim['win_rate']}%")
    print(f"  Total PnL:       ${sim['total_pnl']:,.2f}")
    print(f"  Avg per trade:   ${sim['avg_pnl_trade']:,.2f}")
    print(f"  Total invested:  ${sim['total_invested']:,.2f}")
    print(f"  Total return:    {sim['total_return_pct']:.2f}%")

    # ── 6. Overall stats ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  OVERALL STATS (all members, {args.from_year}–present)")
    print(f"{'='*65}")
    print(f"  Total trades:     {len(results)}")
    print(f"  Avg return:       {results['return_pct'].mean():.2f}%")
    print(f"  Avg alpha vs SPY: {results['alpha_pct'].mean():.2f}%")
    print(f"  Win rate:         {(results['return_pct'] > 0).mean()*100:.1f}%")
    print(f"  Stopped out:      {(results['exit_reason']=='stop').sum()} ({(results['exit_reason']=='stop').mean()*100:.1f}%)")

    print(f"\n{'='*65}")
    print(f"  TOP 20 SYMBOLS TRADED BY ALPHA MEMBERS")
    print(f"{'='*65}")
    top_trades = (results[results["member"].isin(top_members)]
                  .groupby("symbol")
                  .agg(n=("return_pct","count"),
                       avg_ret=("return_pct","mean"),
                       avg_alpha=("alpha_pct","mean"))
                  .sort_values("avg_alpha", ascending=False)
                  .head(20))
    print(top_trades.round(2).to_string())

    if args.save:
        results.to_csv("backtest_results.csv", index=False)
        scores.to_csv("senator_scores.csv", index=False)
        print("\nSaved: backtest_results.csv, senator_scores.csv")


if __name__ == "__main__":
    main()
