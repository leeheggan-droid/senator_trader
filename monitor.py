#!/usr/bin/env python3
"""Senator Trade Monitor — daily live trading service.

Checks for new congressional buy disclosures from the watchlist,
enters positions 7 trading days after disclosure, and exits after
30 days or 10% stop loss.

Run daily via systemd timer. All state persisted in data/.

Config (environment variables):
  RAPIDAPI_KEY          Politician Trade Tracker key (RapidAPI)
  ALPACA_API_KEY        Alpaca API key
  ALPACA_SECRET_KEY     Alpaca secret key
  ALPACA_PAPER=true     Use paper trading account (default: true)
  TELEGRAM_BOT_TOKEN    Optional: Telegram notifications
  TELEGRAM_CHAT_ID      Optional: Telegram chat ID

Usage (manual test):
  python monitor.py [--dry-run]
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
ENTRY_LAG_DAYS   = 7       # trading days after disclosure before entry
HOLD_DAYS        = 30      # calendar days to hold
STOP_PCT         = 0.10    # stop loss from entry
POSITION_SIZE    = 10_000  # $ per trade

RAPIDAPI_HOST      = "politician-trade-tracker1.p.rapidapi.com"
TRADES_BY_TYPE_URL = f"https://{RAPIDAPI_HOST}/trades/type"

DATA_DIR          = Path("data")
POSITIONS_FILE    = DATA_DIR / "positions.json"
SEEN_FILE         = DATA_DIR / "seen_disclosures.json"
TRADE_LOG_FILE    = DATA_DIR / "trade_log.jsonl"
CACHE_DIR         = DATA_DIR / "cache"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

KNOWN_ETFS = frozenset({
    "SPY","QQQ","IVV","VOO","VTI","GLD","TLT","IEF","HYG","LQD",
    "XLF","XLE","XLK","XLV","DIA","MDY","IWM","EEM","EFA","VWO",
    "AGG","BND","VUG","VTV","IAU","GDX","TQQQ","SQQQ","UPRO","SH",
})

WATCHLIST_NAMES = [
    "nancy pelosi",
    "tommy tuberville",
    "michael mccaul",
    "ro khanna",
    "markwayne mullin",
    "josh gottheimer",
    "dan crenshaw",
    "debbie wasserman schultz",
    "kevin hern",
    "ron wyden",
]


def _on_watchlist(name: str) -> bool:
    n = name.lower().strip()
    return any(w in n or n in w for w in WATCHLIST_NAMES)


# ── State helpers ─────────────────────────────────────────────────────────────

def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_positions() -> list[dict]:
    return _load_json(POSITIONS_FILE, [])


def save_positions(positions: list[dict]) -> None:
    _save_json(POSITIONS_FILE, positions)


def load_seen() -> set[str]:
    return set(_load_json(SEEN_FILE, []))


def save_seen(seen: set[str]) -> None:
    _save_json(SEEN_FILE, sorted(seen))


def _seen_key(member: str, symbol: str, trade_date: str) -> str:
    return f"{member.lower()}|{symbol.upper()}|{trade_date}"


def _log_trade(event: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRADE_LOG_FILE, "a") as f:
        f.write(json.dumps({**event, "logged_at": datetime.utcnow().isoformat()}) + "\n")


# ── Notifications ─────────────────────────────────────────────────────────────

def notify(msg: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        log.warning("Telegram failed: %s", exc)


# ── Alpaca ────────────────────────────────────────────────────────────────────

def _alpaca_client():
    from alpaca.trading.client import TradingClient
    key    = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET_KEY"]
    paper  = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    return TradingClient(key, secret, paper=paper)


def _get_price(tc, symbol: str) -> Optional[float]:
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestBarRequest
        dc  = StockHistoricalDataClient(
            os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
        bar = dc.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=symbol))
        return float(bar[symbol].c) if bar and symbol in bar else None
    except Exception as exc:
        log.warning("Price fetch failed for %s: %s", symbol, exc)
        return None


def place_buy(tc, symbol: str, notional: float, dry_run: bool = False) -> Optional[dict]:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    import uuid

    price = _get_price(tc, symbol)
    if not price or price <= 0:
        log.warning("Cannot buy %s — no price", symbol)
        return None
    qty = max(1, int(notional // price))
    oid = f"SEN-{uuid.uuid4().hex[:10]}"

    if dry_run:
        log.info("[DRY RUN] Would buy %s qty=%d @ ~$%.2f (notional $%,.0f)", symbol, qty, price, qty * price)
        return {"order_id": oid, "qty": qty, "entry_px": price, "dry_run": True}

    try:
        req = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            client_order_id=oid,
        )
        tc.submit_order(req)
        log.info("BUY %s qty=%d @ ~$%.2f oid=%s", symbol, qty, price, oid)
        return {"order_id": oid, "qty": qty, "entry_px": price}
    except Exception as exc:
        log.error("Buy failed for %s: %s", symbol, exc)
        return None


def place_sell(tc, symbol: str, qty: int, reason: str, dry_run: bool = False) -> bool:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    import uuid

    if dry_run:
        log.info("[DRY RUN] Would sell %s qty=%d reason=%s", symbol, qty, reason)
        return True
    try:
        req = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            client_order_id=f"SEN-X-{uuid.uuid4().hex[:8]}",
        )
        tc.submit_order(req)
        log.info("SELL %s qty=%d reason=%s", symbol, qty, reason)
        return True
    except Exception as exc:
        log.error("Sell failed for %s: %s", symbol, exc)
        return False


# ── Disclosure feed ───────────────────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    try:
        return pd.to_datetime(s.strip()).date()
    except Exception:
        return None


def fetch_recent_disclosures(api_key: str) -> list[dict]:
    """Fetch today's congressional buys. Cached per calendar day."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today      = date.today().isoformat()
    cache_file = CACHE_DIR / f"monitor_{today}.json"

    if cache_file.exists():
        data = _load_json(cache_file, [])
        if data:
            log.info("Using today's cached disclosures (%d records)", len(data))
            return data

    log.info("Fetching latest congressional buys …")
    headers = {
        "User-Agent":      "senator-monitor/1.0",
        "Accept":          "application/json",
        "x-rapidapi-key":  api_key,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    try:
        resp = requests.get(
            TRADES_BY_TYPE_URL,
            headers=headers,
            params={"trade_type": "buy"},
            timeout=30,
        )
        if resp.status_code == 429:
            log.warning("API rate limit hit — using stale cache if available")
            stale = sorted(CACHE_DIR.glob("monitor_*.json"), reverse=True)
            if stale:
                return _load_json(stale[0], [])
            return []
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data = data.get("trades", data.get("data", []))
        if isinstance(data, list):
            _save_json(cache_file, data)
            log.info("Fetched %d disclosure records", len(data))
            return data
    except Exception as exc:
        log.error("Fetch failed: %s", exc)
    return []


def get_new_watchlist_signals(api_key: str, seen: set[str]) -> list[dict]:
    """Return unseen buy disclosures from watchlist that are ≥ ENTRY_LAG_DAYS old."""
    raw     = fetch_recent_disclosures(api_key)
    today   = date.today()
    signals = []

    for r in raw:
        name = str(r.get("name", "") or r.get("politician", "")).strip()
        if not _on_watchlist(name):
            continue

        sym = str(r.get("ticker", "") or r.get("symbol", "")).strip().upper()
        if not sym or sym in KNOWN_ETFS or len(sym) > 5 or sym in ("N/A", "--"):
            continue

        txn = str(r.get("trade_type", "") or r.get("type", "")).lower()
        if "purchase" not in txn and "buy" not in txn:
            continue

        trade_date_str = str(r.get("trade_date", "") or r.get("disclosure_date", ""))
        trade_dt = _parse_date(trade_date_str)
        if not trade_dt:
            continue

        key = _seen_key(name, sym, trade_date_str)
        if key in seen:
            continue

        # Check entry lag: need ≥ ENTRY_LAG_DAYS trading days since disclosure
        bdays_since = len(pd.bdate_range(trade_dt, today)) - 1
        if bdays_since < ENTRY_LAG_DAYS:
            log.info("Not ready yet: %s %s (disclosed %s, %d/%d bdays)",
                     name, sym, trade_dt, bdays_since, ENTRY_LAG_DAYS)
            continue

        signals.append({
            "member":      name,
            "symbol":      sym,
            "trade_date":  trade_date_str,
            "seen_key":    key,
        })

    return signals


# ── Exit monitoring ───────────────────────────────────────────────────────────

def check_exits(tc, positions: list[dict], dry_run: bool = False) -> list[dict]:
    """Check each open position for time or stop exit. Returns updated positions."""
    today    = date.today()
    remaining = []

    for pos in positions:
        symbol   = pos["symbol"]
        qty      = pos.get("qty", 0)
        entry_px = pos.get("entry_px", 0)
        stop_px  = entry_px * (1 - STOP_PCT)
        target   = _parse_date(pos.get("target_exit_date", ""))

        if qty <= 0:
            continue

        current_px = _get_price(tc, symbol)

        # Time exit
        if target and today >= target:
            log.info("TIME EXIT %s (held 30 days)", symbol)
            ok = place_sell(tc, symbol, qty, "TIME_EXIT", dry_run=dry_run)
            if ok:
                pnl = (current_px - entry_px) * qty if current_px else 0
                notify(f"📤 TIME EXIT {symbol} qty={qty} "
                       f"entry=${entry_px:.2f} "
                       f"current≈${current_px:.2f if current_px else 0:.2f} "
                       f"PnL≈${pnl:+.0f}\n"
                       f"Senator: {pos.get('member', '')}")
                _log_trade({**pos, "event": "exit", "reason": "TIME_EXIT",
                             "exit_px": current_px, "pnl": pnl})
                continue

        # Stop loss
        if current_px and current_px <= stop_px:
            log.info("STOP EXIT %s price=$%.2f stop=$%.2f", symbol, current_px, stop_px)
            ok = place_sell(tc, symbol, qty, "STOP", dry_run=dry_run)
            if ok:
                pnl = (current_px - entry_px) * qty
                notify(f"🛑 STOP EXIT {symbol} qty={qty} "
                       f"entry=${entry_px:.2f} stop=${stop_px:.2f} "
                       f"PnL≈${pnl:+.0f}\n"
                       f"Senator: {pos.get('member', '')}")
                _log_trade({**pos, "event": "exit", "reason": "STOP",
                             "exit_px": current_px, "pnl": pnl})
                continue

        remaining.append(pos)

    return remaining


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Check signals and exits but don't place orders")
    args = parser.parse_args()

    rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
    if not rapidapi_key:
        log.error("RAPIDAPI_KEY not set")
        sys.exit(1)
    if not os.environ.get("ALPACA_API_KEY"):
        log.error("ALPACA_API_KEY not set")
        sys.exit(1)

    dry = args.dry_run
    if dry:
        log.info("=== DRY RUN — no orders will be placed ===")

    tc        = _alpaca_client()
    positions = load_positions()
    seen      = load_seen()

    log.info("Open positions: %d", len(positions))

    # ── 1. Check exits ─────────────────────────────────────────────────────
    positions = check_exits(tc, positions, dry_run=dry)

    # ── 2. Check for new signals ───────────────────────────────────────────
    signals = get_new_watchlist_signals(rapidapi_key, seen)
    log.info("New signals: %d", len(signals))

    held_symbols = {p["symbol"] for p in positions}

    for sig in signals:
        sym    = sig["symbol"]
        member = sig["member"]

        if sym in held_symbols:
            log.info("Already holding %s — skipping", sym)
            seen.add(sig["seen_key"])
            continue

        log.info("SIGNAL: %s buying %s", member, sym)
        result = place_buy(tc, sym, POSITION_SIZE, dry_run=dry)

        if result:
            entry_date = date.today()
            pos = {
                "symbol":           sym,
                "member":           member,
                "trade_date":       sig["trade_date"],
                "entry_date":       entry_date.isoformat(),
                "entry_px":         result["entry_px"],
                "qty":              result["qty"],
                "stop_px":          round(result["entry_px"] * (1 - STOP_PCT), 2),
                "target_exit_date": (entry_date + timedelta(days=HOLD_DAYS)).isoformat(),
            }
            positions.append(pos)
            held_symbols.add(sym)
            seen.add(sig["seen_key"])
            _log_trade({**pos, "event": "entry"})
            notify(f"📈 SENATOR BUY\n"
                   f"{member}\n"
                   f"{sym} qty={result['qty']} @ ~${result['entry_px']:.2f}\n"
                   f"Exit target: {pos['target_exit_date']}\n"
                   f"Stop: ${pos['stop_px']:.2f}")
        else:
            seen.add(sig["seen_key"])

    # ── 3. Save state ──────────────────────────────────────────────────────
    save_positions(positions)
    save_seen(seen)

    # ── 4. Daily summary ───────────────────────────────────────────────────
    log.info("Open positions after run: %d", len(positions))
    for p in positions:
        px = _get_price(tc, p["symbol"]) or p["entry_px"]
        pnl = (px - p["entry_px"]) * p["qty"]
        log.info("  %s: entry=$%.2f current≈$%.2f qty=%d PnL≈$%+.0f (exit %s)",
                 p["symbol"], p["entry_px"], px, p["qty"], pnl, p["target_exit_date"])

    notify(f"📊 Senator Monitor — {date.today()}\n"
           f"Open positions: {len(positions)}\n"
           f"New entries today: {sum(1 for s in signals if s['seen_key'] in seen)}")


if __name__ == "__main__":
    main()
