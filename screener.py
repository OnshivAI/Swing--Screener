"""
NSE Swing Screener - signal-only system (NO order placement).

Daily flow:
  1. Load universe (universe.csv from NSE index constituent list)
  2. Download ~1 year of daily prices (yfinance, bulk)
  3. Evaluate previously issued signals as paper trades (target / stop / time exit)
  4. Momentum gate on all stocks (cheap, local calculation)
  5. Fundamental gate ONLY on momentum survivors (few yfinance .info calls, cached)
  6. Build trade cards (ATR stop, fixed % target, risk cap, cost-adjusted)
  7. Log signals, save report, send Telegram message

Run:  python screener.py            (normal daily run)
      python screener.py --force    (generate signals even if today's bar is missing)

IMPORTANT: Signals only. Not investment advice. Rules are NOT backtested.
Use the paper-trade record for several weeks before risking real money.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # allows offline testing with a mock
    yf = None

# --------------------------------------------------------------------------
# CONFIG - tune here, nowhere else
# --------------------------------------------------------------------------
CONFIG = {
    "universe_file": "universe.csv",
    "price_period": "2y",           # 2 years: enough history for the backtest
    "download_batch": 50,
    "fundamentals_cache_days": 7,

    # Fundamental gate (yfinance .info fields; coverage for Indian stocks is patchy)
    "max_debt_to_equity": 100,       # yfinance appears to report D/E x100 (100 = 1.0x) - verify
    "min_profit_margin": 0.05,       # 5%
    "min_roe": 0.12,                 # 12%
    "max_trailing_pe": 60,
    "skip_de_for_sectors": ["Financial Services"],  # D/E not meaningful for banks/NBFCs
    "missing_fundamentals": "flag",   # "flag" = keep pick, mark unverified | "reject"
    "require_debt_to_equity": False,  # Yahoo often omits D/E (e.g. for low/no-debt firms)

    # Momentum gate
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_min": 55,
    "rsi_max": 68,
    "volume_mult": 1.5,              # today's volume vs prior 20-day average
    "min_avg_turnover_cr": 20,       # 20-day avg traded value, Rs crore (liquidity floor)

    # Trade card
    "atr_period": 14,
    "atr_stop_mult": 1.5,
    "target_pct": 2.5,
    "max_risk_pct": 2.0,             # reject setups whose stop is wider than this
    "round_trip_cost_pct": 0.20,     # APPROXIMATE - confirm with your broker's contract notes
    "max_hold_days": 8,              # time exit after N sessions
    "top_n": 2,

    # Reporting / backtest
    "recent_sessions": 10,           # picks table covers the last N trading sessions
    "backtest_warmup": 60,           # skip first N bars while indicators settle

    # Scheduling: runs before this IST hour stay silent if data is not ready (a later run retries)
    "final_attempt_hour_ist": 20,
    "data_ready_hour_ist": 16,       # before this hour, report on the previous trading day
}

IST = timezone(timedelta(hours=5, minutes=30))
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
REPORTS = DATA / "reports"
SIGNAL_LOG = DATA / "signals.csv"
FUND_CACHE = DATA / "fundamentals.csv"
MARKER = DATA / "last_report_date.txt"

LOG_COLUMNS = [
    "signal_date", "ticker", "ref_close", "stop", "target", "risk_pct", "target_pct",
    "rsi", "vol_ratio", "score", "status", "entry_date", "entry_price",
    "exit_date", "exit_price", "net_result_pct", "fund_status",
]
FUND_FIELDS = ["debtToEquity", "profitMargins", "returnOnEquity",
               "trailingPE", "forwardPE", "sector"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("screener")


# --------------------------------------------------------------------------
# Indicators (plain pandas - no third-party TA library needed)
# --------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(avg_gain.notna())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_universe() -> list:
    f = BASE / CONFIG["universe_file"]
    if not f.exists():
        sys.exit(f"{f.name} not found. Download the index constituent CSV from NSE and save it here.")
    df = pd.read_csv(f)
    col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
    if col is None:
        sys.exit(f"{f.name} must have a 'Symbol' column.")
    syms = df[col].astype(str).str.strip().str.upper()
    syms = [s for s in syms.unique() if s and s != "NAN"]
    return [s if s.endswith(".NS") else f"{s}.NS" for s in syms]


def get_prices(tickers: list) -> dict:
    """Bulk daily OHLCV download. Returns {ticker: DataFrame}."""
    frames = {}
    step = CONFIG["download_batch"]
    for i in range(0, len(tickers), step):
        batch = tickers[i:i + step]
        try:
            raw = yf.download(batch, period=CONFIG["price_period"], interval="1d",
                              group_by="ticker", auto_adjust=True, threads=True,
                              progress=False)
        except Exception as e:
            log.warning("Download failed for batch %d: %s", i // step + 1, e)
            continue
        if raw is None or raw.empty:
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        for t in batch:
            try:
                if multi:
                    if t not in raw.columns.get_level_values(0):
                        continue
                    df = raw[t]
                else:
                    df = raw
                df = df.dropna(subset=["Close"])
                if len(df) >= 60:
                    frames[t] = df.copy()
            except Exception as e:
                log.debug("Skipping %s: %s", t, e)
        time.sleep(1)
    log.info("Price data received for %d of %d tickers", len(frames), len(tickers))
    return frames


def _statement_row(df, names):
    """Return a date-indexed, newest-first Series for the first matching row label, else None."""
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            if len(s):
                s.index = pd.to_datetime(s.index, errors="coerce")
                return s[s.index.notna()].sort_index(ascending=False)
    return None


def _same_year_ratio(num, den, positive_den=True):
    """num/den using the most recent fiscal year present in BOTH series."""
    if num is None or den is None:
        return None
    common = [d for d in num.index if d in den.index]
    if not common:
        return None
    d = max(common)
    n, m = float(num[d]), float(den[d])
    if m == 0 or (positive_den and m < 0):
        return None
    return n / m


# Row labels vary between companies/yfinance versions - first match wins (verify on real data)
ROW_NET_INCOME = ["Net Income", "Net Income Common Stockholders",
                  "Net Income From Continuing Operation Net Minority Interest"]
ROW_REVENUE = ["Total Revenue", "Operating Revenue"]
ROW_EQUITY = ["Stockholders Equity", "Common Stock Equity",
              "Total Equity Gross Minority Interest"]
ROW_DEBT = ["Total Debt"]


def statement_ratios(tk) -> dict:
    """Calculate margin, ROE and D/E from annual statements (layer 2 fallback)."""
    try:
        inc, bs = tk.financials, tk.balance_sheet
    except Exception as e:
        log.debug("Statements unavailable: %s", e)
        return {}
    ni = _statement_row(inc, ROW_NET_INCOME)
    rev = _statement_row(inc, ROW_REVENUE)
    eq = _statement_row(bs, ROW_EQUITY)
    debt = _statement_row(bs, ROW_DEBT)
    out = {"profitMargins": _same_year_ratio(ni, rev),
           "returnOnEquity": _same_year_ratio(ni, eq)}
    de = _same_year_ratio(debt, eq)
    out["debtToEquity"] = None if de is None else de * 100   # same x100 convention as .info
    return {k: v for k, v in out.items() if v is not None}


def get_fundamentals(tickers: list) -> pd.DataFrame:
    """Layer 1: yfinance .info ratios. Layer 2: calculate missing ones from statements.
    Results cached for N days."""
    today = datetime.now(IST).date()
    cols = ["ticker", "fetched_at", "src"] + FUND_FIELDS
    cache = {}
    if FUND_CACHE.exists():
        old = pd.read_csv(FUND_CACHE)
        if "src" in old.columns:                     # ignore caches from older script versions
            cache = {r["ticker"]: r for r in old.to_dict("records")}

    for t in tickers:
        if t in cache:
            fetched = pd.to_datetime(cache[t]["fetched_at"]).date()
            if (today - fetched).days < CONFIG["fundamentals_cache_days"]:
                continue
        tk = yf.Ticker(t)
        try:
            info = tk.info or {}
        except Exception as e:
            log.warning("Fundamentals fetch failed for %s: %s", t, e)
            info = {}
        row = {k: info.get(k) for k in FUND_FIELDS}
        src = "ratios"
        need = [k for k in ("profitMargins", "returnOnEquity", "debtToEquity") if row.get(k) is None]
        if need:
            calc = statement_ratios(tk)
            filled = [k for k in need if k in calc]
            for k in filled:
                row[k] = calc[k]
            if filled:
                src = "calculated" if len(filled) == 3 else "ratios+calculated"
                log.info("%s: calculated from statements: %s", t, ", ".join(filled))
        if any(v is not None for v in row.values()):  # only cache real data
            cache[t] = {"ticker": t, "fetched_at": today.isoformat(), "src": src, **row}
        time.sleep(0.5)

    df = pd.DataFrame(list(cache.values()), columns=cols)
    df.to_csv(FUND_CACHE, index=False)
    return df.set_index("ticker")


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """All momentum indicators as full time series (causal - each row uses only past data).
    Used both for today's screen and for the backtest, so both apply identical rules."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    out["e_fast"] = ema(c, CONFIG["ema_fast"])
    out["e_slow"] = ema(c, CONFIG["ema_slow"])
    out["rsi"] = rsi(c, CONFIG["rsi_period"])
    out["atr"] = atr(h, l, c, CONFIG["atr_period"])
    vol_avg_prior = v.shift(1).rolling(20).mean()          # prior 20 sessions, excludes today
    out["vol_ratio"] = v / vol_avg_prior.where(vol_avg_prior > 0)
    out["turnover_cr"] = (c * v).rolling(20).mean() / 1e7
    out["signal"] = (
        (c > out["e_fast"]) & (out["e_fast"] > out["e_slow"])
        & out["rsi"].between(CONFIG["rsi_min"], CONFIG["rsi_max"])
        & (out["vol_ratio"] >= CONFIG["volume_mult"])
        & (out["turnover_cr"] >= CONFIG["min_avg_turnover_cr"])
    ).fillna(False)
    out["score"] = (out["rsi"] - 50) + 10 * out["vol_ratio"].clip(upper=4.0)
    return out


def momentum_signal(df: pd.DataFrame):
    ind = compute_indicators(df)
    last = ind.iloc[-1]
    if not bool(last["signal"]) or pd.isna(last["atr"]):
        return None
    return {"close": float(last["close"]), "rsi": float(last["rsi"]), "atr": float(last["atr"]),
            "vol_ratio": float(last["vol_ratio"]), "score": float(last["score"])}


# --------------------------------------------------------------------------
# Backtest (price rules only - fundamentals cannot be tested without look-ahead)
# --------------------------------------------------------------------------
def simulate_trades(df: pd.DataFrame, ind: pd.DataFrame = None) -> list:
    """Replays the exact daily rules on past data: signal at close, entry next open,
    exit at stop / target / time. One trade at a time per stock. Returns net % results."""
    if ind is None:
        ind = compute_indicators(df)
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    sig, atr_v = ind["signal"].to_numpy(), ind["atr"].to_numpy()
    cost, n = CONFIG["round_trip_cost_pct"], len(df)
    trades, busy_until = [], -1
    for i in range(CONFIG["backtest_warmup"], n - 1):
        if i <= busy_until or not sig[i] or np.isnan(atr_v[i]):
            continue
        stop = c[i] - CONFIG["atr_stop_mult"] * atr_v[i]
        if (c[i] - stop) / c[i] * 100 > CONFIG["max_risk_pct"]:
            continue                                  # same risk cap as live trade cards
        target = c[i] * (1 + CONFIG["target_pct"] / 100)
        entry = o[i + 1]
        if entry >= target or entry <= stop:
            continue                                  # gap through a level -> no trade
        exit_px, outcome, j = None, None, i + 1
        for j in range(i + 1, n):
            if l[j] <= stop:
                exit_px, outcome = stop, "STOP"
                break
            if h[j] >= target:
                exit_px, outcome = target, "TARGET"
                break
            if j - i >= CONFIG["max_hold_days"]:
                exit_px, outcome = c[j], "TIME_EXIT"
                break
        if exit_px is None:
            break                                     # trade still open at end of data
        trades.append({"date": df.index[i].date(), "outcome": outcome,
                       "net_pct": (exit_px - entry) / entry * 100 - cost})
        busy_until = j
    return trades


def summarize_trades(trades: list) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t["net_pct"] for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    return {"n": len(r), "win_rate": (r > 0).mean() * 100, "avg": r.mean(),
            "best": r.max(), "worst": r.min(), "profit_factor": pf,
            "targets": sum(t["outcome"] == "TARGET" for t in trades),
            "stops": sum(t["outcome"] == "STOP" for t in trades),
            "time_exits": sum(t["outcome"] == "TIME_EXIT" for t in trades)}


def backtest_line(s: dict, label: str) -> str:
    if s["n"] == 0:
        return f"{label}: no past signals in the data window"
    pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    return (f"{label}: {s['n']} trades | win {s['win_rate']:.0f}% | avg net {s['avg']:+.2f}% | "
            f"PF {pf} | target/stop/time {s['targets']}/{s['stops']}/{s['time_exits']}")


# --------------------------------------------------------------------------
# Recent picks table
# --------------------------------------------------------------------------
def recent_picks_table(log_df: pd.DataFrame, prices: dict) -> str:
    if log_df.empty:
        return "No picks in the last %d sessions." % CONFIG["recent_sessions"]
    # last N trading dates, taken from the price calendar
    cal = sorted({d.date() for df in prices.values() for d in df.index[-60:]})
    start = cal[-CONFIG["recent_sessions"]] if len(cal) >= CONFIG["recent_sessions"] else cal[0]
    recent = log_df[pd.to_datetime(log_df["signal_date"]).dt.date >= start]
    if recent.empty:
        return "No picks in the last %d sessions." % CONFIG["recent_sessions"]

    rows = [f"{'Date':<7}{'Stock':<12}{'Entry':>9}{'Last/Exit':>11}{'P&L%':>8}  Status"]
    cost = CONFIG["round_trip_cost_pct"]
    for _, r in recent.sort_values("signal_date").iterrows():
        sym = r["ticker"].replace(".NS", "") + ("*" if r.get("fund_status") == "unverified" else "")
        date = pd.to_datetime(r["signal_date"]).strftime("%d-%b")
        entry = pd.to_numeric(r.get("entry_price"), errors="coerce")
        status = r["status"]
        if status in ("STOP", "TARGET", "TIME_EXIT"):
            last = pd.to_numeric(r["exit_price"], errors="coerce")
            pnl = pd.to_numeric(r["net_result_pct"], errors="coerce")
        elif status == "OPEN" and r["ticker"] in prices:
            last = float(prices[r["ticker"]]["Close"].iloc[-1])
            pnl = (last - entry) / entry * 100 - cost       # unrealised, after est. costs
        else:
            last, pnl = np.nan, np.nan
        fmt = lambda x, w, f: "-".rjust(w) if pd.isna(x) else format(float(x), f).rjust(w)
        rows.append(f"{date:<7}{sym[:11]:<12}{fmt(entry, 9, '.2f')}{fmt(last, 11, '.2f')}"
                    f"{fmt(pnl, 8, '+.2f')}  {status}")
    return "\n".join(rows)


def evaluate_fundamentals(row):
    """Returns (status, reason, details). status: 'pass' | 'fail' | 'unverified'."""
    def val(k):
        x = row.get(k) if row is not None else None
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else x

    if row is None:
        return "unverified", "no fundamental data", {}
    sector = val("sector")
    de, pm, roe, pe = (val("debtToEquity"), val("profitMargins"),
                       val("returnOnEquity"), val("trailingPE"))
    details = {"roe": roe, "pm": pm, "de": de, "pe": pe, "src": val("src") or "ratios"}
    skip_de = sector in CONFIG["skip_de_for_sectors"]

    # 1) any value that IS available and breaks a rule -> fail
    if not skip_de and de is not None and float(de) > CONFIG["max_debt_to_equity"]:
        return "fail", "high D/E", details
    if pm is not None and float(pm) < CONFIG["min_profit_margin"]:
        return "fail", "low margin", details
    if roe is not None and float(roe) < CONFIG["min_roe"]:
        return "fail", "low ROE", details
    if pe is not None and (float(pe) <= 0 or float(pe) > CONFIG["max_trailing_pe"]):
        return "fail", "P/E out of range", details

    # 2) required values missing -> unverified (or fail, per config)
    missing = [n for n, x in (("margin", pm), ("ROE", roe)) if x is None]
    if CONFIG["require_debt_to_equity"] and not skip_de and de is None:
        missing.append("D/E")
    if missing:
        status = "unverified" if CONFIG["missing_fundamentals"] == "flag" else "fail"
        return status, "missing " + ", ".join(missing), details
    return "pass", "ok", details


def build_trade(sig: dict):
    entry = sig["close"]
    stop = entry - CONFIG["atr_stop_mult"] * sig["atr"]
    risk_pct = (entry - stop) / entry * 100
    if risk_pct > CONFIG["max_risk_pct"] or risk_pct <= 0:
        return None
    target = entry * (1 + CONFIG["target_pct"] / 100)
    return {"ref_close": round(entry, 2), "stop": round(stop, 2), "target": round(target, 2),
            "risk_pct": round(risk_pct, 2), "target_pct": CONFIG["target_pct"],
            "rr": round(CONFIG["target_pct"] / risk_pct, 2)}


# --------------------------------------------------------------------------
# Paper-trade tracker
# --------------------------------------------------------------------------
def load_signal_log() -> pd.DataFrame:
    if SIGNAL_LOG.exists():
        text_cols = ["signal_date", "ticker", "status", "entry_date", "exit_date"]
        return pd.read_csv(SIGNAL_LOG, dtype={c: "object" for c in text_cols})
    return pd.DataFrame(columns=LOG_COLUMNS)


def evaluate_open(log_df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Paper entry = next session's open. Same-day stop+target touch counts as stop (conservative)."""
    cost = CONFIG["round_trip_cost_pct"]
    for idx, s in log_df[log_df["status"].isin(["PENDING", "OPEN"])].iterrows():
        df = prices.get(s["ticker"])
        if df is None:
            continue
        sig_day = pd.to_datetime(s["signal_date"]).date()
        after = df[[d.date() > sig_day for d in df.index]]
        if after.empty:
            continue

        entry_bar = after.iloc[0]
        entry = float(entry_bar["Open"])
        stop, target = float(s["stop"]), float(s["target"])
        log_df.at[idx, "entry_date"] = after.index[0].date().isoformat()
        log_df.at[idx, "entry_price"] = round(entry, 2)

        if entry >= target or entry <= stop:
            log_df.at[idx, "status"] = "GAP_SKIP"          # gapped through a level; no trade
            continue

        status, exit_px, exit_day = "OPEN", None, None
        for n, (day, bar) in enumerate(after.iterrows(), start=1):
            if bar["Low"] <= stop:
                status, exit_px, exit_day = "STOP", stop, day
                break
            if bar["High"] >= target:
                status, exit_px, exit_day = "TARGET", target, day
                break
            if n >= CONFIG["max_hold_days"]:
                status, exit_px, exit_day = "TIME_EXIT", float(bar["Close"]), day
                break

        log_df.at[idx, "status"] = status
        if exit_px is not None:
            log_df.at[idx, "exit_date"] = exit_day.date().isoformat()
            log_df.at[idx, "exit_price"] = round(exit_px, 2)
            log_df.at[idx, "net_result_pct"] = round((exit_px - entry) / entry * 100 - cost, 2)
    return log_df


def track_record(log_df: pd.DataFrame) -> str:
    closed = log_df[log_df["status"].isin(["STOP", "TARGET", "TIME_EXIT"])]
    open_n = int(log_df["status"].isin(["PENDING", "OPEN"]).sum())
    if closed.empty:
        return f"Paper record: no closed trades yet | open/pending: {open_n}"
    res = pd.to_numeric(closed["net_result_pct"])
    return (f"Paper record: {len(closed)} closed | win rate {(res > 0).mean() * 100:.0f}% | "
            f"avg net {res.mean():+.2f}% | total net {res.sum():+.2f}% | open/pending: {open_n}")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def send_telegram(text: str) -> None:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.info("Telegram not configured - report printed only.")
        return
    import requests
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": text[:4000]}, timeout=20)
        if r.status_code != 200:
            log.warning("Telegram error %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("Telegram send failed: %s", e)


def send_email(subject: str, text: str) -> None:
    """Gmail via SMTP with an App Password (needs 2-Step Verification on the Google account)."""
    user, pwd = os.getenv("EMAIL_USER"), os.getenv("EMAIL_APP_PASSWORD")
    to = os.getenv("EMAIL_TO") or user
    if not user or not pwd:
        log.info("Email not configured - skipping email.")
        return
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(text)
    import html as _html
    msg.add_alternative(
        '<pre style="font-family:Menlo,Consolas,monospace;font-size:12px;">'
        + _html.escape(text) + "</pre>", subtype="html")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(user, pwd.replace(" ", ""))
            smtp.send_message(msg)
        log.info("Email sent to %s", to)
    except Exception as e:
        log.warning("Email send failed: %s", e)


def _fmt_pct(x, scale=100):
    return "n/a" if x is None else f"{float(x) * scale:.1f}%"


def fund_line(p) -> str:
    sym = p["ticker"].replace(".NS", "")
    d = p.get("fund", {})
    de = d.get("de")
    de_txt = "n/a" if de is None else "{:.2f}x".format(float(de) / 100)
    roe_txt, pm_txt = _fmt_pct(d.get("roe")), _fmt_pct(d.get("pm"))
    vals = f"ROE {roe_txt} | margin {pm_txt} | D/E {de_txt}"
    if p["fund_status"] == "unverified":
        return (f"  Fundamentals: NOT VERIFIED ({p['fund_reason']}) - check manually: "
                f"https://www.screener.in/company/{sym}/\n  Known: {vals}")
    src = {"ratios": "Yahoo ratios", "calculated": "calculated from statements",
           "ratios+calculated": "Yahoo ratios + calculated"}.get(d.get("src"), d.get("src"))
    return f"  Fundamentals: verified ({src})\n  {vals}"


def format_report(run_date, data_date, stats, picks, record, note="",
                  recent_table="", strategy_bt=None):
    lines = [f"NSE Swing Screener - {run_date:%d %b %Y}",
             f"Data as of: {data_date:%d %b %Y}"]
    if note:
        lines.append(note)
    if stats:
        lines.append(f"Scanned {stats['scanned']} | momentum pass {stats['momentum']} | "
                     f"fundamental pass {stats['fundamental']} (unverified {stats['unverified']}) | "
                     f"risk-ok {stats['tradeable']}")
        if stats.get("rejected"):
            lines.append("Momentum ok but failed fundamentals: " + "; ".join(stats["rejected"]))
    lines.append("")
    if picks:
        net = CONFIG["target_pct"] - CONFIG["round_trip_cost_pct"]
        for i, p in enumerate(picks, 1):
            lines += [
                f"Pick {i}: {p['ticker'].replace('.NS', '')}",
                f"  Ref close : Rs {p['ref_close']}",
                f"  Stop      : Rs {p['stop']} (-{p['risk_pct']}%)",
                f"  Target    : Rs {p['target']} (+{p['target_pct']}%, ~{net:.2f}% net of est. costs)",
                f"  R:R {p['rr']} | RSI {p['rsi']:.1f} | Vol {p['vol_ratio']:.1f}x | "
                f"time exit {CONFIG['max_hold_days']} sessions",
                fund_line(p),
                "  " + backtest_line(p["bt"], "Backtest, this stock (2y)"),
                "  Also check: no quarterly results due within the holding period.",
                "",
            ]
    elif stats:
        lines += ["No setup met all rules today. No trade is a valid outcome.", ""]
    lines += [record, ""]
    if recent_table:
        lines += [f"Picks - last {CONFIG['recent_sessions']} sessions (* = fundamentals unverified):",
                  recent_table, ""]
    if strategy_bt is not None:
        lines += [backtest_line(strategy_bt, "Strategy backtest, all stocks (2y)"),
                  "Backtest notes: price rules only (no fundamentals), current index members only "
                  "(survivorship bias), costs estimated. Past results do not guarantee future ones.",
                  ""]
    lines += ["Signals only. Not investment advice."]
    return "\n".join(lines)


def expected_session(now: datetime):
    """The trading day whose closing data we should be reporting on.
    Before the data-ready hour (e.g. a run delayed past midnight) it is the previous weekday."""
    d = now.date()
    if now.hour < CONFIG["data_ready_hour_ist"]:
        d -= timedelta(days=1)
    while d.weekday() >= 5:            # Sat/Sun -> Friday
        d -= timedelta(days=1)
    return d


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    force = "--force" in sys.argv
    if yf is None:
        sys.exit("yfinance not installed. Run: pip install -r requirements.txt")
    DATA.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    now = datetime.now(IST)
    run_date = expected_session(now)   # the session being reported (not the clock date)
    # Scheduled runs fire up to 3 times a day; only the first successful one reports.
    scheduled = os.getenv("GITHUB_EVENT_NAME") == "schedule"
    if not force and MARKER.exists() and MARKER.read_text().strip() == run_date.isoformat():
        log.info("Today's report was already sent - nothing to do.")
        return

    universe = load_universe()
    log_df = load_signal_log()
    open_tickers = log_df.loc[log_df["status"].isin(["PENDING", "OPEN"]), "ticker"].tolist()
    prices = get_prices(sorted(set(universe) | set(open_tickers)))
    if not prices:
        if scheduled and now.date() == run_date and now.hour < CONFIG["final_attempt_hour_ist"]:
            log.warning("No price data; a later scheduled run will retry.")
            return
        err = "Screener: no price data received. Check yfinance / network."
        send_telegram(err)
        send_email("Swing Screener: data error", err)
        sys.exit(1)

    data_date = max(df.index[-1] for df in prices.values()).date()
    log_df = evaluate_open(log_df, prices)

    picks, stats, note = [], None, ""
    if data_date < run_date and not force:
        same_evening = now.date() == run_date and now.hour < CONFIG["final_attempt_hour_ist"]
        if scheduled and same_evening:
            log_df.reindex(columns=LOG_COLUMNS).to_csv(SIGNAL_LOG, index=False)
            log.info("Today's bar not available yet; a later scheduled run will retry.")
            return
        note = "Today's bar not available (holiday or data delay). No new signals."
    else:
        # 1) momentum on everything (local, fast)
        mom = {t: s for t in universe if t in prices and (s := momentum_signal(prices[t]))}
        # 2) fundamentals only for survivors
        fund = get_fundamentals(list(mom)) if mom else pd.DataFrame()
        passed, rejected = {}, []
        for t, s in mom.items():
            row = fund.loc[t] if len(fund) and t in fund.index else None
            status, reason, details = evaluate_fundamentals(row)
            log.info("%s momentum ok, fundamentals: %s (%s)", t, status, reason)
            if status == "fail":
                rejected.append(f"{t.replace('.NS', '')} ({reason})")
            else:
                passed[t] = {**s, "fund_status": status, "fund_reason": reason, "fund": details}
        # 3) trade construction + ranking: verified before unverified, then score
        already = set(open_tickers)
        cands = []
        for t, s in passed.items():
            tr = build_trade(s)
            if tr and t not in already:
                cands.append({"ticker": t, **s, **tr})
        cands.sort(key=lambda x: (x["fund_status"] == "pass", x["score"]), reverse=True)
        picks = cands[:CONFIG["top_n"]]
        for p in picks:
            p["bt"] = summarize_trades(simulate_trades(prices[p["ticker"]]))
        stats = {"scanned": sum(t in prices for t in universe), "momentum": len(mom),
                 "fundamental": len(passed),
                 "unverified": sum(v["fund_status"] == "unverified" for v in passed.values()),
                 "tradeable": len(cands), "rejected": rejected}

        new_rows = [{"signal_date": data_date.isoformat(), "ticker": p["ticker"],
                     "ref_close": p["ref_close"], "stop": p["stop"], "target": p["target"],
                     "risk_pct": p["risk_pct"], "target_pct": p["target_pct"],
                     "rsi": round(p["rsi"], 1), "vol_ratio": round(p["vol_ratio"], 2),
                     "score": round(p["score"], 1), "status": "PENDING",
                     "fund_status": p["fund_status"]} for p in picks]
        if new_rows:
            log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)

    log_df.reindex(columns=LOG_COLUMNS).to_csv(SIGNAL_LOG, index=False)
    all_trades = []
    for t in universe:
        if t in prices:
            all_trades += simulate_trades(prices[t])
    strategy_bt = summarize_trades(all_trades)
    report = format_report(run_date, data_date, stats, picks, track_record(log_df), note,
                           recent_picks_table(log_df, prices), strategy_bt)
    (REPORTS / f"{run_date.isoformat()}.txt").write_text(report, encoding="utf-8")
    print(report)
    send_telegram(report)
    if picks:
        subject = f"Swing picks {run_date:%d %b}: " + ", ".join(
            p["ticker"].replace(".NS", "") + ("*" if p["fund_status"] == "unverified" else "")
            for p in picks)
    elif note:
        subject = f"Swing Screener {run_date:%d %b}: no run (data not updated)"
    else:
        subject = f"Swing Screener {run_date:%d %b}: no setup today"
    send_email(subject, report)
    MARKER.write_text(run_date.isoformat())


if __name__ == "__main__":
    main()
