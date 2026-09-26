# NSE Swing Screener (signal-only)

Daily screen of NSE stocks: momentum gate, then fundamental gate, then two trade cards
(ATR stop, fixed % target, risk cap). Every signal is logged and tracked as a paper trade,
so you build a real track record before risking money. **It never places orders.**

## One-time setup (about 30 minutes)

1. **GitHub repo** — create a *private* repo and upload all files in this folder
   (including `.github/workflows/`).
2. **Universe** — replace `universe.csv` with the official constituent list
   (e.g. Nifty LargeMidcap 250) downloaded from the NSE website. The script only needs a
   `Symbol` column. The sample file has 14 large caps just for a first test.
3. **Telegram bot** (optional but recommended)
   - In Telegram, message `@BotFather` → `/newbot` → copy the token.
   - Send any message to your new bot, then open
     `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy `chat.id`.
4. **Secrets** — repo Settings → Secrets and variables → Actions → add
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
5. **First run** — Actions tab → "Daily swing screener" → Run workflow.
   After that it runs automatically at ~18:00 IST on weekdays.

## Running on your own machine instead

```bash
pip install -r requirements.txt
python screener.py           # normal run
python screener.py --force   # ignore the "today's bar missing" check
```
Schedule with cron (Linux/Mac): `0 18 * * 1-5 cd /path/to/swing-screener && python3 screener.py`
or Windows Task Scheduler.

## Files produced (in `data/`)
- `signals.csv` — every signal and its paper-trade outcome
  (PENDING → OPEN → TARGET / STOP / TIME_EXIT, or GAP_SKIP)
- `fundamentals.csv` — cached yfinance fundamentals (refreshed weekly)
- `reports/YYYY-MM-DD.txt` — the daily report

## Paper-trade rules
Entry = next session's open. Exit at stop, target, or close of session 8.
If stop and target are both touched on the same day, it is counted as a stop (conservative).
Estimated round-trip cost is deducted from every result.

## Tuning
All thresholds are in the `CONFIG` block at the top of `screener.py`.
Change one thing at a time and let the paper record tell you if it helped.

## Known limitations — read before trusting any output
- **Not backtested.** The rules are reasonable-looking, not proven.
- **yfinance is unofficial** (it reads Yahoo Finance). It can break, rate-limit, or
  return stale/incomplete fundamentals for Indian stocks. Verify signals on your broker
  before acting.
- **Fundamental fields** (D/E units, sector labels) come from Yahoo and may be wrong or missing.
  Stocks with missing fundamentals are excluded by default.
- **Cost estimate** (0.20% round trip) is approximate — check your broker's contract notes.
- **Scheduled GitHub runs** can start late, and may be paused on inactive repos — check GitHub's docs.
- Signals only. Not investment advice.
