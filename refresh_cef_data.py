#!/usr/bin/env python3
"""
CEF Income Screener — Daily Data Refresh
Runs at 6 PM via launchd. Pulls from SEC EDGAR + Yahoo Finance only.
Output: data.json  →  git commit + push to GitHub Pages repo.

Dependencies: pip install yfinance requests pandas
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

# ─── CONFIG ────────────────────────────────────────────────────────────────────
REPO_DIR    = os.path.dirname(os.path.abspath(__file__))   # script lives in repo root
DATA_FILE   = os.path.join(REPO_DIR, "data.json")
GIT_REMOTE  = "origin"                                     # assumes remote already set
GIT_BRANCH  = "main"

# EDGAR requires a descriptive User-Agent (your contact info)
EDGAR_HEADERS = {
    "User-Agent": "CEF Screener jen81715@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_BASE        = "https://data.sec.gov"
EDGAR_SEARCH_URL  = "https://efts.sec.gov/LATEST/search-index"

# Exchanges we care about
VALID_EXCHANGES = {"NYSE", "AMEX", "NYSE MKT", "NYSE ARCA", "NYSEARCA", "BATS"}

# How long to wait between EDGAR calls (10 req/sec allowed)
EDGAR_SLEEP = 0.12
YFINANCE_SLEEP = 0.3

# ─── EDGAR: CEF UNIVERSE ───────────────────────────────────────────────────────

def fetch_n2_ciks() -> dict[str, str]:
    """
    Return {cik: company_name} for all companies that have ever filed Form N-2.
    N-2 is the registration statement exclusively for closed-end management
    investment companies — the cleanest legal definition of 'CEF'.
    """
    print("Fetching CEF universe from EDGAR (Form N-2 filers)…")
    ciks: dict[str, str] = {}
    start = 0
    page_size = 100

    while True:
        params = {
            "q": "",
            "forms": "N-2",
            "dateRange": "custom",
            "startdt": "2000-01-01",
            "enddt": datetime.now().strftime("%Y-%m-%d"),
            "from": start,
            "size": page_size,
        }
        try:
            resp = requests.get(EDGAR_SEARCH_URL, params=params,
                                headers=EDGAR_HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  EDGAR search error at offset {start}: {e}")
            break

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            raw_id = hit.get("_id", "")
            # _id format: "<cik>:<accession>" or just "<cik>"
            cik = raw_id.split(":")[0].lstrip("0")
            src = hit.get("_source", {})
            name = src.get("entity_name", "")
            if cik and name:
                ciks[cik] = name

        total = data.get("hits", {}).get("total", {}).get("value", 0)
        start += page_size
        if start >= total:
            break

        time.sleep(EDGAR_SLEEP)

    print(f"  Found {len(ciks)} N-2 filers")
    return ciks


def get_tickers_for_cik(cik: str) -> list[tuple[str, str]]:
    """Return [(ticker, exchange), …] for a CIK, filtered to valid exchanges."""
    padded = cik.zfill(10)
    url = f"{EDGAR_BASE}/submissions/CIK{padded}.json"
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    tickers   = data.get("tickers", [])
    exchanges = data.get("exchanges", [])

    results = []
    for t, ex in zip(tickers, exchanges):
        ex_norm = ex.strip().upper()
        if ex_norm in VALID_EXCHANGES or "NYSE" in ex_norm or "AMEX" in ex_norm:
            results.append((t.upper(), ex_norm))
    return results


# ─── YAHOO FINANCE: PRICE + DISTRIBUTION DATA ──────────────────────────────────

def classify_distribution_frequency(dividends: pd.Series) -> str:
    """Infer payment frequency from recent dividend history."""
    if dividends.empty:
        return "Unknown"
    one_year_ago = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=1)
    recent = dividends[dividends.index >= one_year_ago]
    n = len(recent)
    if n >= 10:
        return "Monthly"
    if n >= 3:
        return "Quarterly"
    if n >= 1:
        return "Annual"
    return "Unknown"


def get_last_ex_date(dividends: pd.Series) -> str | None:
    """Return the most recent ex-dividend date as a string."""
    if dividends.empty:
        return None
    last_ts = dividends.index[-1]
    try:
        return pd.Timestamp(last_ts).strftime("%Y-%m-%d")
    except Exception:
        return None


def get_fund_data(ticker: str, cik: str | None = None) -> dict | None:
    """
    Fetch all data for one ticker from Yahoo Finance.
    Returns None if the ticker is not a CEF or has no price.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        # Confirm it's a closed-end fund
        quote_type = info.get("quoteType", "").upper()
        if quote_type not in ("CLOSED-END FUND", "CLOSED_END_FUND", "CEF"):
            return None

        price = info.get("regularMarketPrice") or info.get("previousClose")
        if not price or price <= 0:
            return None

        high52 = info.get("fiftyTwoWeekHigh")
        low52  = info.get("fiftyTwoWeekLow")

        # ── Dividends ──
        dividends = yf_ticker.dividends  # DatetimeIndex → float, UTC-aware
        if not dividends.empty:
            one_year_ago = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=1)
            recent_divs  = dividends[dividends.index >= one_year_ago]
            annual_dist  = float(recent_divs.sum()) if not recent_divs.empty \
                           else float(dividends.tail(12).sum())
            last_amount  = float(dividends.iloc[-1])
        else:
            annual_dist = 0.0
            last_amount = 0.0

        dist_rate  = round((annual_dist / price) * 100, 2) if price else 0.0
        frequency  = classify_distribution_frequency(dividends)
        ex_date    = get_last_ex_date(dividends)

        # ── Inception date ──
        inception_ts = info.get("fundInceptionDate")
        if inception_ts:
            inception_dt  = datetime.utcfromtimestamp(inception_ts)
            inception_str = inception_dt.strftime("%Y-%m-%d")
            age_years     = round((datetime.utcnow() - inception_dt).days / 365.25, 1)
        else:
            inception_str = None
            age_years     = None

        fund_name = info.get("longName") or info.get("shortName") or ticker

        return {
            "ticker":         ticker,
            "name":           fund_name,
            "cik":            cik,
            "inception_date": inception_str,
            "age_years":      age_years,
            "price":          round(float(price), 2),
            "week52_high":    round(float(high52), 2) if high52 else None,
            "week52_low":     round(float(low52), 2) if low52 else None,
            "dist_type":      "Income",   # overridden by N-CEN parse below
            "dist_rate":      dist_rate,
            "dist_amount":    round(last_amount, 4),
            "frequency":      frequency,
            "ex_date":        ex_date,
            "pay_date":       None,       # enriched by N-PORT if available
            "holdings_count": None,       # enriched by N-PORT below
        }

    except Exception as e:
        print(f"    yfinance error [{ticker}]: {e}")
        return None


# ─── EDGAR: N-PORT HOLDINGS COUNT ──────────────────────────────────────────────

def get_holdings_count(cik: str) -> int | None:
    """
    Find the most recent N-PORT filing for a CIK and count portfolio positions.
    Uses the EDGAR XBRL inline viewer data (no heavy XML parsing needed).
    """
    padded = cik.zfill(10)
    submissions_url = f"{EDGAR_BASE}/submissions/CIK{padded}.json"
    try:
        resp = requests.get(submissions_url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        sub = resp.json()
    except Exception:
        return None

    filings = sub.get("filings", {}).get("recent", {})
    forms       = filings.get("form", [])
    accessions  = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    # Find most recent N-PORT or N-PORT/A
    nport_accession = None
    for form, acc in zip(forms, accessions):
        if form.upper() in ("N-PORT", "N-PORT/A"):
            nport_accession = acc.replace("-", "")
            break

    if not nport_accession:
        return None

    # Fetch the filing index
    idx_url = (f"{EDGAR_BASE}/Archives/edgar/data/{cik}/"
               f"{nport_accession}/{nport_accession}-index.json")
    try:
        time.sleep(EDGAR_SLEEP)
        resp = requests.get(idx_url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        idx = resp.json()
    except Exception:
        return None

    # Find the primary XML document
    xml_file = None
    for doc in idx.get("directory", {}).get("item", []):
        name = doc.get("name", "")
        if name.lower().endswith(".xml") and "nport" in name.lower():
            xml_file = name
            break
        if xml_file is None and name.lower().endswith(".xml"):
            xml_file = name

    if not xml_file:
        return None

    xml_url = (f"{EDGAR_BASE}/Archives/edgar/data/{cik}/"
               f"{nport_accession}/{xml_file}")
    try:
        time.sleep(EDGAR_SLEEP)
        resp = requests.get(xml_url, headers=EDGAR_HEADERS, timeout=30)
        resp.raise_for_status()
        xml_text = resp.text
    except Exception:
        return None

    # Count <invstOrSec> elements (each is one holding)
    count = xml_text.count("<invstOrSec>")
    return count if count > 0 else None


# ─── EDGAR: N-CEN MANAGED DISTRIBUTION FLAG ────────────────────────────────────

def check_managed_distribution(cik: str) -> bool:
    """
    Return True if the most recent N-CEN indicates a managed distribution plan
    (item 40: does the fund have a managed distribution plan?).
    """
    padded = cik.zfill(10)
    try:
        resp = requests.get(f"{EDGAR_BASE}/submissions/CIK{padded}.json",
                            headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        sub = resp.json()
    except Exception:
        return False

    filings    = sub.get("filings", {}).get("recent", {})
    forms      = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])

    ncen_accession = None
    for form, acc in zip(forms, accessions):
        if form.upper() in ("N-CEN", "N-CEN/A"):
            ncen_accession = acc.replace("-", "")
            break

    if not ncen_accession:
        return False

    idx_url = (f"{EDGAR_BASE}/Archives/edgar/data/{cik}/"
               f"{ncen_accession}/{ncen_accession}-index.json")
    try:
        time.sleep(EDGAR_SLEEP)
        resp = requests.get(idx_url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        idx = resp.json()
    except Exception:
        return False

    xml_file = None
    for doc in idx.get("directory", {}).get("item", []):
        name = doc.get("name", "")
        if name.lower().endswith(".xml"):
            xml_file = name
            break

    if not xml_file:
        return False

    xml_url = (f"{EDGAR_BASE}/Archives/edgar/data/{cik}/"
               f"{ncen_accession}/{xml_file}")
    try:
        time.sleep(EDGAR_SLEEP)
        resp = requests.get(xml_url, headers=EDGAR_HEADERS, timeout=30)
        resp.raise_for_status()
        xml_text = resp.text
    except Exception:
        return False

    # Look for managedDistributionPlan = Y or item40 = Y
    patterns = [
        r"<managedDistributionPlan>\s*Y\s*</managedDistributionPlan>",
        r"<item40>\s*Y\s*</item40>",
        r"<managedDist[^>]*>\s*Y\s*</managedDist[^>]*>",
    ]
    for pat in patterns:
        if re.search(pat, xml_text, re.IGNORECASE):
            return True

    return False


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def build_dataset() -> list[dict]:
    cik_map = fetch_n2_ciks()   # {cik: name}
    results: list[dict] = []
    seen_tickers: set[str] = set()

    total = len(cik_map)
    print(f"\nResolving tickers + fetching fund data for {total} CIKs…")

    for idx, (cik, company_name) in enumerate(cik_map.items(), 1):
        print(f"  [{idx}/{total}] CIK {cik} ({company_name[:40]})", end="", flush=True)
        time.sleep(EDGAR_SLEEP)

        ticker_pairs = get_tickers_for_cik(cik)
        if not ticker_pairs:
            print(" — no NYSE/AMEX ticker, skip")
            continue

        for ticker, exchange in ticker_pairs:
            if ticker in seen_tickers:
                print(f" — {ticker} already processed, skip")
                continue
            seen_tickers.add(ticker)

            print(f" → {ticker} ({exchange})", end="", flush=True)
            time.sleep(YFINANCE_SLEEP)

            fund = get_fund_data(ticker, cik)
            if fund is None:
                print(" ✗ not a CEF / no price")
                continue

            # Enrich with N-PORT holdings count (slow — only if CIK known)
            if cik:
                holdings = get_holdings_count(cik)
                fund["holdings_count"] = holdings

                # Check managed distribution flag
                is_managed = check_managed_distribution(cik)
                fund["dist_type"] = "Managed" if is_managed else "Income"

            results.append(fund)
            print(f" ✓  rate={fund['dist_rate']}%  holdings={fund.get('holdings_count')}")

    return results


def write_json(funds: list[dict]) -> None:
    payload = {
        "updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count":   len(funds),
        "funds":   funds,
    }
    with open(DATA_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {len(funds)} funds to {DATA_FILE}")


def git_push() -> None:
    cmds = [
        ["git", "-C", REPO_DIR, "add", "data.json"],
        ["git", "-C", REPO_DIR, "commit", "-m",
         f"chore: refresh CEF data {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        ["git", "-C", REPO_DIR, "push", GIT_REMOTE, GIT_BRANCH],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # "nothing to commit" is fine
            if "nothing to commit" in result.stdout + result.stderr:
                print("git: nothing new to commit")
                return
            print(f"git error: {result.stderr}")
            return
        print(f"git: {' '.join(cmd[3:])} — OK")


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = datetime.now()
    print(f"=== CEF Screener Refresh  {start.strftime('%Y-%m-%d %H:%M')} ===\n")

    try:
        funds = build_dataset()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)

    if not funds:
        print("No funds collected — not overwriting data.json")
        sys.exit(1)

    write_json(funds)
    git_push()

    elapsed = (datetime.now() - start).seconds // 60
    print(f"\nDone in {elapsed} min. {len(funds)} CEFs in data.json.")
