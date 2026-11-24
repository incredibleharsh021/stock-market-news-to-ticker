"""
ticker_map.py — Stock Ticker Extraction from News Text

PURPOSE:
    Extracts NSE stock ticker symbols from news article text using a
    two-phase approach: exact matching first, fuzzy matching as fallback.

TWO-PHASE MATCHING STRATEGY:
    Phase 1 — Exact matching (regex word-boundary):
        Fast, zero false positives. Catches well-known names.
    
    Phase 2 — Fuzzy fallback (only when Phase 1 finds nothing):
        Uses RapidFuzz to handle spelling variants, abbreviations,
        and minor differences (e.g. "Bajaj Fin" → "Bajaj Finance").
        High threshold (88) + minimum name length guard to limit
        false positives.

WHY NOT ALWAYS FUZZY?
    Fuzzy matching on short strings is noisy — "India" scores high
    against "IndiaMART", "ICICI", etc. Running exact first and only
    falling back to fuzzy when needed gives the best of both worlds.
"""

import csv
import os
import re
from functools import lru_cache

# ============================================================
# Try to import rapidfuzz — graceful fallback if not installed
# ============================================================
try:
    from rapidfuzz import process as rfprocess, fuzz as rffuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    print("⚠️  rapidfuzz not installed — fuzzy matching disabled. "
          "Run: pip install rapidfuzz")


# ============================================================
# Global ticker dictionary — loaded once from CSV file
# ============================================================
_ticker_dict = None          # sorted list of (name, ticker) for exact match
_fuzzy_candidates = None     # list of (name, ticker) suitable for fuzzy match


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
FUZZY_THRESHOLD   = 88    # Minimum WRatio score to accept a fuzzy match
FUZZY_MIN_LEN     = 8     # Only fuzzy-match company names ≥ this many chars
                          # Guards against "SBI" fuzzing to random short text
EXACT_MIN_LEN     = 3     # Skip exact-match entries with names ≤ this long
                          # e.g. "IT" is too ambiguous

# Hard list of generic words we never want to fuzzy-match
# (they appear in company names but also in everyday news text)
_FUZZY_STOPWORDS = {
    "india", "indian", "limited", "ltd", "corp", "corporation",
    "industries", "services", "solutions", "technologies", "group",
    "finance", "financial", "capital", "bank", "energy", "power",
    "infrastructure", "holdings", "international", "national",
    "enterprise", "enterprises"
}


def load_ticker_dictionary():
    """
    Load company name → ticker mapping from the official NSE CSV file.

    Builds two internal structures:
        _ticker_dict       — all (name, ticker) pairs sorted longest-first
                             for exact regex matching.
        _fuzzy_candidates  — filtered subset where name length ≥ FUZZY_MIN_LEN
                             and name is not a generic stopword, used for
                             fuzzy Phase 2.

    Returns:
        list[tuple]: _ticker_dict (sorted longest-first)
    """
    global _ticker_dict, _fuzzy_candidates

    if _ticker_dict is not None:
        return _ticker_dict

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path    = os.path.join(current_dir, '..', 'data', 'nse_equities.csv')

    temp_list = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get('SYMBOL', '').strip()
                name   = row.get('NAME OF COMPANY', '').strip()
                if not symbol or not name:
                    continue

                ticker = f"{symbol}.NS"
                temp_list.append((name, ticker))

                # Strip common corporate suffixes for looser matching
                # "Reliance Industries Limited" → "Reliance Industries"
                clean = re.sub(
                    r'(?i)\b(limited|ltd\.?|corporation|corp\.?|inc\.?)\b',
                    '', name
                ).strip()
                if clean != name and len(clean) > EXACT_MIN_LEN:
                    temp_list.append((clean, ticker))

        # ── Manual aliases for companies that go by nicknames in news ──
        ALIASES = {
            "RELIANCE":    ["Reliance", "RIL", "Jio"],
            "TCS":         ["TCS", "Tata Consultancy"],
            "INFY":        ["Infosys"],
            "HDFCBANK":    ["HDFC Bank", "HDFC"],
            "ICICIBANK":   ["ICICI Bank", "ICICI"],
            "BHARTIARTL":  ["Bharti Airtel", "Airtel"],
            "SBIN":        ["SBI", "State Bank"],
            "HINDUNILVR":  ["Hindustan Unilever", "HUL"],
            "LT":          ["Larsen & Toubro", "L&T"],
            "WIPRO":       ["Wipro"],
            "HCLTECH":     ["HCL Technologies", "HCL Tech", "HCL"],
            "BAJFINANCE":  ["Bajaj Finance"],
            "BAJAJFINSV":  ["Bajaj Finserv"],
            "MARUTI":      ["Maruti Suzuki", "Maruti"],
            "TATAMOTORS":  ["Tata Motors"],
            "TATASTEEL":   ["Tata Steel"],
            "ADANIENT":    ["Adani Enterprises", "Adani Group"],
            "ADANIPORTS":  ["Adani Ports"],
            "ADANIGREEN":  ["Adani Green"],
            "ZOMATO":      ["Zomato"],
            "PAYTM":       ["Paytm", "One97"],
            "ONGC":        ["ONGC", "Oil India"],
            "NTPC":        ["NTPC"],
            "POWERGRID":   ["Power Grid"],
            "COALINDIA":   ["Coal India"],
            "SUNPHARMA":   ["Sun Pharma", "Sun Pharmaceutical"],
            "DRREDDY":     ["Dr Reddy", "Dr. Reddy"],
            "CIPLA":       ["Cipla"],
            "TITAN":       ["Titan"],
            "NESTLEIND":   ["Nestle India", "Nestle"],
            "ULTRACEMCO":  ["UltraTech Cement", "UltraTech"],
            "ASIANPAINT":  ["Asian Paints"],
        }
        for sym, names in ALIASES.items():
            for n in names:
                temp_list.append((n, f"{sym}.NS"))

        # Deduplicate and sort longest-first for exact matching
        # (ensures "Reliance Industries" matches before "Reliance")
        _ticker_dict = sorted(
            set(temp_list),
            key=lambda x: len(x[0]),
            reverse=True
        )

        # Build fuzzy candidate list — only names long enough and
        # not made of generic stopwords
        _fuzzy_candidates = [
            (name, ticker)
            for name, ticker in _ticker_dict
            if len(name) >= FUZZY_MIN_LEN
            and name.lower() not in _FUZZY_STOPWORDS
            and not all(w.lower() in _FUZZY_STOPWORDS
                        for w in name.split())
        ]

        print(f"✅ Loaded {len(_ticker_dict)} ticker aliases "
              f"({len(_fuzzy_candidates)} fuzzy candidates) from NSE dataset")

    except FileNotFoundError:
        print("❌ nse_equities.csv not found — check data/ directory")
        _ticker_dict      = []
        _fuzzy_candidates = []

    return _ticker_dict


# ─────────────────────────────────────────────────────────────
# PHASE 1 — Exact regex matching
# ─────────────────────────────────────────────────────────────

def _exact_match(search_text: str) -> set:
    """
    Phase 1: Find tickers via exact word-boundary regex.

    Args:
        search_text (str): Combined title + description

    Returns:
        set[str]: Matched ticker symbols
    """
    ticker_entries = load_ticker_dictionary()
    found = set()

    for company_name, ticker_symbol in ticker_entries:
        if len(company_name) <= EXACT_MIN_LEN:
            continue
        pattern = r'\b' + re.escape(company_name) + r'\b'
        if re.search(pattern, search_text, re.IGNORECASE):
            found.add(ticker_symbol)

    return found


# ─────────────────────────────────────────────────────────────
# PHASE 2 — Fuzzy fallback
# ─────────────────────────────────────────────────────────────

def _extract_candidate_phrases(text: str):
    """
    Pull candidate phrases from text for fuzzy comparison.

    Strategy:
        - Extract runs of Title-Cased or ALL-CAPS words (likely proper nouns)
        - Also extract 2- and 3-word windows to catch multi-word names
        - Skip candidates that are pure stopwords

    Args:
        text (str): The news text

    Returns:
        list[str]: Candidate phrases to fuzzy-match against company names
    """
    # 1. Capitalised token runs (e.g. "Bajaj Finance", "HDFC Bank")
    tokens = text.split()
    caps_runs = []
    current_run = []
    for tok in tokens:
        # Keep tokens that start with uppercase or are all-caps (>=2 chars)
        clean_tok = re.sub(r'[^A-Za-z&\-]', '', tok)
        if clean_tok and (clean_tok[0].isupper() or clean_tok.isupper()):
            current_run.append(clean_tok)
        else:
            if len(current_run) >= 1:
                # yield 1, 2, and 3-word combos from the run
                for size in (1, 2, 3):
                    for i in range(len(current_run) - size + 1):
                        phrase = ' '.join(current_run[i:i+size])
                        if len(phrase) >= FUZZY_MIN_LEN:
                            caps_runs.append(phrase)
            current_run = []
    # Flush last run
    if current_run:
        for size in (1, 2, 3):
            for i in range(len(current_run) - size + 1):
                phrase = ' '.join(current_run[i:i+size])
                if len(phrase) >= FUZZY_MIN_LEN:
                    caps_runs.append(phrase)

    # 2. Deduplicate
    return list(dict.fromkeys(caps_runs))


def _fuzzy_match(search_text: str) -> set:
    """
    Phase 2: Fuzzy fallback using RapidFuzz WRatio.

    For each candidate phrase extracted from the text, finds the
    closest company name in our fuzzy-candidate list. Only accepts
    matches scoring ≥ FUZZY_THRESHOLD.

    Args:
        search_text (str): Combined title + description

    Returns:
        set[str]: Matched ticker symbols (may be empty)
    """
    if not _RAPIDFUZZ_AVAILABLE or not _fuzzy_candidates:
        return set()

    load_ticker_dictionary()   # ensure _fuzzy_candidates is populated

    candidate_phrases = _extract_candidate_phrases(search_text)
    if not candidate_phrases:
        return set()

    # Build lookup: company_name → ticker  (for O(1) reverse lookup)
    name_to_ticker = {name: ticker for name, ticker in _fuzzy_candidates}
    company_names  = list(name_to_ticker.keys())

    found = set()

    for phrase in candidate_phrases:
        # Skip generic phrases that would produce noisy fuzzy matches
        if phrase.lower() in _FUZZY_STOPWORDS:
            continue

        # rfprocess.extractOne returns (match, score, index) or None
        hit = rfprocess.extractOne(
            phrase,
            company_names,
            scorer=rffuzz.WRatio,
            score_cutoff=FUZZY_THRESHOLD
        )
        if hit:
            matched_name, score, _ = hit
            ticker = name_to_ticker[matched_name]
            found.add(ticker)

    return found


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def extract_tickers(title: str, description: str) -> list:
    """
    Extract NSE ticker symbols from article text.

    Two-phase approach:
        1. Exact word-boundary regex matching (fast, zero false positives)
        2. Fuzzy matching via RapidFuzz — ONLY if Phase 1 found nothing
           (handles spelling variants, abbreviations, nicknames)

    Args:
        title       (str): Article headline
        description (str): Article body/summary

    Returns:
        list[str]: Sorted list of matched tickers, e.g. ["INFY.NS", "TCS.NS"]
                   Empty list if nothing found.
    """
    search_text = f"{title} {description}".strip()
    if not search_text:
        return []

    # ── Phase 1: exact matching ──────────────────────────────
    exact_tickers = _exact_match(search_text)
    if exact_tickers:
        return sorted(exact_tickers)

    # ── Phase 2: fuzzy fallback (only when exact found nothing) ──
    fuzzy_tickers = _fuzzy_match(search_text)
    return sorted(fuzzy_tickers)


def get_ticker_display(tickers: list) -> str:
    """
    Format ticker list for UI display.

    Args:
        tickers (list[str]): List of ticker symbols

    Returns:
        str: "RELIANCE.NS, TCS.NS" or "No ticker detected"
    """
    if not tickers:
        return "No ticker detected"
    return ", ".join(tickers)