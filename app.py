"""
app.py — Stock-Market News → Ticker
SMAI Assignment 3 | T9.5 | IIIT Hyderabad

NEW FEATURES (vs. previous version):
    1. Market Mood Meter   — Sidebar gauge showing Bullish / Bearish / Neutral
                             article breakdown as colour-coded progress bars.
    2. Ask the News        — "Ask the News" expander: type any question and
                             Gemini answers from the loaded articles as context.
                             Single user-triggered API call; no rate-limit impact.
    3. Freshness Badge     — 🟢 Fresh (<2 h) / 🟡 Recent (2-6 h) / 🔴 Old (>6 h)
                             shown on every article card.
    4. Bookmark Articles   — ⭐/☆ button on each card saves/unsaves to
                             session_state. A "⭐ Saved" tab shows bookmarks.
    5. Source Chart        — Bar chart in sidebar showing article count per
                             feed source (ET vs MoneyControl breakdown).
"""

import streamlit as st
from collections import Counter
from datetime import datetime, timezone

from utils.fetcher    import fetch_all_news
from utils.ml_models  import (batch_classify, batch_summarize,
                               ask_news_question)
from utils.ticker_map import extract_tickers, get_ticker_display


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock-Market News Navigator",
    page_icon="📈",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Scrolling ticker bar ──────────────────────────────── */
.ticker-wrapper {
    background: #0a0a1a; color: #00e676; overflow: hidden;
    white-space: nowrap; padding: 8px 0; border-radius: 6px;
    font-family: monospace; font-size: 14px; margin-bottom: 12px;
}
.ticker-content {
    display: inline-block;
    animation: scroll-left 40s linear infinite;
}
@keyframes scroll-left {
    0%   { transform: translateX(100vw); }
    100% { transform: translateX(-100%); }
}

/* ── Article card ──────────────────────────────────────── */
.article-card {
    background: #1e1e2e; border: 1px solid #2d2d44;
    border-left: 4px solid #6c63ff;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;
}
.article-card h4 { margin: 0 0 6px 0; font-size: 15px; color: #e0e0ff; }
.article-card p  { margin: 4px 0; font-size: 13px; color: #aaaacc; }
.article-card .meta { font-size: 11px; color: #666688; margin-top: 8px; }

/* ── Badges ────────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; margin-right: 4px;
}
.badge-green  { background:#1b5e20; color:#a5d6a7; }
.badge-red    { background:#b71c1c; color:#ef9a9a; }
.badge-yellow { background:#f57f17; color:#fff9c4; }
.badge-blue   { background:#0d47a1; color:#90caf9; }
.badge-grey   { background:#37474f; color:#b0bec5; }
.badge-ticker { background:#1a237e; color:#82b1ff; font-family:monospace; }
.badge-fresh  { background:#004d40; color:#80cbc4; }
.badge-recent { background:#e65100; color:#ffccbc; }
.badge-old    { background:#37474f; color:#b0bec5; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
# "⭐ Saved" tab is appended dynamically at render time
BASE_CATEGORIES = ["All", "Stock Market", "Business",
                   "Economy", "Technology", "Others"]

CATEGORY_COLOURS = {
    "Stock Market": "badge-green", "Business":   "badge-blue",
    "Economy":      "badge-yellow", "Technology": "badge-red",
    "Others":       "badge-grey",
}


# ─────────────────────────────────────────────────────────────
# SENTIMENT HELPER
# ─────────────────────────────────────────────────────────────

def infer_sentiment(title: str) -> str:
    t = title.lower()
    bullish = ["surge", "jump", "rally", "record", "gain", "profit",
               "high", "boom", "rise", "soar", "up", "growth"]
    bearish = ["fall", "drop", "slump", "low", "loss", "bear",
               "decline", "cut", "weak", "crash", "down", "concern"]
    b = sum(w in t for w in bullish)
    s = sum(w in t for w in bearish)
    if b > s: return "🟢 Bullish"
    if s > b: return "🔴 Bearish"
    return "⚪ Neutral"


# ─────────────────────────────────────────────────────────────
# FEATURE 3 — FRESHNESS BADGE
# ─────────────────────────────────────────────────────────────

def compute_freshness(published_str: str) -> str:
    """
    Compare article publish time to now and return a freshness label.

    Returns:
        "🟢 Fresh"   — published within last 2 hours
        "🟡 Recent"  — published 2–6 hours ago
        "🔴 Old"     — published more than 6 hours ago (or unparseable)
    """
    formats = [
        "%d %b %Y, %I:%M %p",   # our own formatted string from fetcher
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    pub_dt = None
    for fmt in formats:
        try:
            pub_dt = datetime.strptime(published_str, fmt)
            # Make timezone-aware if naive
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue

    if pub_dt is None:
        return "🔴 Old"   # can't parse → assume old

    now_utc  = datetime.now(tz=timezone.utc)
    age_hrs  = (now_utc - pub_dt).total_seconds() / 3600

    if age_hrs < 2:   return "🟢 Fresh"
    if age_hrs < 6:   return "🟡 Recent"
    return "🔴 Old"


def freshness_badge(freshness: str) -> str:
    css_map = {"🟢 Fresh": "badge-fresh", "🟡 Recent": "badge-recent"}
    css = css_map.get(freshness, "badge-old")
    return f'<span class="badge {css}">{freshness}</span>'


# ─────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline() -> list:
    """
    Fetch → batch-classify → batch-summarise → tickers → sentiment → freshness.

    Summarisation uses a single Gemini API call (batch_summarize),
    so the free-tier rate limit is never hit.
    """
    raw = fetch_all_news(max_articles=30)
    if not raw:
        return []

    prog = st.progress(0, text="📡 Fetching articles...")

    prog.progress(0.2, text="🤖 Classifying articles (Zero-Shot BART)...")
    classifications = batch_classify(raw)

    prog.progress(0.6, text="✨ Generating summaries (1 Gemini call)...")
    summaries = batch_summarize(raw)

    prog.progress(0.9, text="🏷️ Extracting tickers & metadata...")
    processed = []
    for i, art in enumerate(raw):
        cat, conf = classifications[i]
        processed.append({
            **art,
            "category":   cat,
            "confidence": conf,
            "tickers":    extract_tickers(art["title"], art["description"]),
            "summary":    summaries[i],
            "sentiment":  infer_sentiment(art["title"]),
            "freshness":  compute_freshness(art.get("published", "")),
        })

    prog.empty()
    return processed


def ensure_data_loaded():
    """Load pipeline into session_state. No-op if already loaded."""
    if "articles" not in st.session_state or not st.session_state.articles:
        st.session_state.articles   = run_pipeline()
        st.session_state.last_fetch = datetime.now().strftime("%d %b %Y, %I:%M %p")
    # Ensure bookmark set exists
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = set()


# ─────────────────────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────────────────────

def render_ticker_bar(articles: list):
    items = [
        f"📌 {get_ticker_display(a.get('tickers', []))} &nbsp;|&nbsp; {a['title'][:55]}"
        for a in articles[:15]
    ]
    content = " &nbsp;&nbsp;•&nbsp;&nbsp; ".join(items)
    st.markdown(
        f'<div class="ticker-wrapper"><span class="ticker-content">{content}</span></div>',
        unsafe_allow_html=True,
    )


def category_badge(cat: str) -> str:
    css = CATEGORY_COLOURS.get(cat, "badge-grey")
    return f'<span class="badge {css}">{cat}</span>'


def sentiment_badge(sent: str) -> str:
    if   "Bullish" in sent: css = "badge-green"
    elif "Bearish" in sent: css = "badge-red"
    else:                    css = "badge-grey"
    return f'<span class="badge {css}">{sent}</span>'


def ticker_badges(tickers: list) -> str:
    if not tickers:
        return '<span class="badge badge-grey">No ticker</span>'
    return " ".join(
        f'<span class="badge badge-ticker">{t}</span>' for t in tickers[:4]
    )


def render_article_card(art: dict, card_key: str, show_bookmark: bool = True):
    """
    Render one article card with badges and an optional bookmark button.

    Args:
        art           (dict): Processed article dict.
        card_key      (str):  Caller-supplied unique string used as the
                              Streamlit widget key for the bookmark button.
                              Must be unique across ALL rendered cards in the
                              entire app — pass f"{tab_name}_{index}".
        show_bookmark (bool): Whether to show the ⭐/☆ bookmark button.

    WHY card_key instead of hash?
        The same article can appear in both "All" and its category tab.
        hash(url) % N produces the same number both times → duplicate key
        error. Passing tab_name + index guarantees uniqueness because no
        two cards share both the same tab and the same position.
    """
    col_btn, col_card = st.columns([0.04, 0.96])

    with col_card:
        st.markdown(
            f"""
            <div class="article-card">
                <h4><a href="{art['link']}" target="_blank"
                       style="color:#c8b8ff;text-decoration:none;">
                    {art['title']}
                </a></h4>
                <p>{art['summary']}</p>
                <p>
                    {category_badge(art['category'])}
                    {sentiment_badge(art['sentiment'])}
                    {freshness_badge(art.get('freshness', '🔴 Old'))}
                    {ticker_badges(art['tickers'])}
                </p>
                <p class="meta">
                    🕐 {art['published']} &nbsp;·&nbsp;
                    📡 {art['source']} &nbsp;·&nbsp;
                    Confidence: {art['confidence']:.0%}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_bookmark:
        with col_btn:
            article_id = art.get("link", art["title"])
            is_saved   = article_id in st.session_state.bookmarks
            icon       = "⭐" if is_saved else "☆"
            # card_key is guaranteed unique by the caller (tab_name + index)
            if st.button(icon, key=f"bm_{card_key}", help="Save / unsave article"):
                if is_saved:
                    st.session_state.bookmarks.discard(article_id)
                else:
                    st.session_state.bookmarks.add(article_id)
                st.rerun()


# ─────────────────────────────────────────────────────────────
# FEATURE 1 — MARKET MOOD METER (sidebar)
# ─────────────────────────────────────────────────────────────

def render_market_mood(articles: list):
    """
    Show a Bullish / Bearish / Neutral breakdown as colour-coded
    percentage bars in the sidebar.
    """
    if not articles:
        return

    counts = Counter(a["sentiment"] for a in articles)
    total  = len(articles)

    b_count = counts.get("🟢 Bullish", 0)
    r_count = counts.get("🔴 Bearish", 0)
    n_count = counts.get("⚪ Neutral",  0)

    b_pct = int(b_count / total * 100)
    r_pct = int(r_count / total * 100)
    n_pct = 100 - b_pct - r_pct   # absorb rounding remainder

    # Overall mood label
    if b_count > r_count:
        mood_label = "📈 Market leaning Bullish"
        mood_color = "#a5d6a7"
    elif r_count > b_count:
        mood_label = "📉 Market leaning Bearish"
        mood_color = "#ef9a9a"
    else:
        mood_label = "➡️ Market is Neutral"
        mood_color = "#b0bec5"

    st.sidebar.markdown("### 🌡️ Market Mood Meter")
    st.sidebar.markdown(
        f'<p style="color:{mood_color};font-weight:600;margin:0">{mood_label}</p>',
        unsafe_allow_html=True,
    )
    # Colour-coded mini bars using HTML
    st.sidebar.markdown(
        f"""
        <div style="margin-top:6px">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
                <div style="width:{b_pct}%;height:10px;background:#388e3c;
                            border-radius:4px;min-width:2px"></div>
                <span style="font-size:11px;color:#a5d6a7">🟢 {b_pct}%</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
                <div style="width:{r_pct}%;height:10px;background:#c62828;
                            border-radius:4px;min-width:2px"></div>
                <span style="font-size:11px;color:#ef9a9a">🔴 {r_pct}%</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px">
                <div style="width:{n_pct}%;height:10px;background:#546e7a;
                            border-radius:4px;min-width:2px"></div>
                <span style="font-size:11px;color:#b0bec5">⚪ {n_pct}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")


# ─────────────────────────────────────────────────────────────
# FEATURE 5 — SOURCE CHART (sidebar)
# ─────────────────────────────────────────────────────────────

def render_source_chart(articles: list):
    """
    Bar chart in sidebar showing article counts by source.
    Groups all ET feeds under "Economic Times" and all MC feeds
    under "MoneyControl" for a cleaner two-bar view.
    """
    if not articles:
        return

    source_map = {}
    for art in articles:
        raw = art.get("source", "Other")
        if raw.startswith("ET"):
            label = "Economic Times"
        elif raw.startswith("MoneyControl"):
            label = "MoneyControl"
        else:
            label = raw
        source_map[label] = source_map.get(label, 0) + 1

    st.sidebar.markdown("### 📰 Source Breakdown")
    # st.sidebar.bar_chart expects a dict or dataframe
    st.sidebar.bar_chart(source_map)
    st.sidebar.markdown("---")


# ─────────────────────────────────────────────────────────────
# EXISTING SIDEBAR STATS
# ─────────────────────────────────────────────────────────────

def render_sidebar_stats(articles: list):
    st.sidebar.markdown("### 📊 Category Breakdown")
    counts = Counter(a["category"] for a in articles)
    for cat in ["Stock Market", "Business", "Economy", "Technology", "Others"]:
        n   = counts.get(cat, 0)
        bar = "█" * n + "░" * max(0, 10 - n)
        st.sidebar.markdown(f"`{cat:<13}` {bar} {n}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏷️ Top Mentioned Tickers")
    all_t = [t for a in articles for t in a["tickers"]]
    freq  = Counter(all_t).most_common(8)
    if freq:
        for ticker, n in freq:
            st.sidebar.markdown(f"**{ticker}** — {n} articles")
    else:
        st.sidebar.markdown("_No tickers detected_")
    st.sidebar.markdown("---")


# ─────────────────────────────────────────────────────────────
# FEATURE 2 — ASK THE NEWS (inline expander)
# ─────────────────────────────────────────────────────────────

def render_ask_the_news(articles: list):
    """
    Expander below the ticker bar. User types a question; Gemini answers
    using the loaded article summaries as context.

    This is a single user-triggered Gemini call and is completely separate
    from the automatic pipeline — it will never cause rate-limit issues.
    """
    with st.expander("🤖 Ask the News — Ask anything about today's articles"):
        st.caption(
            "Powered by Gemini · Answers are grounded in today's loaded articles only"
        )
        question = st.text_input(
            "Your question:",
            placeholder="e.g. What is happening with HDFC Bank? "
                        "Which sectors are most in the news today?",
            key="ask_news_input",
        )
        if st.button("Ask", key="ask_news_btn"):
            if not question.strip():
                st.warning("Please type a question first.")
            else:
                with st.spinner("Thinking…"):
                    answer = ask_news_question(question, articles)
                st.markdown(f"**Answer:** {answer}")


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    st.title("📈 Stock-Market News Navigator")
    st.caption("Live Indian financial news · Zero-shot classified · NSE ticker mapped")

    # ── Sidebar ────────────────────────────────────────────────
    st.sidebar.title("⚙️ Controls")

    search_query = st.sidebar.text_input(
        "🔍 Search ticker / keyword",
        placeholder="e.g. RELIANCE",
    ).strip().upper()

    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Refresh News", use_container_width=True):
        st.session_state.pop("articles",   None)
        st.session_state.pop("last_fetch", None)
        # Note: bookmarks are intentionally kept across refreshes

    if "last_fetch" in st.session_state:
        st.sidebar.caption(f"Last fetched: {st.session_state.last_fetch}")

    st.sidebar.markdown("---")

    # ── Load data ─────────────────────────────────────────────
    with st.spinner("Initialising pipeline…"):
        ensure_data_loaded()

    all_articles: list = st.session_state.articles

    # ── Sidebar features ──────────────────────────────────────
    render_market_mood(all_articles)        # Feature 1
    render_sidebar_stats(all_articles)
    render_source_chart(all_articles)       # Feature 5

    # ── Search filter (applies to tab views only) ─────────────
    view_articles = all_articles
    if search_query:
        view_articles = [
            a for a in all_articles
            if any(search_query in t for t in a["tickers"])
            or search_query in a["title"].upper()
        ]
        st.info(f"Showing {len(view_articles)} results for **{search_query}**")

    # ── Scrolling ticker bar ──────────────────────────────────
    render_ticker_bar(view_articles)

    # ── Feature 2: Ask the News ────────────────────────────────
    render_ask_the_news(all_articles)

    st.markdown(
        f"**{len(view_articles)} articles** loaded · "
        "Tab switches are instant (no re-fetch)"
    )

    # ── Category + Saved tabs ─────────────────────────────────
    # Feature 4: append "⭐ Saved" tab dynamically
    saved_ids = st.session_state.bookmarks
    saved_count = sum(
        1 for a in all_articles
        if a.get("link", a["title"]) in saved_ids
    )
    tab_labels  = BASE_CATEGORIES + [f"⭐ Saved ({saved_count})"]
    tabs        = st.tabs(tab_labels)

    for tab, category in zip(tabs[:-1], BASE_CATEGORIES):
        with tab:
            filtered = (
                view_articles if category == "All"
                else [a for a in view_articles if a["category"] == category]
            )
            if not filtered:
                st.markdown("_No articles in this category right now._")
                continue
            tab_slug = category.replace(" ", "_")
            for i, art in enumerate(filtered):
                render_article_card(art, card_key=f"{tab_slug}_{i}", show_bookmark=True)

    # ── Saved tab ─────────────────────────────────────────────
    with tabs[-1]:
        saved_articles = [
            a for a in all_articles
            if a.get("link", a["title"]) in saved_ids
        ]
        if not saved_articles:
            st.markdown("_No saved articles yet. Click ☆ on any article to save it._")
        else:
            st.markdown(f"**{len(saved_articles)} saved articles**")
            for i, art in enumerate(saved_articles):
                render_article_card(art, card_key=f"saved_{i}", show_bookmark=True)

    # ── CSV download ──────────────────────────────────────────
    st.markdown("---")
    import csv, io
    buf = io.StringIO()
    w   = csv.DictWriter(
        buf,
        fieldnames=["title", "category", "sentiment", "freshness",
                    "tickers", "source", "published", "link"],
    )
    w.writeheader()
    for a in all_articles:
        w.writerow({
            "title":     a["title"],
            "category":  a["category"],
            "sentiment": a["sentiment"],
            "freshness": a.get("freshness", ""),
            "tickers":   get_ticker_display(a["tickers"]),
            "source":    a["source"],
            "published": a["published"],
            "link":      a["link"],
        })

    st.download_button(
        label="⬇️ Download today's news as CSV",
        data=buf.getvalue(),
        file_name=f"news_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()