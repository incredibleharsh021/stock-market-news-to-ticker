"""
ml_models.py — ML Pipeline for Classification and Summarization

PURPOSE:
    Provides two key ML capabilities:
    1. Zero-Shot Classification: Categorize news articles without any training
    2. Summarization: Generate 2-3 line summaries of articles (batched)
    3. "Ask the News": Answer a user question from loaded article context

HOW IT WORKS:
    - Classification: BART-MNLI zero-shot (no training needed).
    - Summarization: All articles sent to Gemini 2.5 Flash in ONE prompt
      using XML <summary> tags for robust parsing. Completely avoids
      free-tier rate limits (1 call instead of 30).
    - Ask the News: User types a question; we pass all loaded summaries
      as context in a single Gemini call and get a grounded answer.

CHANGES vs. PREVIOUS VERSION:
    - Partial mismatch fix in batch_summarize_with_gemini: if Gemini
      returns fewer summaries than articles, we no longer throw away
      the matched ones — matched articles get Gemini summaries,
      unmatched get extractive fallbacks.
    - Added ask_news_question() for the "Ask the News" chatbot feature.

Set GEMINI_API_KEY in the environment or Streamlit secrets to enable Gemini.
"""

import os
import re
from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()


# ============================================================
# Category labels for zero-shot classification
# ============================================================
CATEGORY_LABELS = [
    "Stock Market",
    "Business",
    "Economy",
    "Technology",
    "Others",
]

_GEMINI_UNAVAILABLE = object()   # sentinel: "tried and failed, don't retry"

_classifier    = None   # BART-MNLI pipeline singleton
_gemini_client = None   # Gemini client singleton OR _GEMINI_UNAVAILABLE


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_classifier():
    global _classifier
    if _classifier is None:
        print("📦 Loading BART-MNLI (first run may take ~20 s)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,   # CPU — works on any laptop or Colab free tier
        )
        print("✅ BART-MNLI loaded")
    return _classifier


def setup_gemini():
    """
    Initialise Gemini client. Uses sentinel to avoid retrying after failure.

    Reads GEMINI_API_KEY from the environment or Streamlit secrets.
    """
    global _gemini_client

    if _gemini_client is _GEMINI_UNAVAILABLE:
        return None
    if _gemini_client is not None:
        return _gemini_client

    # ── Key resolution ─────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass

    if not api_key:
        print("ℹ️  No GEMINI_API_KEY — extractive fallback will be used")
        _gemini_client = _GEMINI_UNAVAILABLE
        return None

    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        print("✅ Gemini API initialised")
        return _gemini_client
    except Exception as e:
        print(f"⚠️  Gemini setup failed: {e}")
        _gemini_client = _GEMINI_UNAVAILABLE
        return None


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_article(title: str, description: str) -> tuple:
    """Zero-shot classify one article. Returns (category, confidence)."""
    text = f"{title}. {description}".strip()
    if not text or text == ".":
        return "Others", 0.0
    text = text[:512]
    try:
        result = load_classifier()(text, candidate_labels=CATEGORY_LABELS)
        return result["labels"][0], round(result["scores"][0], 3)
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return "Others", 0.0


def batch_classify(articles: list) -> list:
    """Classify a list of article dicts. Returns [(category, confidence), ...]."""
    return [
        classify_article(a.get("title", ""), a.get("description", ""))
        for a in articles
    ]


# ─────────────────────────────────────────────────────────────
# SUMMARISATION — BATCHED (primary path)
# ─────────────────────────────────────────────────────────────

def batch_summarize_with_gemini(articles: list) -> list | None:
    """
    Send all articles to Gemini in ONE prompt. Parse with <summary> XML tags.

    PARTIAL MISMATCH FIX:
        If Gemini returns fewer summaries than articles (e.g. it truncated),
        we no longer discard everything. Instead we use what we got and
        return None only for the extras — the caller fills those with
        extractive fallbacks.

    Returns:
        list[str | None]  — one entry per article; None means "use fallback"
        None              — if the API call itself failed
    """
    if not articles:
        return []

    client = setup_gemini()
    if client is None:
        return None

    prompt = (
        "Summarize each of the following Indian financial news articles in "
        "exactly 2-3 short sentences. Focus on key financial impact and "
        "market relevance for a retail investor.\n\n"
        "IMPORTANT: Wrap each summary in exactly one <summary> tag like this:\n"
        "<summary>Your 2-3 sentence summary here.</summary>\n\n"
    )
    for i, art in enumerate(articles):
        prompt += (
            f"Article {i+1}:\n"
            f"Title: {art.get('title', '')}\n"
            f"Description: {art.get('description', '')[:300]}\n\n"
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        matches   = re.findall(r'<summary>(.*?)</summary>',
                               response.text, flags=re.DOTALL)
        summaries = [m.strip() for m in matches]

        # Build a per-article result list (None = use extractive fallback)
        result = []
        for i in range(len(articles)):
            result.append(summaries[i] if i < len(summaries) else None)

        gemini_count = len(summaries)
        if gemini_count == len(articles):
            print("✅ Gemini batch summarisation: all articles covered")
        else:
            print(f"⚠️  Gemini returned {gemini_count}/{len(articles)} summaries "
                  f"— extractive fallback for the rest")
        return result

    except Exception as e:
        print(f"⚠️  Gemini batch call failed: {e}")
        return None


def summarize_extractive(title: str, description: str) -> str:
    """Instant extractive fallback — no model, no API, always returns something."""
    text = f"{title}. {description}".strip()
    if len(text) <= 250:
        return text
    cut = text[:250].rfind(".")
    return text[:cut + 1] if cut > 100 else text[:250] + "…"


def batch_summarize(articles: list) -> list:
    """
    Public entry point for summarisation.

    Strategy:
        1. Try Gemini batch (1 API call).
        2. For any article where Gemini returned None (partial mismatch
           or full failure), substitute extractive summary.

    Always returns a list of the same length as `articles`.
    """
    gemini_results = batch_summarize_with_gemini(articles)

    summaries = []
    for i, art in enumerate(articles):
        # gemini_results is None  → full failure, use extractive for all
        # gemini_results[i] is None → partial mismatch, use extractive for this one
        gemini_val = None if gemini_results is None else gemini_results[i]
        if gemini_val:
            summaries.append(gemini_val)
        else:
            summaries.append(
                summarize_extractive(art.get("title", ""), art.get("description", ""))
            )

    return summaries


# ─────────────────────────────────────────────────────────────
# ASK THE NEWS  (NEW — powers the "Ask the News" chatbot)
# ─────────────────────────────────────────────────────────────

def ask_news_question(question: str, articles: list) -> str:
    """
    Answer a user question grounded in the currently loaded articles.

    HOW IT WORKS:
        1. Build a context block from article titles + summaries (already
           computed — no extra I/O needed).
        2. Send ONE Gemini call with the question and context.
        3. Gemini answers only from what the articles say, so hallucination
           is minimised.

    This is a single user-triggered API call — not part of the automatic
    pipeline — so it never contributes to rate-limit pressure.

    Args:
        question (str):        User's natural language question.
        articles (list[dict]): Processed article dicts with 'title' and
                               'summary' keys.

    Returns:
        str: Grounded answer, or an informative fallback string.
    """
    client = setup_gemini()
    if client is None:
        return "⚠️ Gemini is not available. Please set GEMINI_API_KEY to use this feature."

    if not articles:
        return "No articles are loaded yet. Please refresh the news first."

    # Build context from titles + summaries (already in memory)
    context_lines = []
    for i, art in enumerate(articles, 1):
        title   = art.get("title",   "").strip()
        summary = art.get("summary", "").strip()
        tickers = ", ".join(art.get("tickers", [])) or "—"
        context_lines.append(
            f"[{i}] {title}\n"
            f"     Tickers: {tickers}\n"
            f"     Summary: {summary}"
        )
    context = "\n\n".join(context_lines)

    prompt = (
        "You are a concise Indian financial news analyst. "
        "Answer the user's question using ONLY the news articles provided below. "
        "If the answer is not found in the articles, say so clearly. "
        "Keep your answer to 3-4 sentences maximum.\n\n"
        f"=== TODAY'S NEWS ARTICLES ===\n{context}\n\n"
        f"=== USER QUESTION ===\n{question}\n\n"
        "Answer:"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Could not get an answer: {e}"