# 📈 Indian Stock-Market News Navigator

**SMAI Assignment 3 | Topic T9.5 | IIIT Hyderabad**

A Streamlit web app that fetches Indian financial news via RSS, classifies articles into categories, generates AI-powered summaries, and extracts relevant NSE stock tickers.

---


https://github.com/user-attachments/assets/9fd9377b-cdeb-4b18-8d77-aea7e5d66277



## 🎯 Features

- **Live RSS Feeds**: Fetches latest news from MoneyControl & Economic Times
- **Zero-Shot Classification**: Categorizes articles (Stock Market / Business / Economy / Technology / Others) using BART-MNLI
- **AI Summarization**: Generates 2-3 line summaries using Gemini Flash (with HuggingFace fallback)
- **Ticker Extraction**: Identifies NSE stock tickers from article text using curated dictionary
- **Interactive Filters**: Filter by category or specific ticker symbol
- **Dark-Themed UI**: Premium card-based design with hover effects

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/incredibleharsh021/stock-market-news-to-ticker.git
cd stock-market-news-to-ticker
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Set Gemini API Key

For better summarization quality, get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey):

```bash
export GEMINI_API_KEY="your-api-key-here"
```

> **Note**: The app works without the API key — it automatically falls back to HuggingFace BART-CNN for summarization.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📂 Project Structure

```
SMAI proj 3/
├── app.py                          # Main Streamlit application
├── training_evaluation.ipynb       # Training & evaluation notebook
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── utils/
│   ├── __init__.py                 # Package init
│   ├── fetcher.py                  # RSS feed fetching
│   ├── ml_models.py                # Classification + Summarization pipeline
│   └── ticker_map.py               # Ticker extraction logic
├── data/
│   └── company_tickers.json        # Company name → NSE ticker mapping
├── evaluation/
│   └── evaluation_script.py        # Model accuracy evaluation
└── report/
    ├── technical_summary.md        # Technical report (Intro/Data/Method/Results/Limitations)
    └── one_slide_pitch.png         # LinkedIn-ready project pitch
```

---

## 🔧 Tech Stack

| Component | Tool | Why |
|---|---|---|
| UI Framework | Streamlit | Simple, Python-native, free hosting |
| RSS Parsing | feedparser | Standard RSS library, no scraping |
| Classification | facebook/bart-large-mnli | Zero-shot, no training needed |
| Summarization (Primary) | Google Gemini 1.5 Flash | Fast, high quality, free tier |
| Summarization (Fallback) | facebook/bart-large-cnn | Local, always works |
| Ticker Extraction | Custom dictionary | Simple, explainable, reliable |

---

## 📊 Running Evaluation

### Option 1: Run the Jupyter Notebook (recommended)
```bash
jupyter notebook training_evaluation.ipynb
```

### Option 2: Run the CLI evaluation script
```bash
python evaluation/evaluation_script.py
```

**Output**:
- Overall accuracy on 38 hand-labeled Indian financial headlines
- Per-category precision, recall, F1 scores
- Confusion matrix
- Misclassified examples with analysis

---

## 📝 Usage Guide

### Main App (Streamlit)

#### 1. **All Articles View**
- Default landing page with all 30 fetched articles
- Each article card shows: title, source, category badge, confidence, summary, tickers, sentiment, and freshness

#### 2. **Filter by Category**
- Tabs: "All", "Stock Market", "Business", "Economy", "Technology", "Others"
- Clicking a tab filters articles dynamically
- Useful for focusing on specific market segments

#### 3. **Filter by Ticker**
- Sidebar selector: choose any extracted NSE ticker symbol
- View all articles mentioning that company
- Quickly see how a specific stock is covered across news

#### 4. **Market Mood Meter** (Sidebar)
- Color-coded progress bars showing:
  - **🟢 Bullish**: Articles with positive sentiment
  - **⚪ Neutral**: Mixed or neutral tone
  - **🔴 Bearish**: Articles with negative sentiment
- Instant gauge of overall market sentiment

#### 5. **Source Breakdown Chart** (Sidebar)
- Bar chart showing article count per RSS feed (Economic Times vs MoneyControl)
- Understand which source is most active

#### 6. **Ask the News** (Expandable)
- Type any question about the loaded articles
- Gemini API summarizes all article context and answers your question
- Example: "Which sectors are gaining?" or "What's happening with banking stocks?"
- Single API call — no rate-limit issues

#### 7. **Bookmarks** ("⭐ Saved" Tab)
- Click ⭐ on any article card to bookmark it
- All bookmarks saved in session state
- Access via "⭐ Saved" tab at the top
- Persist for the current session

#### 8. **Freshness Badges**
- **🟢 Fresh**: Published within last 2 hours
- **🟡 Recent**: Published 2–6 hours ago
- **🔴 Old**: Published >6 hours ago

---

## 🔍 Technical Details

### Classification Logic

```python
# Zero-shot classification using Natural Language Inference
text = "{title}. {description}"  # max 512 tokens
model = facebook/bart-large-mnli
candidates = ["Stock Market", "Business", "Economy", "Technology", "Others"]
result = model(text, candidates)
# Output: (predicted_category, confidence_score)
```

**Why zero-shot?**
- No training data needed
- Works on custom categories
- Generalizes to new article styles
- Production-ready out of the box

### Summarization Logic

**Primary (Gemini Flash):**
```python
# Batch all 30 articles in ONE request using XML tags
prompt = """
Provide 2-3 line summaries for each article using <summary> tags:

<article>
Title: {title}
Text: {text}
</article>

<summary>{2-3 lines}</summary>
...
"""
# Parsed with regex: <summary>...</summary>
```

**Fallback (BART-CNN):**
```python
# If Gemini unavailable, use local HuggingFace model
# Extractive summarization: selects key sentences from original text
model = facebook/bart-large-cnn
summary = model(text, max_length=100)
```

### Ticker Extraction — Two-Phase

**Phase 1 — Exact Matching (regex word-boundary):**
```python
# Fast, zero false positives
# Regex: \b(Bajaj Finance|TCS|ICICI Bank)\b
```

**Phase 2 — Fuzzy Fallback (RapidFuzz):**
```python
# Only triggered if Phase 1 finds nothing
# Uses WRatio (token-level matching) with threshold 88
# Handles: "Bajaj Fin" → "Bajaj Finance", abbreviations, typos
# Guards: only names ≥ 8 chars, excludes generic words
```

**Why two-phase?**
- Exact: Fast, precise, handles common names
- Fuzzy: Catches variants and edge cases
- Combined: Best precision without sacrificing recall

### Sentiment Analysis

**Keyword-based**:
```python
bullish_words = ["surge", "jump", "rally", "record", "gain", "profit", ...]
bearish_words = ["fall", "drop", "slump", "loss", "decline", "cut", ...]
sentiment = "🟢 Bullish" if bullish_count > bearish_count else "🔴 Bearish"
```

### Freshness Calculation

```python
age_hours = (now_utc - article_publish_time).total_seconds() / 3600
if age_hours < 2:    return "🟢 Fresh"
if age_hours < 6:    return "🟡 Recent"
return "🔴 Old"
```

---

## 📦 Dependencies Explained

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.30.0 | Web UI framework |
| `feedparser` | ≥6.0.0 | RSS feed parsing (no scraping) |
| `transformers` | ≥4.36.0 | BART models (classification, summarization) |
| `torch` | ≥2.0.0 | Deep learning backend for transformers |
| `google-genai` | ≥1.0.0 | Gemini API client |
| `pandas` | ≥2.0.0 | Data manipulation |
| `scikit-learn` | ≥1.3.0 | Evaluation metrics (confusion matrix, F1, etc.) |
| `matplotlib` | ≥3.7.0 | Plotting (eval metrics) |
| `seaborn` | ≥0.12.0 | Enhanced plotting |
| `rapidfuzz` | (optional) | Fuzzy ticker matching (gracefully disabled if missing) |

---

## 🛠️ Development Notes

### Adding a New RSS Feed

Edit [utils/fetcher.py](utils/fetcher.py#L22-L31):

```python
RSS_FEEDS = {
    "Your Feed Name": "https://example.com/rss.xml",
    ...
}
```

### Adding a New Classification Category

Edit [utils/ml_models.py](utils/ml_models.py#L17-L21):

```python
CATEGORY_LABELS = [
    "Stock Market",
    "Your New Category",
    ...
]
```

Then update [app.py](app.py#L95-L96):

```python
BASE_CATEGORIES = ["All", "Your New Category", ...]
CATEGORY_COLOURS = {"Your New Category": "badge-blue", ...}
```

### Updating the Ticker Dictionary

Replace [data/nse_equities.csv](data/nse_equities.csv) with a fresh list from [NSE India](https://www.nseindia.com/market-data/securities-available-for-trading).

The CSV must have columns: `SYMBOL`, `NAME OF COMPANY`.

---

## 🎓 Project Context

**Assignment**: SMAI T9.5 — "Stock-Market News → Ticker"  
**Institution**: IIIT Hyderabad  
**Tier**: 1  
**Learning Goals**:
- ✅ Zero-shot text classification (no labeled training data)
- ✅ Generative summarization (Gemini API + fallback)
- ✅ Entity extraction (ticker symbols)
- ✅ Streamlit web app development
- ✅ RSS feed parsing (not web scraping)
- ✅ Production evaluation (38-article test set, confusion matrix, F1 scores)

---

## 📊 Evaluation Results

**Test Set**: 38 hand-labeled Indian financial headlines

**Metrics Reported** (from [evaluation_script.py](evaluation/evaluation_script.py)):
- Overall Accuracy
- Per-category Precision, Recall, F1
- Confusion Matrix

**Note**: 38 samples intentionally small — demonstrates methodology, not claims state-of-the-art accuracy.

---

## ⚠️ Known Limitations & Future Work

### Current Limitations
1. **Ticker coverage**: Only ~130 companies (Nifty 50 + selected mid-caps)
   - *Fix*: Expand NSE CSV with full securities list
2. **Sentiment**: Keyword-based (no deep learning)
   - *Enhancement*: Use pre-trained sentiment classifier (DistilBERT)
3. **Summarization**: Extractive fallback (BART-CNN) may be redundant
   - *Future*: Cache Gemini responses to reduce API calls
4. **Rate limiting**: Gemini free tier capped at ~2000 requests/month
   - *Mitigation*: Batch all articles in single call (already implemented)

### Potential Enhancements
- [ ] Fine-tune BART-MNLI on Indian financial terminology
- [ ] Add price impact prediction (correlate sentiment with stock returns)
- [ ] Multi-language support (Hindi, Tamil, Telugu)
- [ ] Export bookmarks to CSV / email digest
- [ ] Historical sentiment tracking (trend graphs)
- [ ] Dark/light theme toggle

---

## 📄 Files Reference

- [app.py](app.py) — Main Streamlit application (news fetching → UI rendering)
- [training_evaluation.ipynb](training_evaluation.ipynb) — Jupyter notebook for model training & evaluation
- [evaluation/evaluation_script.py](evaluation/evaluation_script.py) — CLI evaluation script (38-article benchmark)
- [utils/fetcher.py](utils/fetcher.py) — RSS feed fetching & cleaning
- [utils/ml_models.py](utils/ml_models.py) — Classification, summarization, "Ask the News"
- [utils/ticker_map.py](utils/ticker_map.py) — Ticker extraction (exact + fuzzy matching)
- [data/nse_equities.csv](data/nse_equities.csv) — Company → NSE ticker mapping
- [report/technical_summary.md](report/technical_summary.md) — Detailed methodology report

---

## 🤝 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a pull request

**Before pushing to GitHub**:

---

## 📜 License

This project is part of SMAI Course Assignment 3. Use for educational purposes.

---



Both methods test zero-shot classification on ~38 curated Indian financial headlines and report:
- Overall accuracy & per-category accuracy
- Confusion matrix visualization
- Misclassified examples analysis
- Limitations discussion

---


## ⚠️ Limitations

- **RSS Feed Availability**: Some feeds may be temporarily unavailable
- **Ticker Extraction**: Dictionary-based — only detects ~100 known companies
- **Zero-Shot Accuracy**: May misclassify ambiguous articles (e.g., Economy vs Business overlap)
- **Summarization**: Gemini requires API key; fallback model is slower

---

## 🙏 Acknowledgements

- **Models**: Meta AI (BART-MNLI, BART-CNN), Google (Gemini 1.5 Flash)
- **Data Sources**: MoneyControl, Economic Times (public RSS feeds)
- **Ticker Data**: NSE India (securities list)
- **Tools**: Claude AI was used for code scaffolding and report drafting

---

## Team 
- **Member 1**: Sangarsu Abhijith (2025202032)
- **Member 2**: Harsh Jaiswal (2025204011)