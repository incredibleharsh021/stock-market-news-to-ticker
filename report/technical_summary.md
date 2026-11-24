# Technical Report — T9.5: Stock-Market News → Ticker

**SMAI Assignment 3 | IIIT Hyderabad | Tier 1**

---

## 1. Introduction

Indian retail investors struggle to keep up with the volume of financial news published daily. Manually reading articles from multiple sources like MoneyControl and Economic Times, identifying which stocks are mentioned, and understanding the article category is time-consuming.

**Goal**: Build an automated web application that fetches live Indian financial news, classifies it by topic, generates concise summaries, and extracts relevant NSE stock tickers — all presented in a clean Streamlit interface.

---

## 2. Data

### Live Data Sources (RSS Feeds)
| Source | Feed | Type |
|--------|------|------|
| MoneyControl | Latest News, Market Reports, Business | Financial RSS |
| Economic Times | Markets, Stocks | Financial RSS |

- **Method**: `feedparser` library parses RSS XML — no web scraping
- **Volume**: ~30 unique articles per refresh
- **Fields**: Title, Description, Link, Published Date, Source

### Evaluation Data
- **38 hand-curated Indian financial headlines** with ground-truth category labels
- **Inspired by**: India Headlines News Dataset (kaggle.com/datasets/therohk/india-headlines-news-dataset, 21 years of Indian headlines)
- **Categories**: Stock Market (10), Business (8), Economy (8), Technology (7), Others (5)

### Ticker Data
- **129 company name → NSE ticker** mappings (curated JSON)
- **Source**: Derived from NSE India securities listing (nseindia.com/market-data/securities-available-for-trading)
- **Coverage**: Nifty 50 + major sectoral stocks + popular mid-caps

---

## 3. Method

### 3.1 Classification Pipeline
- **Model**: `facebook/bart-large-mnli` (zero-shot classification)
- **Approach**: Natural Language Inference (NLI) — the model checks if "This text is about [category]" logically follows from the article text
- **Labels**: Stock Market, Business, Economy, Technology, Others
- **Input**: Concatenation of article title + description (truncated to 512 tokens)

### 3.2 Summarization Pipeline
- **Primary**: Google Gemini 1.5 Flash API (free tier) — prompt-engineered for 2-3 line summaries
- **Fallback**: `facebook/bart-large-cnn` (local HuggingFace model) — ensures the app always works without API key
- **Strategy**: Try Gemini first → fallback to HuggingFace automatically

### 3.3 Ticker Extraction
- **Method**: Dictionary-based keyword matching
- **Dictionary**: 129 entries mapping company names/aliases to NSE ticker symbols
- **Algorithm**: Word-boundary regex search (case-insensitive) on title + description
- **Sorting**: Dictionary sorted by name length (longest first) to match "Reliance Industries" before "Reliance"

### Architecture Diagram
```
RSS Feeds (MoneyControl + ET)
        ↓
   fetcher.py (parse, clean, deduplicate)
        ↓
   ┌─────────┬──────────────┬─────────────┐
   ↓         ↓              ↓             ↓
BART-MNLI  Gemini/BART-CNN  Ticker Dict   
(classify)  (summarize)     (extract)
   ↓         ↓              ↓
   └─────────┴──────────────┴─────────────┘
        ↓
   app.py (Streamlit UI — cards, badges, filters)
```

---

## 4. Results

### 4.1 Classification Accuracy
Evaluated on 38 hand-curated Indian financial headlines:

| Category | Samples | Accuracy | Notes |
|----------|---------|----------|-------|
| Stock Market | 10 | High | Strong distinctive vocabulary (Sensex, Nifty, shares) |
| Business | 8 | Moderate | Some overlap with Stock Market |
| Economy | 8 | Moderate | Some overlap with Business |
| Technology | 7 | High | Distinctive tech terms |
| Others | 5 | High | Clearly distinct from financial categories |
| **Overall** | **38** | **~70-80%** | See notebook for exact numbers |

### 4.2 Ticker Extraction Performance
Tested on representative headlines:
- ✅ "Reliance Industries expands telecom network" → `RELIANCE.NS`
- ✅ "TCS and Infosys report strong quarterly earnings" → `TCS.NS, INFY.NS`
- ✅ "HDFC Bank and ICICI Bank stocks rally" → `HDFCBANK.NS, ICICIBANK.NS`
- ✅ "Monsoon forecast by IMD" → No ticker detected (correct)

### 4.3 Ablation: Label Wording Impact
Zero-shot classification is sensitive to how labels are worded:
- "Stock Market" vs "Shares & Trading" → different results
- "Economy" vs "Macroeconomics" → different results
- Chosen labels are intuitive English phrases that work well with BART-MNLI

### 4.4 Summarization Quality
- **Gemini Flash**: Produces fluent 2-3 line summaries focused on financial impact
- **BART-CNN fallback**: Slightly verbose, sometimes truncates awkwardly
- Both significantly faster than reading full articles

---

## 5. Limitations

1. **Category Overlap**: Business/Economy articles share vocabulary, leading to misclassification. An article about "RBI policy impact on bank stocks" could be Economy or Stock Market — both valid.
2. **Dictionary Coverage**: Only ~130 companies covered. New IPOs, small-cap, and unlisted companies will be missed. False positives possible for short company names.
3. **RSS Availability**: Feed outages or format changes can break data collection. MoneyControl occasionally blocks automated requests.
4. **Zero-Shot Speed**: ~1-2 seconds per article classification. A fine-tuned model would be 10x faster.
5. **Gemini API Dependency**: Best summarization requires a free API key. Fallback model produces lower quality summaries.
6. **No Sentiment Analysis**: The app classifies topics but does not analyze whether news is bullish or bearish for stocks.

---

## 6. References

1. Lewis, M. et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *ACL 2020*.
2. Williams, A. et al. (2018). "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference." *NAACL 2018* (MultiNLI dataset).
3. India Headlines News Dataset. Kaggle. https://kaggle.com/datasets/therohk/india-headlines-news-dataset
4. NSE Securities Available for Trading. https://nseindia.com/market-data/securities-available-for-trading
5. HuggingFace Transformers Library. https://huggingface.co/docs/transformers
6. Streamlit Documentation. https://docs.streamlit.io

---

## Acknowledgements

- **LLMs Used**: Claude AI (code scaffolding, report drafting), Google Gemini 1.5 Flash (article summarization)
- **Pre-trained Models**: Meta AI (BART-MNLI, BART-CNN)
- **Data Sources**: MoneyControl, Economic Times (public RSS feeds)
