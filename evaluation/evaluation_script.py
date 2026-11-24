"""
evaluation_script.py — Zero-Shot Classification Evaluation

SMAI Assignment 3 | Topic T9.5 | IIIT Hyderabad

PURPOSE:
    Evaluate the zero-shot classification model (BART-MNLI) on a
    hand-curated set of 38 Indian financial headlines.
    
    Produces:
        - Overall accuracy
        - Per-category precision / recall / F1
        - Confusion matrix (console)
        - List of misclassified examples with analysis

HOW TO RUN:
    python evaluation/evaluation_script.py

DATASET:
    38 manually labelled Indian financial headlines.
    Inspired by the India Headlines News Dataset
    (kaggle.com/datasets/therohk/india-headlines-news-dataset).

NOTE:
    38 samples is intentionally small — sufficient for Tier 1 to
    demonstrate evaluation methodology, not claim SOTA accuracy.
"""

import sys
import os
from collections import defaultdict

# ── Path setup so we can import from utils/ ──────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ml_models import classify_article, CATEGORY_LABELS


# ============================================================
# Hand-curated evaluation dataset
# Format: (title, description, ground_truth_category)
# ============================================================
EVAL_DATASET = [
    # ── Stock Market (10) ────────────────────────────────────
    ("Sensex jumps 750 points as IT stocks rally",
     "Indian benchmark indices surged on strong quarterly earnings from TCS and Infosys",
     "Stock Market"),

    ("Nifty hits all-time high of 24,000",
     "The NSE Nifty 50 crossed the 24,000 mark for the first time, driven by banking and IT stocks",
     "Stock Market"),

    ("FIIs pull out Rs 5,000 crore from Indian markets",
     "Foreign institutional investors continued selling Indian equities amid rising US bond yields",
     "Stock Market"),

    ("Reliance shares surge 5% after Q2 results beat estimates",
     "Reliance Industries reported strong quarterly numbers with Jio subscriber additions ahead of expectations",
     "Stock Market"),

    ("IPO market booms as 5 new issues hit the market this week",
     "Five companies including a fintech startup and an EV maker are launching their IPOs this week on NSE and BSE",
     "Stock Market"),

    ("HDFC Bank stock falls 3% on weak loan growth guidance",
     "HDFC Bank shares declined after the management issued cautious outlook on credit growth",
     "Stock Market"),

    ("Midcap index outperforms Nifty 50 by wide margin",
     "Small and mid-cap stocks have delivered 40% returns this year versus 15% for large-cap Nifty 50",
     "Stock Market"),

    ("Adani Group stocks recover after credit agency upgrades outlook",
     "Adani Enterprises and Adani Ports gained on positive credit rating revision",
     "Stock Market"),

    ("Mutual fund SIP investments hit record Rs 20,000 crore monthly",
     "Indian retail investors continue to pour money into systematic investment plans",
     "Stock Market"),

    ("Bears tighten grip as Sensex falls below 70,000",
     "Markets slumped on global cues with Sensex losing over 1000 points in intraday trade",
     "Stock Market"),

    # ── Business (8) ─────────────────────────────────────────
    ("Tata Group acquires Air India for Rs 18,000 crore",
     "Tata Sons won the bid to acquire the national carrier, marking its return to aviation",
     "Business"),

    ("Zomato reports first-ever quarterly profit",
     "The food delivery platform turned profitable driven by Blinkit growth and improved unit economics",
     "Business"),

    ("Amazon India expands warehouse network to 100 cities",
     "The e-commerce giant is investing Rs 5,000 crore to build fulfillment centers in tier-2 cities",
     "Business"),

    ("Infosys wins $1.5 billion deal from European bank",
     "Infosys has signed its largest deal ever for digital transformation of a major European bank",
     "Business"),

    ("Ola Electric files for Rs 5,500 crore IPO",
     "The electric vehicle maker is planning to raise capital for expanding manufacturing capacity",
     "Business"),

    ("Flipkart valuation drops to $35 billion in latest funding round",
     "Walmart-backed Flipkart saw its valuation cut as investors reassess e-commerce growth",
     "Business"),

    ("Maruti Suzuki launches new hybrid SUV at Rs 12 lakh",
     "India's largest carmaker unveiled a strong hybrid vehicle targeting the growing SUV segment",
     "Business"),

    ("Bharti Airtel raises tariffs by 15% across all plans",
     "The telecom operator increased mobile recharge prices citing rising input costs",
     "Business"),

    # ── Economy (8) ──────────────────────────────────────────
    ("RBI cuts repo rate by 25 basis points to 6.25%",
     "The Reserve Bank of India reduced the key lending rate for the first time in two years",
     "Economy"),

    ("India GDP grows 7.6% in Q3, beats all estimates",
     "The Indian economy surprised analysts with robust manufacturing and services performance",
     "Economy"),

    ("Inflation rises to 5.5% in March on higher food prices",
     "Consumer price inflation accelerated driven by rising vegetable and cereal prices",
     "Economy"),

    ("Rupee falls to record low of 84.5 against US dollar",
     "The Indian rupee weakened amid foreign capital outflows and strengthening greenback",
     "Economy"),

    ("Government announces Rs 2 lakh crore infrastructure push",
     "Finance ministry allocated major spending for roads, railways and urban infrastructure",
     "Economy"),

    ("India's fiscal deficit widens to 5.8% of GDP",
     "Government deficit exceeded targets due to higher subsidy spending and lower tax collections",
     "Economy"),

    ("GST collections cross Rs 2 lakh crore for first time",
     "GST revenue hit a record high indicating strong economic activity and compliance",
     "Economy"),

    ("Crude oil prices surge to $100, India faces inflation risk",
     "Rising crude prices threaten India's current account deficit and may push up transport costs",
     "Economy"),

    # ── Technology (7) ───────────────────────────────────────
    ("ISRO successfully launches Gaganyaan test module",
     "India's space agency completed a critical uncrewed test of its human spaceflight program",
     "Technology"),

    ("Indian startups raise $10 billion in 2024 despite funding winter",
     "India's startup ecosystem showed resilience with AI and deeptech companies attracting investment",
     "Technology"),

    ("UPI transactions cross 15 billion monthly milestone",
     "Digital payments through UPI continue to grow rapidly with increasing rural adoption",
     "Technology"),

    ("India launches 6G research initiative with IITs",
     "Government has allocated Rs 500 crore for 6G research in partnership with IITs",
     "Technology"),

    ("Wipro launches AI-powered cybersecurity platform",
     "The IT major unveiled a new artificial intelligence based security solution for enterprises",
     "Technology"),

    ("ChatGPT alternatives: Indian AI startups building local language models",
     "Multiple Indian startups are developing homegrown LLMs for Hindi and regional languages",
     "Technology"),

    ("Semiconductor fab in Gujarat gets Cabinet approval",
     "The Tata-PSMC joint venture for chip manufacturing received fast-track government clearance",
     "Technology"),

    # ── Others (5) ───────────────────────────────────────────
    ("Cricket: India beats Australia to win Border-Gavaskar Trophy",
     "The Indian cricket team retained the trophy with a series win at home",
     "Others"),

    ("PM Modi inaugurates new Parliament building",
     "Prime Minister laid the foundation for democratic events in the new parliament complex",
     "Others"),

    ("Monsoon expected to be above normal this year: IMD",
     "India Meteorological Department predicts 106% of long period average rainfall",
     "Others"),

    ("Bollywood box office: Jawan crosses Rs 1000 crore worldwide",
     "Shah Rukh Khan's action film becomes the highest grossing Indian movie of the year",
     "Others"),

    ("Delhi pollution: Air quality drops to severe category",
     "National capital recorded AQI above 400 as stubble burning worsens air quality",
     "Others"),
]


# ─────────────────────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute per-class precision, recall, F1 and overall accuracy.

    Args:
        y_true (list[str]): Ground-truth labels
        y_pred (list[str]): Predicted labels

    Returns:
        dict: {
            'accuracy': float,
            'per_class': {label: {'precision', 'recall', 'f1', 'support'}}
        }
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    per_class = {}
    for label in CATEGORY_LABELS:
        p_denom = tp[label] + fp[label]
        r_denom = tp[label] + fn[label]
        precision = tp[label] / p_denom if p_denom else 0.0
        recall    = tp[label] / r_denom if r_denom else 0.0
        f1_denom  = precision + recall
        f1        = 2 * precision * recall / f1_denom if f1_denom else 0.0
        per_class[label] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "support":   r_denom,   # = TP + FN = total true instances
        }

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return {"accuracy": round(accuracy, 4), "per_class": per_class}


def print_confusion_matrix(y_true: list, y_pred: list):
    """Print a simple ASCII confusion matrix to stdout."""
    labels = CATEGORY_LABELS
    size   = len(labels)
    matrix = [[0] * size for _ in range(size)]
    idx    = {lbl: i for i, lbl in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        if true in idx and pred in idx:
            matrix[idx[true]][idx[pred]] += 1

    col_w = 14
    print("\nConfusion Matrix (rows = True, cols = Predicted):")
    print("-" * (col_w * (size + 1) + 2))
    header = f"{'':>{col_w}}" + "".join(f"{lbl[:12]:>{col_w}}" for lbl in labels)
    print(header)
    print("-" * (col_w * (size + 1) + 2))
    for i, lbl in enumerate(labels):
        row = f"{lbl[:12]:>{col_w}}" + "".join(
            f"{matrix[i][j]:>{col_w}}" for j in range(size)
        )
        print(row)


def run_evaluation():
    """
    Classify all EVAL_DATASET samples and print a full evaluation report.

    Steps:
        1. Classify each headline via BART-MNLI
        2. Compute accuracy + per-class P/R/F1
        3. Print confusion matrix
        4. Show misclassified examples with analysis
    """
    print("=" * 70)
    print("📊  Zero-Shot Classification Evaluation — BART-MNLI")
    print("    SMAI Assignment 3 · T9.5 · IIIT Hyderabad")
    print("=" * 70)
    print(f"Model      : facebook/bart-large-mnli")
    print(f"Categories : {', '.join(CATEGORY_LABELS)}")
    print(f"Samples    : {len(EVAL_DATASET)}")
    print()

    y_true       = []
    y_pred       = []
    confidences  = []
    results      = []

    print("Running classification (this may take a minute)…")
    for i, (title, desc, true_label) in enumerate(EVAL_DATASET):
        pred_label, confidence = classify_article(title, desc)
        correct = pred_label == true_label

        y_true.append(true_label)
        y_pred.append(pred_label)
        confidences.append(confidence)
        results.append({
            "title":      title,
            "true":       true_label,
            "predicted":  pred_label,
            "confidence": confidence,
            "correct":    correct,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(EVAL_DATASET)} done…")

    # ── Overall accuracy ─────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred)
    print()
    print("=" * 70)
    print(f"📈  OVERALL ACCURACY: {metrics['accuracy']:.1%}  "
          f"({sum(r['correct'] for r in results)}/{len(results)} correct)")
    print("=" * 70)

    # ── Per-class table ──────────────────────────────────────
    print(f"\n{'Category':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("-" * 55)
    for label in CATEGORY_LABELS:
        m = metrics["per_class"][label]
        print(f"{label:<16} {m['precision']:>10.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['support']:>9}")

    # ── Confusion matrix ─────────────────────────────────────
    print_confusion_matrix(y_true, y_pred)

    # ── Confidence analysis ──────────────────────────────────
    correct_confs   = [r["confidence"] for r in results if r["correct"]]
    incorrect_confs = [r["confidence"] for r in results if not r["correct"]]
    if correct_confs:
        print(f"\nAvg confidence (correct predictions)   : "
              f"{sum(correct_confs)/len(correct_confs):.3f}")
    if incorrect_confs:
        print(f"Avg confidence (incorrect predictions) : "
              f"{sum(incorrect_confs)/len(incorrect_confs):.3f}")

    # ── Misclassified examples ───────────────────────────────
    incorrect = [r for r in results if not r["correct"]]
    if incorrect:
        print(f"\n❌  Misclassified examples ({len(incorrect)} total):")
        print("-" * 70)
        for r in incorrect:
            print(f"  Headline  : {r['title'][:65]}…")
            print(f"  True → Pred : {r['true']} → {r['predicted']} "
                  f"(confidence: {r['confidence']:.2f})")
            print()

    # ── Correct examples sample ──────────────────────────────
    correct_examples = [r for r in results if r["correct"]]
    print(f"✅  Correctly classified sample (5 of {len(correct_examples)}):")
    print("-" * 70)
    for r in correct_examples[:5]:
        print(f"  {r['title'][:65]}…")
        print(f"  → {r['predicted']} (conf: {r['confidence']:.2f})")
        print()

    # ── Analysis block ───────────────────────────────────────
    print("=" * 70)
    print("📝  WHY ZERO-SHOT WORKS (AND WHERE IT STRUGGLES)")
    print("=" * 70)
    print("""
STRENGTHS:
  • BART-MNLI is trained on NLI: it checks "Does this text imply the
    category label?" without needing any labelled financial training data.
  • Financial news uses distinctive vocabulary per category:
    "Sensex", "Nifty" → Stock Market; "GDP", "inflation" → Economy.
  • Zero training cost — ideal for Tier 1 scope.

KNOWN FAILURE MODES:
  1. Category overlap: A TCS quarterly results story is valid for both
     "Stock Market" and "Business" — the model picks one arbitrarily.
  2. Economy vs Business: Both use similar financial vocabulary, leading
     to cross-category confusion.
  3. Label sensitivity: Renaming "Stock Market" → "Equities" changes
     predictions meaningfully.
  4. Inference speed: ~1-2 s per article on CPU (BART is 1.6 GB).

IMPROVEMENT IDEAS:
  • Fine-tune a DistilBART/DistilBERT on 500+ labelled Indian headlines.
  • Hybrid approach: zero-shot + keyword rule override for high-confidence
    terms (e.g. if "Sensex" or "Nifty" appears → force "Stock Market").
  • Use Gemini API for classification too and ensemble with BART scores.
""")


if __name__ == "__main__":
    run_evaluation()