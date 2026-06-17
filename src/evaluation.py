"""
src/evaluation.py
=================
Model Evaluation Module — EduBot LLM Project
Industry: Education and Training

Evaluates the EduBot on:
  1. BLEU score (n-gram precision)
  2. ROUGE scores (recall-oriented)
  3. Perplexity (model confidence)
  4. Response quality (length, coherence, relevance)
  5. Human evaluation rubric template

Author: EduBot Project
"""

import os
import re
import json
import math
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent


# ==============================================================================
# TEXT TOKENIZATION FOR METRIC COMPUTATION
# ==============================================================================

def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for metric computation."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ==============================================================================
# BLEU SCORE
# ==============================================================================

def ngrams(tokens: List[str], n: int) -> Counter:
    """Generate n-gram counts from a list of tokens."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    reference: str,
    hypothesis: str,
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score (1-4 gram precision with brevity penalty).

    Parameters
    ----------
    reference  : ground truth reference text
    hypothesis : model-generated text
    max_n      : maximum n-gram order (BLEU-4 is standard)

    Returns
    -------
    dict with bleu_1, bleu_2, bleu_3, bleu_4, bleu_avg
    """
    ref_tokens  = simple_tokenize(reference)
    hyp_tokens  = simple_tokenize(hypothesis)

    if not hyp_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    scores = {}
    for n in range(1, max_n + 1):
        ref_ngrams = ngrams(ref_tokens, n)
        hyp_ngrams = ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            scores[f"bleu_{n}"] = 0.0
            continue

        # Clipped precision: count how many hypothesis n-grams appear in reference
        clipped_count = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        total_count   = sum(hyp_ngrams.values())
        precision     = clipped_count / total_count if total_count > 0 else 0.0

        # Smoothing to avoid log(0)
        precision = max(precision, 1e-10)
        scores[f"bleu_{n}"] = precision

    # Brevity penalty (penalize very short hypotheses)
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    # BLEU = BP × exp(weighted average of log precisions)
    log_avg = sum(math.log(max(scores[f"bleu_{n}"], 1e-10)) / max_n for n in range(1, max_n + 1))
    bleu_4  = bp * math.exp(log_avg)

    scores["bleu_4_combined"] = round(bleu_4, 4)
    return {k: round(v, 4) for k, v in scores.items()}


def corpus_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute average BLEU scores over a corpus."""
    all_scores = [bleu_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    avg = {}
    for key in all_scores[0]:
        avg[key] = round(float(np.mean([s[key] for s in all_scores])), 4)
    return avg


# ==============================================================================
# ROUGE SCORE
# ==============================================================================

def rouge_n(reference: str, hypothesis: str, n: int = 1) -> Dict[str, float]:
    """
    Compute ROUGE-N (recall-oriented n-gram overlap).

    Returns precision, recall, and F1 score.
    """
    ref_tokens = simple_tokenize(reference)
    hyp_tokens = simple_tokenize(hypothesis)

    ref_ngrams = ngrams(ref_tokens, n)
    hyp_ngrams = ngrams(hyp_tokens, n)

    if not ref_ngrams or not hyp_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = sum(min(count, hyp_ngrams[gram]) for gram, count in ref_ngrams.items())

    precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0.0
    recall    = overlap / sum(ref_ngrams.values())  if ref_ngrams else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1"       : round(f1,        4),
    }


def rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-L (Longest Common Subsequence).
    """
    ref_tokens = simple_tokenize(reference)
    hyp_tokens = simple_tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # LCS dynamic programming
    m, n_    = len(ref_tokens), len(hyp_tokens)
    dp       = [[0] * (n_ + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n_ + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len   = dp[m][n_]
    precision = lcs_len / n_ if n_ > 0 else 0.0
    recall    = lcs_len / m  if m > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1"       : round(f1,        4),
    }


def compute_all_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L in one call."""
    r1 = rouge_n(reference, hypothesis, n=1)
    r2 = rouge_n(reference, hypothesis, n=2)
    rl = rouge_l(reference, hypothesis)
    return {
        "rouge_1_f1": r1["f1"],
        "rouge_1_p" : r1["precision"],
        "rouge_1_r" : r1["recall"],
        "rouge_2_f1": r2["f1"],
        "rouge_2_p" : r2["precision"],
        "rouge_2_r" : r2["recall"],
        "rouge_l_f1": rl["f1"],
        "rouge_l_p" : rl["precision"],
        "rouge_l_r" : rl["recall"],
    }


# ==============================================================================
# QUALITATIVE METRICS
# ==============================================================================

def response_quality_metrics(response: str) -> Dict:
    """
    Compute simple response quality metrics without a reference.
    Useful for evaluating open-ended chatbot responses.
    """
    words     = response.split()
    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Avg words per sentence (good responses: 15-25 words)
    avg_wps = len(words) / len(sentences) if sentences else 0

    # Lexical diversity (unique words / total words)
    lexical_diversity = len(set(w.lower() for w in words)) / len(words) if words else 0

    # Check for educational keywords
    edu_keywords = [
        "learning", "student", "teacher", "education", "teaching", "classroom",
        "curriculum", "assessment", "skill", "knowledge", "understanding",
        "research", "study", "academic", "develop", "strategy", "approach"
    ]
    resp_lower = response.lower()
    keyword_count = sum(1 for kw in edu_keywords if kw in resp_lower)

    return {
        "word_count"        : len(words),
        "sentence_count"    : len(sentences),
        "avg_words_per_sent": round(avg_wps, 1),
        "lexical_diversity" : round(lexical_diversity, 3),
        "edu_keyword_count" : keyword_count,
        "has_numbered_list" : bool(re.search(r"\d\)", response) or re.search(r"\d\.", response)),
    }


# ==============================================================================
# FULL EVALUATION PIPELINE
# ==============================================================================

def evaluate_bot(
    bot,
    test_data_path: str,
    n_samples: int = 50,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate EduBot on the test set using BLEU, ROUGE, and quality metrics.

    Parameters
    ----------
    bot            : EduBot instance
    test_data_path : path to test JSON file
    n_samples      : number of test examples to evaluate
    verbose        : print progress

    Returns
    -------
    DataFrame with per-example metrics
    """
    logger.info("Loading test data from %s …", test_data_path)
    test_df = pd.read_json(test_data_path, orient="records")
    test_df = test_df.sample(min(n_samples, len(test_df)), random_state=42)

    results = []

    logger.info("Evaluating %d examples …", len(test_df))
    for i, (_, row) in enumerate(test_df.iterrows()):
        instruction = row["instruction"]
        reference   = row["response"]

        # Generate model response (fresh context each time)
        bot.reset_history()
        hypothesis = bot.chat(instruction, use_history=False)

        # Compute metrics
        bleu_scores  = bleu_score(reference, hypothesis)
        rouge_scores = compute_all_rouge(reference, hypothesis)
        quality      = response_quality_metrics(hypothesis)

        record = {
            "id"           : i,
            "instruction"  : instruction,
            "reference"    : reference[:200],
            "hypothesis"   : hypothesis[:200],
            **bleu_scores,
            **rouge_scores,
            **quality,
        }
        results.append(record)

        if verbose and (i + 1) % 10 == 0:
            logger.info("  Evaluated %d/%d examples", i + 1, len(test_df))

    eval_df = pd.DataFrame(results)
    return eval_df


def print_evaluation_summary(eval_df: pd.DataFrame) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 60)
    print("  EduBot Evaluation Results")
    print("=" * 60)

    metric_groups = {
        "BLEU Scores": ["bleu_1", "bleu_2", "bleu_4_combined"],
        "ROUGE Scores": ["rouge_1_f1", "rouge_2_f1", "rouge_l_f1"],
        "Response Quality": ["word_count", "avg_words_per_sent", "lexical_diversity", "edu_keyword_count"],
    }

    for group, metrics in metric_groups.items():
        print(f"\n  {group}:")
        for metric in metrics:
            if metric in eval_df.columns:
                mean_val = eval_df[metric].mean()
                std_val  = eval_df[metric].std()
                print(f"    {metric:30s}: {mean_val:.4f} ± {std_val:.4f}")

    print("\n  Top 3 Best Responses (by ROUGE-1 F1):")
    top3 = eval_df.nlargest(3, "rouge_1_f1")
    for _, row in top3.iterrows():
        print(f"\n    Q: {row['instruction'][:80]}…")
        print(f"    A: {row['hypothesis'][:100]}…")
        print(f"    ROUGE-1 F1: {row['rouge_1_f1']:.4f}")


def save_evaluation_results(eval_df: pd.DataFrame, output_dir: str = None) -> str:
    """Save evaluation results to CSV."""
    if output_dir is None:
        output_dir = str(ROOT_DIR / "models")
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "evaluation_results.csv")
    eval_df.to_csv(path, index=False)
    logger.info("Evaluation results saved → %s", path)
    return path


# ==============================================================================
# HUMAN EVALUATION RUBRIC (printed template)
# ==============================================================================

HUMAN_EVAL_RUBRIC = """
╔══════════════════════════════════════════════════════════╗
║       EduBot Human Evaluation Rubric                    ║
║       Industry: Education and Training                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║ Rate each response on a 1-5 scale:                       ║
║                                                          ║
║ 1. RELEVANCE (1-5)                                       ║
║    Does the response directly address the question?      ║
║    1 = Completely off-topic                             ║
║    5 = Perfectly on-topic                               ║
║                                                          ║
║ 2. ACCURACY (1-5)                                        ║
║    Is the educational information factually correct?     ║
║    1 = Significant errors                               ║
║    5 = Completely accurate                              ║
║                                                          ║
║ 3. COHERENCE (1-5)                                       ║
║    Is the response well-structured and easy to follow?   ║
║    1 = Incoherent or confusing                          ║
║    5 = Perfectly clear and well-organized               ║
║                                                          ║
║ 4. COMPLETENESS (1-5)                                    ║
║    Does the response fully answer the question?          ║
║    1 = Very incomplete                                  ║
║    5 = Comprehensive and thorough                       ║
║                                                          ║
║ 5. EDUCATIONAL VALUE (1-5)                               ║
║    Would this response be helpful to an educator/student?║
║    1 = No educational value                             ║
║    5 = Highly valuable and actionable                   ║
║                                                          ║
║ OVERALL SCORE = Average of 5 dimensions                  ║
╚══════════════════════════════════════════════════════════╝
"""


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT_DIR / "src"))

    print("=" * 60)
    print("  EduBot Evaluation Module")
    print("=" * 60)

    # ── Unit test: BLEU ────────────────────────────────────────────────────────
    ref = "Bloom's Taxonomy classifies educational objectives into six hierarchical levels."
    hyp = "Bloom's Taxonomy organizes learning objectives into six levels from remember to create."

    bleu = bleu_score(ref, hyp)
    rouge = compute_all_rouge(ref, hyp)

    print(f"\nSample BLEU scores : {bleu}")
    print(f"Sample ROUGE scores: {rouge}")

    # ── Run full evaluation (requires chatbot and test data) ──────────────────
    test_path = str(ROOT_DIR / "data" / "processed" / "education_test.json")

    if not Path(test_path).exists():
        print("\nTest data not found. Running preprocessing first …")
        from data_collection import collect_all_data, save_raw_data
        from preprocessing   import run_full_preprocessing
        df = collect_all_data(scrape_web=False)
        raw = save_raw_data(df)
        run_full_preprocessing(raw)

    from chatbot import EduBot
    bot = EduBot()  # Fallback mode

    eval_df = evaluate_bot(bot, test_path, n_samples=20)
    print_evaluation_summary(eval_df)
    save_evaluation_results(eval_df)

    print(HUMAN_EVAL_RUBRIC)
    print("\n✅ Evaluation complete!")
