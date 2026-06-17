"""
src/preprocessing.py
====================
Data Preprocessing Module — EduBot LLM Project
Industry: Education and Training

Handles:
  1. Text cleaning (noise removal, normalization)
  2. Tokenization & length analysis
  3. Deduplication
  4. Instruction-Response format preparation for fine-tuning
  5. Train/validation split
  6. Data augmentation

Author: EduBot Project
"""

import os
import re
import json
import random
import logging
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR  = Path(__file__).parent.parent
RAW_DIR   = ROOT_DIR / "data" / "raw"
PROC_DIR  = ROOT_DIR / "data" / "processed"
AUG_DIR   = ROOT_DIR / "data" / "augmented"

for d in [PROC_DIR, AUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. TEXT CLEANING
# ==============================================================================

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII-compatible form."""
    return unicodedata.normalize("NFKC", text)


def remove_noise(text: str) -> str:
    """
    Remove common text noise:
    - Multiple whitespace → single space
    - Non-printable control characters
    - HTML tags
    - Excessive punctuation repetition
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove non-printable characters (keep standard punctuation)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Normalize multiple spaces/newlines to single
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def normalize_text(text: str) -> str:
    """
    Full text normalization pipeline:
    1. Unicode normalization
    2. Noise removal
    3. Punctuation cleanup
    """
    text = normalize_unicode(text)
    text = remove_noise(text)

    # Normalize quotation marks
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

    # Fix multiple punctuation
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    return text.strip()


def clean_record(record: Dict) -> Optional[Dict]:
    """
    Clean a single instruction-response pair.
    Returns None if the record fails quality checks.
    """
    instruction = normalize_text(str(record.get("instruction", "")))
    response    = normalize_text(str(record.get("response",    "")))

    # Quality gates
    if len(instruction) < 10:
        return None   # Too short to be meaningful
    if len(response) < 30:
        return None   # Response too brief
    if len(instruction) > 512:
        instruction = instruction[:512]  # Truncate overly long instructions
    if len(response) > 2048:
        response = response[:2048]       # Truncate very long responses

    return {
        "instruction": instruction,
        "response"   : response,
        "source"     : record.get("source", "unknown"),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full cleaning pipeline to the entire dataset.

    Steps:
    1. Clean each record
    2. Drop records that failed quality checks
    3. Remove duplicates
    4. Reset index
    """
    logger.info("Cleaning dataset (%d records) …", len(df))

    cleaned = []
    skipped = 0

    for _, row in df.iterrows():
        record = clean_record(row.to_dict())
        if record:
            cleaned.append(record)
        else:
            skipped += 1

    df_clean = pd.DataFrame(cleaned)

    # Deduplicate on normalized instruction
    before = len(df_clean)
    df_clean["inst_lower"] = df_clean["instruction"].str.lower().str.strip()
    df_clean = df_clean.drop_duplicates(subset=["inst_lower"]).drop(columns=["inst_lower"])
    duplicates_removed = before - len(df_clean)

    df_clean = df_clean.reset_index(drop=True)
    df_clean["id"] = df_clean.index

    logger.info("  Skipped (quality)  : %d", skipped)
    logger.info("  Duplicates removed : %d", duplicates_removed)
    logger.info("  Clean dataset size : %d", len(df_clean))

    return df_clean


# ==============================================================================
# 2. FORMAT CONVERSION FOR FINE-TUNING
# ==============================================================================

# ── Prompt templates used for formatting ──────────────────────────────────────
PROMPT_TEMPLATE_ALPACA = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

PROMPT_TEMPLATE_CHATML = """<|system|>
You are EduBot, an intelligent educational assistant specializing in the Education and Training industry. You provide accurate, clear, and helpful answers about pedagogy, curriculum, teaching strategies, e-learning, student development, and all aspects of education.
<|user|>
{instruction}
<|assistant|>
{response}"""

PROMPT_TEMPLATE_INFERENCE = """<|system|>
You are EduBot, an intelligent educational assistant specializing in the Education and Training industry. You provide accurate, clear, and helpful answers about pedagogy, curriculum, teaching strategies, e-learning, student development, and all aspects of education.
<|user|>
{instruction}
<|assistant|>"""


def format_record_alpaca(record: Dict) -> str:
    """Format a record using Alpaca-style prompt template."""
    return PROMPT_TEMPLATE_ALPACA.format(
        instruction=record["instruction"],
        response   =record["response"],
    )


def format_record_chatml(record: Dict, include_response: bool = True) -> str:
    """Format a record using ChatML-style prompt template."""
    if include_response:
        return PROMPT_TEMPLATE_CHATML.format(
            instruction=record["instruction"],
            response   =record["response"],
        )
    else:
        return PROMPT_TEMPLATE_INFERENCE.format(
            instruction=record["instruction"]
        )


def prepare_training_format(df: pd.DataFrame, template: str = "chatml") -> pd.DataFrame:
    """
    Convert cleaned dataset into fine-tuning format.

    Adds columns:
    - text        : full formatted prompt + response (for training)
    - prompt_only : formatted prompt without response (for inference)
    - word_count  : word count of full text
    """
    logger.info("Preparing %s-format training data …", template)
    df = df.copy()

    if template == "alpaca":
        df["text"]        = df.apply(lambda r: format_record_alpaca(r.to_dict()), axis=1)
        df["prompt_only"] = df.apply(lambda r: PROMPT_TEMPLATE_ALPACA.format(
            instruction=r["instruction"], response=""), axis=1)
    else:  # chatml (default)
        df["text"]        = df.apply(lambda r: format_record_chatml(r.to_dict(), include_response=True),  axis=1)
        df["prompt_only"] = df.apply(lambda r: format_record_chatml(r.to_dict(), include_response=False), axis=1)

    df["word_count"]  = df["text"].str.split().str.len()
    df["char_count"]  = df["text"].str.len()
    df["token_approx"] = (df["word_count"] * 1.3).astype(int)  # rough token estimate

    logger.info("  Avg words per example : %.0f", df["word_count"].mean())
    logger.info("  Max words per example : %d",   df["word_count"].max())
    logger.info("  Examples > 512 tokens : %d",   (df["token_approx"] > 512).sum())

    return df


# ==============================================================================
# 3. TRAIN / VALIDATION SPLIT
# ==============================================================================

def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.05,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    Stratify by source if possible.

    Returns
    -------
    train_df, val_df, test_df
    """
    from sklearn.model_selection import train_test_split

    n = len(df)
    n_val  = max(1, int(n * val_size))
    n_test = max(1, int(n * test_size))

    df_temp, test_df  = train_test_split(df, test_size=n_test,  random_state=random_state)
    train_df, val_df  = train_test_split(df_temp, test_size=n_val, random_state=random_state)

    logger.info("Split → Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


# ==============================================================================
# 4. DATA AUGMENTATION
# ==============================================================================

QUESTION_REPHRASINGS = [
    "Can you explain {topic}?",
    "Tell me about {topic}.",
    "What should I know about {topic}?",
    "Could you describe {topic}?",
    "I'd like to understand {topic}. Can you help?",
    "What are the key points about {topic}?",
    "Give me an overview of {topic}.",
    "How would you describe {topic}?",
]


def augment_instructions(df: pd.DataFrame, augment_ratio: float = 0.3) -> pd.DataFrame:
    """
    Augment the dataset by creating paraphrased versions of existing instructions.

    For questions starting with "What is X?", we generate alternative phrasings
    using templates to increase data diversity.

    Parameters
    ----------
    df            : clean formatted DataFrame
    augment_ratio : fraction of dataset to augment (0.3 = 30% extra examples)
    """
    augmented = []
    n_to_augment = int(len(df) * augment_ratio)

    sample = df.sample(min(n_to_augment, len(df)), random_state=42)

    for _, row in sample.iterrows():
        original = row["instruction"]

        # Extract topic from "What is X?" pattern
        match = re.match(r"(?:What is|Explain|How does|What are)[^?]*\b([\w\s]+)\??$",
                         original, re.IGNORECASE)
        if match:
            topic   = match.group(1).strip().rstrip("?")
            template = random.choice(QUESTION_REPHRASINGS)
            new_inst = template.format(topic=topic)

            if new_inst.lower() != original.lower():
                new_row = row.to_dict()
                new_row["instruction"] = new_inst
                new_row["source"]      = "augmented"
                augmented.append(new_row)

    if augmented:
        aug_df = pd.DataFrame(augmented)
        result = pd.concat([df, aug_df], ignore_index=True)
        result = result.drop_duplicates(subset=["instruction"]).reset_index(drop=True)
        logger.info("Augmentation: added %d examples (total: %d)", len(augmented), len(result))
        return result

    return df


# ==============================================================================
# 5. DATASET STATISTICS
# ==============================================================================

def print_dataset_stats(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print detailed statistics about the dataset."""
    print(f"\n{'='*50}")
    print(f"  {name} Statistics")
    print(f"{'='*50}")
    print(f"  Total examples    : {len(df):,}")

    if "source" in df.columns:
        print(f"  By source         :")
        for src, cnt in df["source"].value_counts().items():
            print(f"    {src:25s}: {cnt}")

    if "word_count" in df.columns:
        print(f"\n  Word counts (text):")
        print(f"    Mean : {df['word_count'].mean():.0f}")
        print(f"    Min  : {df['word_count'].min()}")
        print(f"    Max  : {df['word_count'].max()}")
        print(f"    Std  : {df['word_count'].std():.0f}")

    print(f"\n  Instruction length (chars):")
    print(f"    Mean : {df['instruction'].str.len().mean():.0f}")
    print(f"    Max  : {df['instruction'].str.len().max()}")

    print(f"\n  Response length (chars):")
    print(f"    Mean : {df['response'].str.len().mean():.0f}")
    print(f"    Max  : {df['response'].str.len().max()}")


# ==============================================================================
# 6. SAVE / LOAD FUNCTIONS
# ==============================================================================

def save_processed(df: pd.DataFrame, filename: str) -> str:
    """Save processed DataFrame to JSON."""
    path = str(PROC_DIR / filename)
    df.to_json(path, orient="records", indent=2)
    logger.info("Saved → %s (%d records)", path, len(df))
    return path


def save_augmented(df: pd.DataFrame, filename: str) -> str:
    """Save augmented DataFrame to JSON."""
    path = str(AUG_DIR / filename)
    df.to_json(path, orient="records", indent=2)
    logger.info("Saved → %s (%d records)", path, len(df))
    return path


def load_processed(filename: str) -> pd.DataFrame:
    """Load a processed dataset from disk."""
    path = PROC_DIR / filename
    df = pd.read_json(str(path), orient="records")
    logger.info("Loaded ← %s (%d records)", path, len(df))
    return df


# ==============================================================================
# FULL PIPELINE FUNCTION
# ==============================================================================

def run_full_preprocessing(raw_path: str, template: str = "chatml") -> Dict[str, pd.DataFrame]:
    """
    Execute the complete preprocessing pipeline:
    raw JSON → clean → format → augment → split → save

    Returns
    -------
    dict with keys: full, train, val, test
    """
    logger.info("=" * 60)
    logger.info("  EduBot Preprocessing Pipeline")
    logger.info("=" * 60)

    # Load raw data
    df_raw = pd.read_json(raw_path, orient="records")
    logger.info("Loaded raw data: %d records", len(df_raw))

    # Step 1: Clean
    df_clean = clean_dataset(df_raw)

    # Step 2: Format
    df_formatted = prepare_training_format(df_clean, template=template)

    # Step 3: Augment
    df_augmented = augment_instructions(df_formatted, augment_ratio=0.4)

    # Step 4: Reformat augmented entries
    df_final = prepare_training_format(
        df_augmented[["instruction", "response", "source"]].copy(),
        template=template
    )

    # Step 5: Split
    train_df, val_df, test_df = split_dataset(df_final)

    # Step 6: Save
    save_processed(df_final,  "education_full.json")
    save_processed(train_df,  "education_train.json")
    save_processed(val_df,    "education_val.json")
    save_processed(test_df,   "education_test.json")

    # Step 7: Stats
    print_dataset_stats(df_final,  "Full Dataset")
    print_dataset_stats(train_df,  "Training Set")

    # Save as HuggingFace-compatible format (text only, one per line in jsonl)
    jsonl_path = str(PROC_DIR / "education_train.jsonl")
    with open(jsonl_path, "w") as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps({"text": row["text"]}) + "\n")
    logger.info("JSONL saved → %s", jsonl_path)

    return {"full": df_final, "train": train_df, "val": val_df, "test": test_df}


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys

    raw_path = str(RAW_DIR / "education_raw.json")

    if not Path(raw_path).exists():
        logger.info("Raw data not found. Running data collection first …")
        sys.path.insert(0, str(Path(__file__).parent))
        from data_collection import collect_all_data, save_raw_data
        df_raw = collect_all_data(scrape_web=False)
        save_raw_data(df_raw)

    splits = run_full_preprocessing(raw_path, template="chatml")
    print(f"\n✅ Preprocessing complete!")
    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Val  : {len(splits['val'])} examples")
    print(f"  Test : {len(splits['test'])} examples")

    # Print a sample training example
    print(f"\n{'='*60}")
    print("SAMPLE TRAINING TEXT:")
    print(f"{'='*60}")
    print(splits["train"]["text"].iloc[0])
