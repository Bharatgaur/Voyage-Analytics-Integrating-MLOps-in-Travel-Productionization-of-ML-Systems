"""
tests/test_edubot.py
====================
Pytest test suite for EduBot LLM Project.
Run: pytest tests/ -v

Tests cover:
  - Data collection
  - Preprocessing pipeline
  - BLEU / ROUGE evaluation metrics
  - Chatbot engine
  - Prompt engineering
  - Rule-based fallback
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_qa():
    return [
        {"instruction": "What is Bloom's Taxonomy?",
         "response": "Bloom's Taxonomy classifies educational objectives into six hierarchical levels."},
        {"instruction": "What is constructivism?",
         "response": "Constructivism holds that learners actively build knowledge through experience."},
        {"instruction": "How does project-based learning work?",
         "response": "PBL is an instructional method where students learn by working on real-world projects."},
    ]


@pytest.fixture(scope="module")
def sample_df(sample_qa):
    return pd.DataFrame(sample_qa)


@pytest.fixture(scope="module")
def bot():
    from chatbot import EduBot
    return EduBot()  # fallback mode, no model required


# ── Data Collection Tests ─────────────────────────────────────────────────────

class TestDataCollection:
    def test_collect_returns_dataframe(self):
        from data_collection import collect_all_data
        df = collect_all_data(scrape_web=False)
        assert isinstance(df, pd.DataFrame)

    def test_minimum_examples_collected(self):
        from data_collection import collect_all_data
        df = collect_all_data(scrape_web=False)
        assert len(df) >= 30, "Should collect at least 30 Q&A pairs"

    def test_required_columns_present(self):
        from data_collection import collect_all_data
        df = collect_all_data(scrape_web=False)
        assert "instruction" in df.columns
        assert "response" in df.columns
        assert "source" in df.columns

    def test_no_empty_instructions(self):
        from data_collection import collect_all_data
        df = collect_all_data(scrape_web=False)
        assert df["instruction"].str.len().min() > 5

    def test_no_empty_responses(self):
        from data_collection import collect_all_data
        df = collect_all_data(scrape_web=False)
        assert df["response"].str.len().min() > 20


# ── Preprocessing Tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_normalize_text_removes_extra_spaces(self):
        from preprocessing import normalize_text
        assert normalize_text("Hello   world") == "Hello world"

    def test_normalize_text_strips_html(self):
        from preprocessing import normalize_text
        result = normalize_text("<p>Hello <b>world</b></p>")
        assert "<" not in result and ">" not in result

    def test_normalize_text_strips_urls(self):
        from preprocessing import normalize_text
        result = normalize_text("Visit https://example.com for more")
        assert "https" not in result

    def test_clean_record_accepts_valid(self):
        from preprocessing import clean_record
        rec = clean_record({
            "instruction": "What is education?",
            "response": "Education is the structured process of facilitating learning. " * 3,
            "source": "test"
        })
        assert rec is not None

    def test_clean_record_rejects_too_short(self):
        from preprocessing import clean_record
        rec = clean_record({
            "instruction": "Hi",
            "response": "OK",
            "source": "test"
        })
        assert rec is None

    def test_clean_dataset_removes_duplicates(self):
        from preprocessing import clean_dataset
        df = pd.DataFrame({
            "instruction": ["What is Bloom?", "What is Bloom?", "What is UDL?"],
            "response": ["Bloom response " * 5, "Bloom response " * 5, "UDL response " * 5],
            "source": ["test", "test", "test"]
        })
        cleaned = clean_dataset(df)
        assert len(cleaned) == 2

    def test_prepare_training_format_chatml(self):
        from preprocessing import prepare_training_format
        df = pd.DataFrame({
            "instruction": ["What is education?"],
            "response": ["Education is the structured learning process." * 5],
            "source": ["test"]
        })
        formatted = prepare_training_format(df, template="chatml")
        assert "<|system|>" in formatted["text"].iloc[0]
        assert "<|user|>" in formatted["text"].iloc[0]
        assert "<|assistant|>" in formatted["text"].iloc[0]

    def test_augmentation_adds_examples(self):
        from preprocessing import augment_instructions, prepare_training_format
        df = pd.DataFrame({
            "instruction": ["What is education?", "Explain pedagogy.", "What is STEM?"],
            "response": ["Response text. " * 10] * 3,
            "source": ["test"] * 3
        })
        formatted = prepare_training_format(df, template="chatml")
        augmented = augment_instructions(formatted, augment_ratio=1.0)
        assert len(augmented) >= len(formatted)


# ── BLEU / ROUGE Metric Tests ─────────────────────────────────────────────────

class TestMetrics:
    def test_bleu_perfect_match(self):
        from evaluation import bleu_score
        text = "The quick brown fox jumps over the lazy dog"
        scores = bleu_score(text, text)
        assert scores["bleu_1"] == pytest.approx(1.0)

    def test_bleu_empty_hypothesis(self):
        from evaluation import bleu_score
        scores = bleu_score("Reference text here", "")
        assert all(v == 0.0 for v in scores.values())

    def test_bleu_partial_overlap(self):
        from evaluation import bleu_score
        ref = "Bloom Taxonomy educational six levels objectives"
        hyp = "Bloom Taxonomy six cognitive levels"
        scores = bleu_score(ref, hyp)
        assert 0 < scores["bleu_1"] < 1.0

    def test_rouge1_perfect_match(self):
        from evaluation import rouge_n
        text = "educational objectives taxonomy six levels"
        r = rouge_n(text, text, n=1)
        assert r["f1"] == pytest.approx(1.0)

    def test_rouge1_no_overlap(self):
        from evaluation import rouge_n
        r = rouge_n("apple orange banana", "dog cat fish", n=1)
        assert r["f1"] == 0.0

    def test_rouge_l_longer_common_subsequence(self):
        from evaluation import rouge_l
        ref = "learning objectives educational taxonomy bloom"
        hyp = "bloom educational taxonomy six levels"
        r = rouge_l(ref, hyp)
        assert 0 < r["f1"] < 1.0

    def test_corpus_bleu_averages_correctly(self):
        from evaluation import corpus_bleu
        refs = ["hello world test", "education learning"]
        hyps = ["hello world test", "education learning"]  # perfect
        scores = corpus_bleu(refs, hyps)
        assert scores["bleu_1"] == pytest.approx(1.0)

    def test_response_quality_metrics_structure(self):
        from evaluation import response_quality_metrics
        q = response_quality_metrics("Education helps students develop critical thinking skills.")
        assert "word_count" in q
        assert "lexical_diversity" in q
        assert "edu_keyword_count" in q
        assert q["word_count"] > 0
        assert 0 < q["lexical_diversity"] <= 1.0


# ── Chatbot Engine Tests ──────────────────────────────────────────────────────

class TestChatbot:
    def test_bot_initializes_fallback(self, bot):
        assert bot.get_mode() == "fallback"

    def test_bot_responds_to_query(self, bot):
        resp = bot.chat("What is Bloom's Taxonomy?")
        assert isinstance(resp, str)
        assert len(resp) > 20

    def test_bot_handles_empty_query(self, bot):
        resp = bot.chat("")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_bot_history_tracking(self, bot):
        bot.reset_history()
        bot.chat("What is pedagogy?")
        bot.chat("Give me an example.")
        history = bot.get_history()
        assert len(history) == 4  # 2 user + 2 assistant

    def test_bot_history_reset(self, bot):
        bot.chat("Test question")
        bot.reset_history()
        assert len(bot.get_history()) == 0

    def test_bot_stats_structure(self, bot):
        stats = bot.get_stats()
        assert "mode" in stats
        assert "total_turns" in stats
        assert "user_messages" in stats

    def test_bloom_taxonomy_response(self, bot):
        bot.reset_history()
        resp = bot.chat("What is Bloom's Taxonomy?")
        assert "bloom" in resp.lower() or "taxonomy" in resp.lower() or "levels" in resp.lower()

    def test_pbl_response(self, bot):
        bot.reset_history()
        resp = bot.chat("What is project-based learning?")
        assert any(kw in resp.lower() for kw in ["project", "learning", "students"])

    def test_bot_repr(self, bot):
        repr_str = repr(bot)
        assert "EduBot" in repr_str


# ── Prompt Engineering Tests ──────────────────────────────────────────────────

class TestPromptEngineering:
    def test_system_prompt_contains_edubot(self):
        from chatbot import SYSTEM_PROMPT
        assert "EduBot" in SYSTEM_PROMPT
        assert "Education" in SYSTEM_PROMPT

    def test_few_shot_prompt_structure(self):
        from chatbot import build_few_shot_prompt
        prompt = build_few_shot_prompt("What is Bloom's Taxonomy?")
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "EduBot" in prompt

    def test_few_shot_examples_included(self):
        from chatbot import build_few_shot_prompt, FEW_SHOT_EXAMPLES
        prompt = build_few_shot_prompt("test question", conversation_history=None)
        # Should contain at least one few-shot example
        assert FEW_SHOT_EXAMPLES[0]["user"] in prompt or "Bloom" in prompt

    def test_history_included_in_prompt(self):
        from chatbot import build_few_shot_prompt
        history = [
            {"role": "user",      "content": "What is constructivism?"},
            {"role": "assistant", "content": "Constructivism is a learning theory."},
        ]
        prompt = build_few_shot_prompt("Give me an example.", conversation_history=history)
        assert "constructivism" in prompt.lower()

    def test_qa_prompt_structure(self):
        from chatbot import build_qa_prompt
        prompt = build_qa_prompt("What is UDL?")
        assert "UDL" in prompt
        assert "<|assistant|>" in prompt

    def test_knowledge_base_coverage(self):
        from chatbot import KNOWLEDGE_BASE, get_rule_based_response
        # Test key education topics are covered
        key_topics = ["bloom", "gamification", "stem", "mooc", "flipped classroom"]
        for topic in key_topics:
            resp = get_rule_based_response(f"What is {topic}?")
            assert resp is not None, f"No fallback response for topic: {topic}"

    def test_clean_response_strips_tokens(self):
        from chatbot import clean_response
        raw = "<|assistant|>\nThis is the actual response.\n<|user|>\nNext question"
        cleaned = clean_response(raw)
        assert "<|" not in cleaned
        assert "This is the actual response" in cleaned

    def test_clean_response_handles_empty(self):
        from chatbot import clean_response
        result = clean_response("")
        assert isinstance(result, str)
        assert len(result) > 0
