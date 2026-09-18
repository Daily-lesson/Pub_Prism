"""
train/test_tokenizer.py — the Python half of the tokenizer parity pin (CONTRACTS.md §3).

Run:  python -m unittest discover -s train -p 'test_*.py'

Two layers, so the test is never vacuous:
  1. The shared fixture `tests/fixtures/tokenizer.parity.json` (written by the core) — if
     present, every case's `words` and `ids` must be reproduced exactly.
  2. Inline cases against a hand-built vocab, with expected ids typed by hand (never
     computed by the function under test): unicode letters, digits, punctuation, emoji,
     combining marks, truncation, OOV, empty/whitespace-only input.
"""

from __future__ import annotations

import json
import os
import unittest

from tokenizer import (
    PAD_ID,
    UNK_ID,
    TokenizerConfig,
    build_tokenizer_config,
    build_vocab,
    is_word_char,
    tokenize,
    word_split,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_PATH = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "tokenizer.parity.json")

# A hand-built vocab. Ids typed by hand below; nothing in this table is derived at runtime.
INLINE_VOCAB = {
    "how": 2, "do": 3, "i": 4, "find": 5, "the": 6, "reports": 7,
    "show": 8, "me": 9, "north": 10, "campus": 11, "café": 12, "日本語": 13, "42": 14,
}
INLINE_CFG = TokenizerConfig(version="1", lower=True, max_len=6, pad_id=PAD_ID, unk_id=UNK_ID, vocab=INLINE_VOCAB)

# (text, expected words after lowercasing, expected ids at maxLen=6)
INLINE_CASES = [
    ("how do i find the reports", ["how", "do", "i", "find", "the", "reports"], [2, 3, 4, 5, 6, 7]),
    ("", [], [0, 0, 0, 0, 0, 0]),
    ("   \t  ", [], [0, 0, 0, 0, 0, 0]),
    ("HOW DO I", ["how", "do", "i"], [2, 3, 4, 0, 0, 0]),
    ("show me... the REPORTS?!", ["show", "me", "the", "reports"], [8, 9, 6, 7, 0, 0]),
    ("north-campus", ["north", "campus"], [10, 11, 0, 0, 0, 0]),
    ("Café", ["café"], [12, 0, 0, 0, 0, 0]),
    ("日本語 42", ["日本語", "42"], [13, 14, 0, 0, 0, 0]),
    ("hello 👋 world", ["hello", "world"], [1, 1, 0, 0, 0, 0]),
    # 8 words: `words` is the full split, `ids` is truncated at maxLen=6
    ("how do i find the reports show me", ["how", "do", "i", "find", "the", "reports", "show", "me"], [2, 3, 4, 5, 6, 7]),
    ("reports reports reports", ["reports", "reports", "reports"], [7, 7, 7, 0, 0, 0]),
    ("a1b2", ["a1b2"], [1, 0, 0, 0, 0, 0]),
    # underscore is Pc (connector punctuation), not L/N — a separator
    ("the_reports", ["the", "reports"], [6, 7, 0, 0, 0, 0]),
    ("I  find\nthe   reports", ["i", "find", "the", "reports"], [4, 5, 6, 7, 0, 0]),
    ("42,000 reports", ["42", "000", "reports"], [14, 1, 7, 0, 0, 0]),
    ("ünïcödé", ["ünïcödé"], [1, 0, 0, 0, 0, 0]),
    # lower("İ") is "i" + U+0307 COMBINING DOT ABOVE (category Mn, a separator) — the
    # same thing JS toLowerCase() produces, so both copies split identically here.
    ("İstanbul", ["i", "stanbul"], [4, 1, 0, 0, 0, 0]),
    ("ß", ["ß"], [1, 0, 0, 0, 0, 0]),
    ("(the) [reports] {42}", ["the", "reports", "42"], [6, 7, 14, 0, 0, 0]),
]


class InlineTokenizerCases(unittest.TestCase):
    def test_inline_case_count_is_not_vacuous(self) -> None:
        self.assertGreaterEqual(len(INLINE_CASES), 15)

    def test_inline_word_split_and_ids(self) -> None:
        for text, words, ids in INLINE_CASES:
            with self.subTest(text=text):
                self.assertEqual(word_split(text.lower()), words)
                self.assertEqual(tokenize(text, INLINE_CFG), ids)
                self.assertEqual(len(tokenize(text, INLINE_CFG)), INLINE_CFG.max_len)

    def test_is_word_char_categories(self) -> None:
        for ch in "aZ9é日٣":  # letters, digits, Arabic-Indic digit (Nd)
            self.assertTrue(is_word_char(ch), ch)
        for ch in " -_.,!?👋\n\ṫ":  # space, punctuation, emoji, whitespace, combining mark
            self.assertFalse(is_word_char(ch), repr(ch))

    def test_lower_false_keeps_case(self) -> None:
        cfg = TokenizerConfig(version="1", lower=False, max_len=4, pad_id=0, unk_id=1, vocab={"How": 2, "how": 3})
        self.assertEqual(tokenize("How how", cfg), [2, 3, 0, 0])

    def test_build_vocab_ordering(self) -> None:
        # frequency descending, then alphabetical; ids start at 2
        vocab = build_vocab(["b a a", "c b a", "C"])
        self.assertEqual(vocab, {"a": 2, "b": 3, "c": 4})
        cfg = build_tokenizer_config(["b a a", "c b a"], max_len=4)
        self.assertEqual(cfg.pad_id, 0)
        self.assertEqual(cfg.unk_id, 1)
        self.assertEqual(cfg.max_len, 4)
        self.assertEqual(tokenize("zzz a", cfg), [1, 2, 0, 0])

    def test_json_round_trip(self) -> None:
        d = INLINE_CFG.to_json_dict()
        self.assertEqual(
            list(d.keys()), ["version", "lower", "maxLen", "padId", "unkId", "padToken", "unkToken", "vocab"]
        )
        self.assertEqual(TokenizerConfig.from_json_dict(json.loads(json.dumps(d))), INLINE_CFG)


class SharedFixtureParity(unittest.TestCase):
    """Pins this copy against `tests/fixtures/tokenizer.parity.json` when the core has
    written it. Skips (visibly) when the fixture is absent — the inline cases above still
    run, so absence never makes this file vacuous."""

    def test_shared_fixture(self) -> None:
        if not os.path.isfile(FIXTURE_PATH):
            self.skipTest(f"shared fixture not present at {FIXTURE_PATH}")
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            fixture = json.load(f)
        cfg = TokenizerConfig.from_json_dict(fixture["config"])
        cases = fixture["cases"]
        self.assertGreater(len(cases), 0, "fixture has no cases")
        for case in cases:
            with self.subTest(text=case["text"]):
                text = case["text"]
                lowered = text.lower() if cfg.lower else text
                # `words` boundaries must match; compare case-insensitively so a fixture
                # that records original-cased words still pins the same split.
                self.assertEqual([w.lower() for w in word_split(lowered)], [w.lower() for w in case["words"]])
                self.assertEqual(tokenize(text, cfg), case["ids"])


if __name__ == "__main__":
    unittest.main()
