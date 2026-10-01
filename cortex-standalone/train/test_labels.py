"""
train/test_labels.py — gold BIO construction from char spans (CONTRACTS.md §2.7–2.8, §3.5).

Run:  python -m unittest discover -s train -p 'test_*.py'
"""

from __future__ import annotations

import unittest

from labels import (
    O_LABEL,
    bio_ids_for_example,
    build_slot_labels,
    collapse_bio_to_name,
    label_to_id_map,
    word_split_with_offsets,
)
from tokenizer import word_split


class SlotLabelDerivation(unittest.TestCase):
    def test_build_slot_labels(self) -> None:
        self.assertEqual(build_slot_labels([]), ["O"])
        self.assertEqual(build_slot_labels(["region"]), ["O", "B-region", "I-region"])
        self.assertEqual(
            build_slot_labels(["zone", "region"]), ["O", "B-zone", "I-zone", "B-region", "I-region"]
        )

    def test_label_to_id_map_follows_given_order(self) -> None:
        self.assertEqual(label_to_id_map(["O", "B-x", "I-x"]), {"O": 0, "B-x": 1, "I-x": 2})
        self.assertEqual(label_to_id_map(["O", "B-y", "I-y", "B-x", "I-x"])["B-x"], 3)

    def test_collapse(self) -> None:
        self.assertEqual(collapse_bio_to_name("O"), "O")
        self.assertEqual(collapse_bio_to_name("B-region"), "region")
        self.assertEqual(collapse_bio_to_name("I-region"), "region")
        self.assertEqual(collapse_bio_to_name("B-a-b"), "a-b")


class Offsets(unittest.TestCase):
    def test_offsets_are_half_open_and_match_word_split(self) -> None:
        text = "Show me the North campus reports!"
        toks = word_split_with_offsets(text)
        self.assertEqual([t[0] for t in toks], ["Show", "me", "the", "North", "campus", "reports"])
        for word, s, e in toks:
            self.assertEqual(text[s:e], word)
        # same token COUNT as the lowercased tokenizer split (§3.5: casing never moves boundaries)
        self.assertEqual(len(toks), len(word_split(text.lower())))

    def test_offsets_unicode(self) -> None:
        text = "¿dónde están los informes?"
        toks = word_split_with_offsets(text)
        self.assertEqual([t[0] for t in toks], ["dónde", "están", "los", "informes"])
        self.assertEqual(toks[0][1:], (1, 6))
        self.assertEqual(toks[-1][2], len(text) - 1)

    def test_empty(self) -> None:
        self.assertEqual(word_split_with_offsets(""), [])
        self.assertEqual(word_split_with_offsets(" ... "), [])


class BioIds(unittest.TestCase):
    LABELS = ["O", "B-region", "I-region"]

    def test_single_multi_token_span(self) -> None:
        utt = "show me the North campus reports"
        start = utt.index("North campus")
        slots = [{"name": "region", "value": "North campus", "start": start, "end": start + len("North campus")}]
        ids = bio_ids_for_example(utt, slots, 8, self.LABELS)
        # show me the North campus reports -> O O O B I O, then padded with O to 8
        self.assertEqual(ids, [0, 0, 0, 1, 2, 0, 0, 0])

    def test_two_spans_two_slots(self) -> None:
        labels = ["O", "B-zone", "I-zone", "B-region", "I-region"]
        utt = "reports for East wing in zone 7"
        s1 = utt.index("East wing")
        s2 = utt.index("zone 7")
        slots = [
            {"name": "region", "value": "East wing", "start": s1, "end": s1 + 9},
            {"name": "zone", "value": "zone 7", "start": s2, "end": s2 + 6},
        ]
        ids = bio_ids_for_example(utt, slots, 7, labels)
        # reports for East wing in zone 7 -> O O B-region I-region O B-zone I-zone
        self.assertEqual(ids, [0, 0, 3, 4, 0, 1, 2])

    def test_truncation_at_max_len(self) -> None:
        utt = "a b c North campus"
        s = utt.index("North")
        slots = [{"name": "region", "value": "North campus", "start": s, "end": len(utt)}]
        self.assertEqual(bio_ids_for_example(utt, slots, 4, self.LABELS), [0, 0, 0, 1])

    def test_no_slots_all_o(self) -> None:
        self.assertEqual(bio_ids_for_example("hello there", [], 3, self.LABELS), [0, 0, 0])
        self.assertEqual(bio_ids_for_example("hello there", [], 3, ["O"]), [0, 0, 0])

    def test_partial_overlap_tags_token(self) -> None:
        # a span covering only part of a token still tags that token (overlap rule)
        utt = "northcampus reports"
        slots = [{"name": "region", "value": "north", "start": 0, "end": 5}]
        self.assertEqual(bio_ids_for_example(utt, slots, 2, self.LABELS), [1, 0])

    def test_unknown_slot_name_raises(self) -> None:
        utt = "reports for East wing"
        slots = [{"name": "nope", "value": "East wing", "start": 12, "end": 21}]
        with self.assertRaises(ValueError):
            bio_ids_for_example(utt, slots, 4, self.LABELS)

    def test_ids_follow_label_order_not_a_hardcoded_table(self) -> None:
        utt = "reports for East wing"
        slots = [{"name": "region", "value": "East wing", "start": 12, "end": 21}]
        a = bio_ids_for_example(utt, slots, 4, ["O", "B-region", "I-region"])
        b = bio_ids_for_example(utt, slots, 4, ["O", "B-zone", "I-zone", "B-region", "I-region"])
        self.assertEqual(a, [0, 0, 1, 2])
        self.assertEqual(b, [0, 0, 3, 4])
        self.assertEqual(O_LABEL, "O")


if __name__ == "__main__":
    unittest.main()
