"""
train/gen_tokenizer_fixture.py

Regenerates the tokenizer PARITY FIXTURE (docs/CONTRACTS.md §3): `(text -> words, ids)`
cases produced by THIS (Python) tokenizer against a given `tokenizer.json`. The TypeScript
core and the browser widget assert they reproduce every case exactly; `train/test_tokenizer.py`
asserts the same for this copy. That shared pin is the whole point of building three
tokenizers "the same way" instead of one calling the others.

Output shape:
    { "config": { <the tokenizer.json object> },
      "cases": [ { "text": "...", "words": [...], "ids": [...] } ] }

`words` are the post-lowercase word-split tokens (the exact sequence mapped to ids —
lowercasing happens before the split, §3.1).

Usage:
    python train/gen_tokenizer_fixture.py --tokenizer <x.tokenizer.json> --out <fixture.json> [--texts-file <txt>]

Without `--texts-file` a built-in generic case list is used: in-vocab-looking text,
out-of-vocabulary words, punctuation noise, casing variants, digits/hyphens/slashes, empty
and whitespace-only strings (all-pad), accented Latin, non-Latin scripts, emoji, and a
sequence long enough to exercise truncation at `maxLen`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import TokenizerConfig, load_tokenizer_config, tokenize, word_split  # noqa: E402

DEFAULT_TEXTS: List[str] = [
    "how do i find the reports",
    "show me the north campus numbers",
    "where can i book a room for this week",
    "please recalibrate the flux capacitor before liftoff",
    "why is the printer on floor sixteen making a grinding noise",
    "how do i, uh... open the reports?!",
    "so, could you help me set this up please -- thanks!!",
    "HOW DO I FIND THE REPORTS",
    "Where Is The Main Building Report",
    "book 3 rooms on floor 42",
    "3D floor-plan / survey report #7",
    "",
    "   \t  ",
    "¿cómo encuentro los informes?",
    "naïve café résumé over the déjà vu façade",
    "日本語のテキストと数字123",
    "hello 👋 world 🚀 how are you",
    "under_score and mid-dash tokens",
    "please could you kindly help me understand how do i go about finding "
    "every single one of the reports and charts and figures and summaries "
    "for all of the regions at the same time on the same page today",
]


def build_cases(texts: List[str], cfg: TokenizerConfig) -> List[dict]:
    cases = []
    for text in texts:
        lowered = text.lower() if cfg.lower else text
        cases.append({"text": text, "words": word_split(lowered), "ids": tokenize(text, cfg)})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the tokenizer parity fixture.")
    ap.add_argument("--tokenizer", required=True, help="path to a <slug>-<version>.tokenizer.json")
    ap.add_argument("--out", required=True, help="output fixture path (e.g. tests/fixtures/tokenizer.parity.json)")
    ap.add_argument("--texts-file", default=None, help="one text per line; default: the built-in generic list")
    args = ap.parse_args()

    cfg = load_tokenizer_config(args.tokenizer)
    if args.texts_file:
        with open(args.texts_file, "r", encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f]
    else:
        texts = DEFAULT_TEXTS

    fixture = {"config": cfg.to_json_dict(), "cases": build_cases(texts, cfg)}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[cortex-train] wrote {len(fixture['cases'])} tokenizer parity cases -> {args.out}")


if __name__ == "__main__":
    main()
