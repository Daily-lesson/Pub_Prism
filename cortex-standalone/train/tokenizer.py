"""
train/tokenizer.py

The Python (training-time) copy of the tokenizer contract (docs/CONTRACTS.md §3). Three
copies of this algorithm exist on purpose — the TypeScript core, this trainer, and the
browser widget — because each runtime can't import the others. All three are pinned
against one shared fixture (`tests/fixtures/tokenizer.parity.json`), and `test_tokenizer.py`
here is the Python half of that pin.

The algorithm, byte-for-byte:

  1. Lowercase (if `lower`) — Python `str.lower()` / JS `toLowerCase()`.
  2. Split into maximal runs of Unicode Letter or Number codepoints. Python checks
     `unicodedata.category(ch)[0] in ("L", "N")`; JS checks `/[\\p{L}\\p{N}]/u`. Both
     target the same Unicode General Category groups. Everything else is a separator and
     is discarded.
  3. Map each word to `vocab[word]`, else `unkId`.
  4. Truncate to `maxLen`, pad with `padId`.

Standard library only — no `regex` package, no NLP dependency — so parity with the other
two copies is a matter of the same small algorithm, not a shared library version.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
TOKENIZER_VERSION = "1"


def is_word_char(ch: str) -> bool:
    """True for any Unicode Letter (category `L*`) or Number (category `N*`) codepoint —
    the Python spelling of `/[\\p{L}\\p{N}]/u` (CONTRACTS.md §3.1)."""
    return unicodedata.category(ch)[0] in ("L", "N")


def word_split(text: str) -> List[str]:
    """Split `text` into maximal runs of word characters, discarding everything else."""
    tokens: List[str] = []
    cur: List[str] = []
    for ch in text:
        if is_word_char(ch):
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


@dataclass
class TokenizerConfig:
    version: str
    lower: bool
    max_len: int
    pad_id: int
    unk_id: int
    vocab: Dict[str, int]
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN

    def to_json_dict(self) -> dict:
        """The exact `tokenizer.json` shape from CONTRACTS.md §3.3."""
        return {
            "version": self.version,
            "lower": self.lower,
            "maxLen": self.max_len,
            "padId": self.pad_id,
            "unkId": self.unk_id,
            "padToken": self.pad_token,
            "unkToken": self.unk_token,
            "vocab": self.vocab,
        }

    @staticmethod
    def from_json_dict(d: dict) -> "TokenizerConfig":
        return TokenizerConfig(
            version=str(d["version"]),
            lower=bool(d["lower"]),
            max_len=int(d["maxLen"]),
            pad_id=int(d["padId"]),
            unk_id=int(d["unkId"]),
            vocab=dict(d["vocab"]),
            pad_token=d.get("padToken", PAD_TOKEN),
            unk_token=d.get("unkToken", UNK_TOKEN),
        )


def build_vocab(utterances: Iterable[str], lower: bool = True) -> Dict[str, int]:
    """Deterministically build a word -> id vocab from the TRAIN split. Ids start at 2
    (0/1 are reserved for `<pad>`/`<unk>`) and are assigned by (descending frequency, then
    alphabetical) — sorted explicitly, never relying on dict/Counter iteration order
    (CONTRACTS.md §3.3)."""
    counts: Counter[str] = Counter()
    for u in utterances:
        text = u.lower() if lower else u
        counts.update(word_split(text))

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab: Dict[str, int] = {}
    next_id = 2
    for word, _count in ordered:
        vocab[word] = next_id
        next_id += 1
    return vocab


def build_tokenizer_config(utterances: Sequence[str], max_len: int, lower: bool = True) -> TokenizerConfig:
    vocab = build_vocab(utterances, lower=lower)
    return TokenizerConfig(
        version=TOKENIZER_VERSION,
        lower=lower,
        max_len=max_len,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        vocab=vocab,
    )


def tokenize(text: str, config: TokenizerConfig) -> List[int]:
    """text -> a fixed-length `config.max_len` list of ids (padded with `config.pad_id`,
    truncated at `config.max_len`, OOV words mapped to `config.unk_id`). MUST match the
    core's `tokenize()` exactly for the same `(text, config)`."""
    lowered = text.lower() if config.lower else text
    words = word_split(lowered)
    ids = [config.vocab.get(w, config.unk_id) for w in words[: config.max_len]]
    while len(ids) < config.max_len:
        ids.append(config.pad_id)
    return ids


def load_tokenizer_config(path: str) -> TokenizerConfig:
    with open(path, "r", encoding="utf-8") as f:
        return TokenizerConfig.from_json_dict(json.load(f))


def save_tokenizer_config(config: TokenizerConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_json_dict(), f, indent=2, sort_keys=False)
        f.write("\n")
