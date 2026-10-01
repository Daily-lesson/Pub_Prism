"""
train/labels.py

BIO slot-label scheme + gold-label construction from the compiled dataset's char-span
`slots[]` (docs/CONTRACTS.md §2.7–2.8). Training-side only: this module goes from gold
character spans to gold per-token BIO ids so the loss and the eval metrics have something
to compare model output against. (Decoding PREDICTED tags back into spans is an inference
concern — see `gen_session_parity_fixture.py`.)

Nothing here hardcodes a slot name. The label list is ALWAYS the manifest's explicit
`slotLabels` array (`["O"] + flatMap(slotNames, s => ["B-"+s, "I-"+s])`), threaded in
by the caller — a registry decides what slots exist, this file never does.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from tokenizer import is_word_char

O_LABEL = "O"


def build_slot_labels(slot_names: Sequence[str]) -> List[str]:
    """`["O"] + flatMap(slotNames, s => ["B-"+s, "I-"+s])` — the same derivation the
    compiler uses for `manifest.slotLabels`. Used to cross-check a manifest, never to
    replace what it declares."""
    labels = [O_LABEL]
    for name in slot_names:
        labels.append(f"B-{name}")
        labels.append(f"I-{name}")
    return labels


def label_to_id_map(slot_labels: Sequence[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(slot_labels)}


def word_split_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Same word-boundary rule as `tokenizer.word_split`, but keeps each token's
    `[start, end)` offset in `text`. Always called on the ORIGINAL-cased utterance
    (CONTRACTS.md §3.5 — casing never moves boundaries), so token COUNT and POSITION line
    up 1:1 with `tokenizer.tokenize()`'s ids over the lowercased string."""
    tokens: List[Tuple[str, int, int]] = []
    cur: List[str] = []
    cur_start = -1
    for i, ch in enumerate(text):
        if is_word_char(ch):
            if cur_start == -1:
                cur_start = i
            cur.append(ch)
        elif cur:
            tokens.append(("".join(cur), cur_start, i))
            cur = []
            cur_start = -1
    if cur:
        tokens.append(("".join(cur), cur_start, len(text)))
    return tokens


def bio_ids_for_example(
    utterance: str,
    slots: Sequence[dict],
    max_len: int,
    slot_labels: Sequence[str],
) -> List[int]:
    """Gold per-token BIO label ids for the first `max_len` tokens of `utterance`, padded
    with the `O` id out to `max_len`. A token overlapping a slot's `[start, end)` char span
    gets `B-<name>` on the span's first token and `I-<name>` after; everything else is `O`.
    Padding positions are masked out of the loss separately (train.py, `ignore_index`)."""
    label_to_id = label_to_id_map(slot_labels)
    o_id = label_to_id[O_LABEL]
    toks = word_split_with_offsets(utterance)[:max_len]
    ids: List[int] = []
    started: set = set()
    for (_w, s, e) in toks:
        hit = None
        for slot in slots:
            if s < slot["end"] and e > slot["start"]:
                hit = slot
                break
        if hit is None:
            ids.append(o_id)
        else:
            key = (hit["name"], hit["start"], hit["end"])
            prefix = "I" if key in started else "B"
            started.add(key)
            label = f"{prefix}-{hit['name']}"
            if label not in label_to_id:
                raise ValueError(
                    f"slot {hit['name']!r} in utterance {utterance!r} is not in the manifest's slotLabels"
                )
            ids.append(label_to_id[label])
    while len(ids) < max_len:
        ids.append(o_id)
    return ids


def collapse_bio_to_name(label: str) -> str:
    """'B-region'/'I-region' -> 'region'; 'O' -> 'O'. Used at EVAL time so slot metrics are
    per-token by slot NAME with no B/I distinction (the shared metric definition)."""
    return label if label == O_LABEL else label.split("-", 1)[1]
