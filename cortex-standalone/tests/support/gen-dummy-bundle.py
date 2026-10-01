#!/usr/bin/env python3
"""tests/support/gen-dummy-bundle.py — a RANDOM-WEIGHT model bundle for tests.

Emits a bundle shaped exactly like the trainer's (docs/CONTRACTS.md §3.3, §4.1–4.5):

    <slug>-<version>.onnx
    <slug>-<version>.tokenizer.json
    <slug>-<version>.labels.json
    ledger.json            (real sha256s of the three files above)

so the widget's ONNX path (fetch → sha256 verify → onnxruntime-web → decode) can be
exercised before a real trained model exists. The weights are random and seeded; the
predictions mean nothing. Metrics in the ledger are zeros and the ledger says so.

    python tests/support/gen-dummy-bundle.py [--registry PATH] [--out DIR] [--version V]
                                             [--max-len 16] [--seed 42]
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_REGISTRY = os.path.join(PKG, 'registry', 'examples', 'ops-dashboard.registry.json')
DEFAULT_OUT = os.path.join(PKG, 'build', 'dummy-model')
BUILTIN_META = ['greeting', 'out_of_domain']


def word_split(text):
    """CONTRACTS §3.1: maximal runs of Unicode Letter/Number codepoints."""
    tokens, cur = [], ''
    for ch in text:
        if unicodedata.category(ch)[0] in ('L', 'N'):
            cur += ch
        elif cur:
            tokens.append(cur)
            cur = ''
    if cur:
        tokens.append(cur)
    return tokens


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


class DummyCortex(nn.Module):
    """CONTRACTS §4.4: Embedding -> BiGRU -> masked mean-pool intent head + per-token slot head."""

    def __init__(self, vocab_size, embed_dim, hidden, num_intents, num_slot_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.intent_fc1 = nn.Linear(2 * hidden, hidden)
        self.intent_fc2 = nn.Linear(hidden, num_intents)
        self.slot_fc = nn.Linear(2 * hidden, num_slot_labels)

    def forward(self, input_ids):
        mask = (input_ids != 0).unsqueeze(-1).to(torch.float32)
        x = self.embed(input_ids)
        h, _ = self.gru(x)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        intent_logits = self.intent_fc2(torch.relu(self.intent_fc1(pooled)))
        slot_logits = self.slot_fc(h)
        return intent_logits, slot_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=DEFAULT_REGISTRY)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--version', default='0.0.0-dummy')
    ap.add_argument('--max-len', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--embed-dim', type=int, default=16)
    ap.add_argument('--hidden', type=int, default=12)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    with open(args.registry, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    slug = registry['app']['slug']

    # Taxonomy order (§1.6): declaration order, then the built-in meta intents when absent.
    intents = [it['id'] for it in registry['intents']]
    for meta in BUILTIN_META:
        if meta not in intents:
            intents.append(meta)
    slot_names = list(registry.get('slots', {}).keys())
    slot_labels = ['O'] + [lab for s in slot_names for lab in ('B-' + s, 'I-' + s)]

    # Vocab (§3.3): descending frequency then alphabetical, ids from 2, over the registry's
    # own text (templates with placeholders filled from every vocab label, plus paraphrases).
    counts = Counter()
    for it in registry['intents']:
        for t in it.get('templates', []):
            filled = [t]
            for name in it.get('slots', []):
                vocab = registry['slots'].get(name, {}).get('vocab', [])
                nxt = []
                for s in filled:
                    if '{' + name + '}' in s and vocab:
                        nxt.extend(s.replace('{' + name + '}', v['label']) for v in vocab)
                    else:
                        nxt.append(s)
                filled = nxt
            for s in filled:
                counts.update(word_split(re.sub(r'\{[a-z_]+\}', '', s.lower())))
        for p in it.get('paraphrases', []):
            counts.update(word_split(p.lower()))
    for entry in registry.get('heldout', []):
        pass  # never compiled into the vocab (§2.6 spirit)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = {w: i + 2 for i, (w, _) in enumerate(ordered)}
    vocab_size = len(vocab) + 2

    tokenizer = {
        'version': args.version, 'lower': True, 'maxLen': args.max_len,
        'padId': 0, 'unkId': 1, 'padToken': '<pad>', 'unkToken': '<unk>', 'vocab': vocab,
    }
    labels = {'version': args.version, 'intents': intents, 'slots': slot_labels}

    model = DummyCortex(vocab_size, args.embed_dim, args.hidden, len(intents), len(slot_labels)).eval()
    param_count = sum(p.numel() for p in model.parameters())

    os.makedirs(args.out, exist_ok=True)
    stem = '%s-%s' % (slug, args.version)
    onnx_path = os.path.join(args.out, stem + '.onnx')
    tok_path = os.path.join(args.out, stem + '.tokenizer.json')
    lab_path = os.path.join(args.out, stem + '.labels.json')

    dummy = torch.zeros((1, args.max_len), dtype=torch.int64)
    export_kwargs = dict(
        input_names=['input_ids'], output_names=['intent_logits', 'slot_logits'],
        dynamic_axes={'input_ids': {0: 'batch'}, 'intent_logits': {0: 'batch'}, 'slot_logits': {0: 'batch'}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, (dummy,), onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, (dummy,), onnx_path, **export_kwargs)

    with open(tok_path, 'w', encoding='utf-8') as f:
        f.write(canonical_json(tokenizer))
    with open(lab_path, 'w', encoding='utf-8') as f:
        f.write(canonical_json(labels))

    # Sanity: the exported graph runs under onnxruntime with the contract shapes.
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ids = np.zeros((1, args.max_len), dtype=np.int64)
    ids[0, 0] = vocab.get('hello', 1)
    out = sess.run(None, {'input_ids': ids})
    assert out[0].shape == (1, len(intents)), out[0].shape
    assert out[1].shape == (1, args.max_len, len(slot_labels)), out[1].shape

    ledger = {
        'version': args.version,
        'app': slug,
        'note': 'RANDOM-WEIGHT TEST BUNDLE generated by tests/support/gen-dummy-bundle.py. Predictions are meaningless; metrics are zeros.',
        'onnx': {'file': os.path.basename(onnx_path), 'sha256': sha256_file(onnx_path), 'bytes': os.path.getsize(onnx_path), 'quantized': False},
        'tokenizer': {'file': os.path.basename(tok_path), 'sha256': sha256_file(tok_path)},
        'labels': {'file': os.path.basename(lab_path), 'sha256': sha256_file(lab_path)},
        'trainedFrom': {'registryHash': hashlib.sha256(canonical_json(registry).encode('utf-8')).hexdigest(), 'datasetHash': '0' * 64, 'seed': args.seed, 'split': {'train': 0, 'val': 0, 'test': 0}},
        'modelConfig': {'vocabSize': vocab_size, 'maxLen': args.max_len, 'embedDim': args.embed_dim, 'gruHidden': args.hidden, 'numIntents': len(intents), 'numSlotLabels': len(slot_labels), 'paramCount': param_count, 'epochs': 0},
        'metrics': {
            'inDistribution': {'intentAccuracy': 0, 'macroF1': 0, 'perIntentF1': {i: 0 for i in intents}, 'slotF1': 0, 'slotPrecision': 0, 'slotRecall': 0, 'perSlotF1': {s: 0 for s in slot_names}},
            'heldout': None,
        },
        'heldoutGate': 'not declared',
        'acceptanceFloor': {'inDistributionIntentAccuracy': 0.9, 'inDistributionSlotF1': 0.9, 'heldoutIntentAccuracy': 0.85},
    }
    with open(os.path.join(args.out, 'ledger.json'), 'w', encoding='utf-8') as f:
        json.dump(ledger, f, indent=2)

    print(json.dumps({'out': args.out, 'intents': len(intents), 'slotLabels': slot_labels, 'vocabSize': vocab_size, 'onnxBytes': ledger['onnx']['bytes']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
