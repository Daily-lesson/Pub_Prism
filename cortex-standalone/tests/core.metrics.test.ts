import { describe, expect, it } from 'vitest';
import { accuracy, confusionMatrix, perLabelPRF1, slotF1, perSlotF1, tagTokens, macroF1, evaluate, diffPerIntentF1, PER_INTENT_F1_REGRESSION_EPSILON } from '../core/index';

// Hand-computed fixture: 6 examples over labels a/b/c.
const golds = ['a', 'a', 'a', 'b', 'b', 'c'];
const preds = ['a', 'a', 'b', 'b', 'c', 'c'];
// a: tp=2 fp=0 fn=1 → P=1, R=2/3, F1=0.8
// b: tp=1 fp=1 fn=1 → P=0.5 R=0.5 F1=0.5
// c: tp=1 fp=1 fn=0 → P=0.5 R=1 F1=2/3

describe('metrics', () => {
  it('accuracy + confusion matrix', () => {
    expect(accuracy(golds, preds)).toBeCloseTo(4 / 6, 10);
    expect(confusionMatrix(golds, preds, ['a', 'b', 'c'])).toEqual({
      a: { a: 2, b: 1, c: 0 },
      b: { a: 0, b: 1, c: 1 },
      c: { a: 0, b: 0, c: 1 },
    });
    expect(() => accuracy(['a'], [])).toThrow(/length mismatch/);
  });

  it('per-label P/R/F1 and macro F1 match hand computation', () => {
    const p = perLabelPRF1(golds, preds, ['a', 'b', 'c']);
    expect(p.a).toEqual({ precision: 1, recall: 2 / 3, f1: 0.8, support: 3 });
    expect(p.b).toEqual({ precision: 0.5, recall: 0.5, f1: 0.5, support: 2 });
    expect(p.c.precision).toBe(0.5);
    expect(p.c.recall).toBe(1);
    expect(p.c.f1).toBeCloseTo(2 / 3, 10);
    expect(macroF1(p)).toBeCloseTo((0.8 + 0.5 + 2 / 3) / 3, 10);
  });

  it('token-level slot F1', () => {
    const gold = [{ utterance: 'reports for North campus now', slots: [{ name: 'region', value: 'North campus', start: 12, end: 24 }] }];
    expect(tagTokens(gold[0].utterance, gold[0].slots)).toEqual(['O', 'O', 'region', 'region', 'O']);
    const pred = [{ utterance: 'reports for North campus now', slots: [{ name: 'region', value: 'North', start: 12, end: 17 }] }];
    // gold positive 2, pred positive 1, tp 1 → P=1 R=0.5 F1=2/3
    const f = slotF1(gold, pred);
    expect(f).toEqual({ precision: 1, recall: 0.5, f1: 2 / 3, support: 2 });
    expect(perSlotF1(gold, pred, ['region']).region.f1).toBeCloseTo(2 / 3, 10);
  });

  it('evaluate() wires the pieces for a (sync or async) classifier', async () => {
    const split = [
      { utterance: 'one', intent: 'a', slots: [] },
      { utterance: 'two', intent: 'b', slots: [] },
    ];
    const report = await evaluate(async (u) => ({ intent: u === 'one' ? 'a' : 'a', slots: [] }), split, { labels: ['a', 'b'], slotNames: [] });
    expect(report.n).toBe(2);
    expect(report.accuracy).toBe(0.5);
    expect(report.confusion.b.a).toBe(1);
  });

  it('diffPerIntentF1: only baseline intents, drops beyond epsilon', () => {
    expect(PER_INTENT_F1_REGRESSION_EPSILON).toBe(0.03);
    const out = diffPerIntentF1({ a: 0.9, b: 0.9, c: 0.9 }, { a: 0.9, b: 0.86, c: 0.875, d: 0.1 });
    expect(out).toEqual([{ intent: 'b', ledgerF1: 0.9, currentF1: 0.86, delta: expect.closeTo(-0.04, 10) }]);
    expect(diffPerIntentF1({ a: 0.5 }, {})).toEqual([{ intent: 'a', ledgerF1: 0.5, currentF1: 0, delta: -0.5 }]);
    expect(diffPerIntentF1({ a: 0.9 }, { a: 0.8 }, 0.2)).toEqual([]);
  });
});
