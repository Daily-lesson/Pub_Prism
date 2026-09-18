export interface SlotSpan {
  name: string;
  /** Always `utterance.slice(start, end)`. */
  value: string;
  start: number;
  end: number;
}

export interface GeneratedExample {
  utterance: string;
  intent: string;
  slots: SlotSpan[];
}
