import { mulberry32 } from "./forward";

export type Sampling = { type: "top-k" | "top-p"; value: number };

export function probabilities(logits: number[], temperature: number): number[] {
  const safe = Math.max(temperature, 0.05);
  let peak = -Infinity;
  for (const value of logits) peak = Math.max(peak, value / safe);
  const exps = logits.map((value) => Math.exp(value / safe - peak));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

export function sampleClass(
  probs: number[],
  sampling: Sampling,
  seed: number,
): { index: number; pool: boolean[] } {
  const order = probs.map((probability, index) => ({ probability, index }));
  order.sort((a, b) => b.probability - a.probability);
  const pool = new Array<boolean>(probs.length).fill(false);
  if (sampling.type === "top-k") {
    const k = Math.max(1, Math.min(probs.length, Math.round(sampling.value)));
    for (const item of order.slice(0, k)) pool[item.index] = true;
  } else {
    let covered = 0;
    for (const item of order) {
      pool[item.index] = true;
      covered += item.probability;
      if (covered >= sampling.value) break;
    }
  }
  const kept = probs.map((probability, index) => (pool[index] ? probability : 0));
  const mass = kept.reduce((sum, value) => sum + value, 0);
  const random = mulberry32(seed)();
  let cursor = 0;
  for (let index = 0; index < kept.length; index += 1) {
    cursor += kept[index] / mass;
    if (random <= cursor) return { index, pool };
  }
  return { index: order[0].index, pool };
}
