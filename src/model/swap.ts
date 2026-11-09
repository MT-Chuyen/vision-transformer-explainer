import { forward, gelu, geluPrime, type ModelFile } from "./forward";
import { probabilities } from "./sample";

export type SwapRow = {
  label: string;
  model: boolean;
  predicted: string;
  trueProb: number;
  gradients: number[];
};

export type Activation = "gelu" | "relu" | "sigmoid" | "linear";

function activate(kind: Activation, value: number): number {
  if (kind === "gelu") return gelu(value);
  if (kind === "relu") return value > 0 ? value : 0;
  if (kind === "sigmoid") {
    const clipped = Math.max(-40, Math.min(40, value));
    return 1 / (1 + Math.exp(-clipped));
  }
  return value;
}

function activatePrime(kind: Activation, value: number): number {
  if (kind === "gelu") return geluPrime(value);
  if (kind === "relu") return value > 0 ? 1 : 0;
  if (kind === "sigmoid") {
    const output = activate("sigmoid", value);
    return output * (1 - output);
  }
  return 1;
}

function argmax(values: number[]): number {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) if (values[index] > values[best]) best = index;
  return best;
}

function lengthOf(values: number[]): number {
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
}

function normalize(row: number[], weight: number[], bias: number[], eps: number): number[] {
  let mean = 0;
  for (const value of row) mean += value;
  mean /= row.length;
  let variance = 0;
  for (const value of row) variance += (value - mean) * (value - mean);
  variance /= row.length;
  const inverse = 1 / Math.sqrt(variance + eps);
  return row.map((value, index) => (value - mean) * inverse * weight[index] + bias[index]);
}

function linear(row: number[], weight: number[], bias: number[], out: number, input: number): number[] {
  const result = new Array<number>(out);
  for (let output = 0; output < out; output += 1) {
    let sum = bias[output];
    const base = output * input;
    for (let column = 0; column < input; column += 1) sum += row[column] * weight[base + column];
    result[output] = sum;
  }
  return result;
}

function linearBack(grad: number[], weight: number[], out: number, input: number): number[] {
  const result = new Array<number>(input).fill(0);
  for (let output = 0; output < out; output += 1) {
    const base = output * input;
    for (let column = 0; column < input; column += 1) result[column] += weight[base + column] * grad[output];
  }
  return result;
}

function normBack(grad: number[], row: number[], weight: number[], eps: number): number[] {
  const count = row.length;
  let mean = 0;
  for (const value of row) mean += value;
  mean /= count;
  let variance = 0;
  for (const value of row) variance += (value - mean) * (value - mean);
  variance /= count;
  const inverse = 1 / Math.sqrt(variance + eps);
  const hat = row.map((value) => (value - mean) * inverse);
  const scaled = grad.map((value, index) => value * weight[index]);
  let sum = 0;
  let sumHat = 0;
  for (let index = 0; index < count; index += 1) {
    sum += scaled[index];
    sumHat += scaled[index] * hat[index];
  }
  return scaled.map((value, index) => inverse * (value - sum / count - hat[index] * sumHat / count));
}

function replay(
  model: ModelFile,
  start: number[],
  attention: number[][],
  kind: Activation,
  residual: boolean,
  skipScale: number,
  classIndex: number,
): { trueProb: number; predicted: number; gradients: number[] } {
  const dim = model.config.dim;
  const hidden = model.config.mlpRatio * dim;
  const eps = model.config.eps;
  let stream = start.slice();
  const cache: { pre: number[]; up: number[] }[] = [];
  for (let block = 0; block < model.blocks.length; block += 1) {
    const weights = model.blocks[block];
    const joined = stream.map((value, index) => value + attention[block][index]);
    const normed = normalize(joined, weights.ln2.w, weights.ln2.b, eps);
    const up = linear(normed, weights.fc1.w, weights.fc1.b, hidden, dim);
    const down = linear(up.map((value) => activate(kind, value)), weights.fc2.w, weights.fc2.b, dim, hidden);
    stream = residual ? joined.map((value, index) => skipScale * value + down[index]) : down.slice();
    cache.push({ pre: joined, up });
  }
  const final = normalize(stream, model.norm.w, model.norm.b, eps);
  const logits = linear(final, model.head.w, model.head.b, model.config.classes.length, dim);
  const probs = probabilities(logits, 1);
  let grad = model.head.w.slice(classIndex * dim, classIndex * dim + dim);
  grad = normBack(grad, stream, model.norm.w, eps);
  const atBlock: number[] = [];
  for (let block = model.blocks.length - 1; block >= 0; block -= 1) {
    atBlock[block] = lengthOf(grad);
    const weights = model.blocks[block];
    const saved = cache[block];
    const throughDown = linearBack(grad, weights.fc2.w, dim, hidden);
    const throughAct = throughDown.map((value, index) => value * activatePrime(kind, saved.up[index]));
    const throughUp = linearBack(throughAct, weights.fc1.w, hidden, dim);
    const throughNorm = normBack(throughUp, saved.pre, weights.ln2.w, eps);
    grad = residual ? grad.map((value, index) => skipScale * value + throughNorm[index]) : throughNorm;
  }
  return {
    trueProb: probs[classIndex],
    predicted: argmax(probs),
    gradients: [lengthOf(grad), ...atBlock],
  };
}

export type SwapBasis = {
  trueClass: string;
  classIndex: number;
  start: number[];
  attention: number[][];
};

export function swapBasis(model: ModelFile, pixels: Float32Array): SwapBasis {
  const trace = forward(model, pixels, { mask: "full", dropout: 0, seed: 1 });
  const base = probabilities(trace.logits, 1);
  const classIndex = argmax(base);
  return {
    trueClass: model.config.classes[classIndex],
    classIndex,
    start: trace.afterPos[0],
    attention: trace.blocks.map((block) => block.droppedAttn[0]),
  };
}

export function evaluateSwap(
  model: ModelFile,
  basis: SwapBasis,
  kind: Activation,
  residual: boolean,
  skipScale: number,
  label: string,
  trained: boolean,
): SwapRow {
  const result = replay(model, basis.start, basis.attention, kind, residual, skipScale, basis.classIndex);
  return {
    label,
    model: trained,
    predicted: model.config.classes[result.predicted],
    trueProb: result.trueProb,
    gradients: result.gradients,
  };
}

export function architectureSwaps(model: ModelFile, pixels: Float32Array): { trueClass: string; rows: SwapRow[] } {
  const basis = swapBasis(model, pixels);
  const specs: { label: string; model: boolean; kind: Activation; residual: boolean; skipScale: number }[] = [
    { label: "GELU + skip", model: true, kind: "gelu", residual: true, skipScale: 1 },
    { label: "ReLU instead", model: false, kind: "relu", residual: true, skipScale: 1 },
    { label: "sigmoid instead", model: false, kind: "sigmoid", residual: true, skipScale: 1 },
    { label: "sigmoid, no skip", model: false, kind: "sigmoid", residual: false, skipScale: 1 },
    { label: "skip × 3", model: false, kind: "gelu", residual: true, skipScale: 3 },
  ];
  return {
    trueClass: basis.trueClass,
    rows: specs.map((spec) => evaluateSwap(model, basis, spec.kind, spec.residual, spec.skipScale, spec.label, spec.model)),
  };
}
