export type Linear = { w: number[]; b: number[] };
export type Norm = { w: number[]; b: number[] };

export type BlockWeights = {
  ln1: Norm;
  qkv: Linear;
  proj: Linear;
  ln2: Norm;
  fc1: Linear;
  fc2: Linear;
};

export type Example = {
  id: string;
  label: string;
  pixels: number[];
  logits: number[];
  debug?: { afterPos: number[]; attnHead0: number[] };
};

export type ModelFile = {
  config: {
    imageSize: number;
    patchSize: number;
    dim: number;
    heads: number;
    depth: number;
    mlpRatio: number;
    eps: number;
    classes: string[];
    valAccuracy: number;
  };
  patchW: number[];
  patchB: number[];
  cls: number[];
  pos: number[];
  blocks: BlockWeights[];
  norm: Norm;
  head: Linear;
  examples: Example[];
};

export type ForwardOptions = {
  mask: "full" | "causal";
  dropout: number;
  seed: number;
};

export type BlockTrace = {
  ln1: number[][];
  query: number[][][];
  key: number[][][];
  value: number[][][];
  scores: number[][][];
  weights: number[][][];
  headOut: number[][][];
  concat: number[][];
  proj: number[][];
  droppedAttn: number[][];
  resid1: number[][];
  ln2: number[][];
  mlpUp: number[][];
  activated: number[][];
  mlpDown: number[][];
  droppedMlp: number[][];
  resid2: number[][];
};

export type Trace = {
  patchTokens: number[][];
  afterPos: number[][];
  blocks: BlockTrace[];
  classVector: number[];
  logits: number[];
};

function erf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

export function gelu(value: number): number {
  const exact = (Math as Math & { erf?: (x: number) => number }).erf;
  const used = exact ? exact(value / Math.SQRT2) : erf(value / Math.SQRT2);
  return 0.5 * value * (1 + used);
}

export function geluPrime(value: number): number {
  const exact = (Math as Math & { erf?: (x: number) => number }).erf;
  const used = exact ? exact(value / Math.SQRT2) : erf(value / Math.SQRT2);
  return 0.5 * (1 + used) + (value * Math.exp(-0.5 * value * value)) / Math.sqrt(2 * Math.PI);
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let next = state;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function applyLinear(rows: number[][], layer: Linear, out: number, input: number): number[][] {
  return rows.map((row) => {
    const result = new Array<number>(out);
    for (let o = 0; o < out; o += 1) {
      let sum = layer.b[o];
      const base = o * input;
      for (let i = 0; i < input; i += 1) sum += row[i] * layer.w[base + i];
      result[o] = sum;
    }
    return result;
  });
}

function applyNorm(rows: number[][], layer: Norm, eps: number): number[][] {
  return rows.map((row) => {
    let mean = 0;
    for (const value of row) mean += value;
    mean /= row.length;
    let variance = 0;
    for (const value of row) {
      const delta = value - mean;
      variance += delta * delta;
    }
    variance /= row.length;
    const inverse = 1 / Math.sqrt(variance + eps);
    return row.map((value, index) => (value - mean) * inverse * layer.w[index] + layer.b[index]);
  });
}

function softmaxRows(rows: number[][]): number[][] {
  return rows.map((row) => {
    let peak = -Infinity;
    for (const value of row) peak = Math.max(peak, value);
    const exps = row.map((value) => (value === -Infinity ? 0 : Math.exp(value - peak)));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map((value) => value / total);
  });
}

function dropoutRows(rows: number[][], rate: number, random: () => number): number[][] {
  if (rate <= 0) return rows;
  const keep = 1 - rate;
  return rows.map((row) => row.map((value) => (random() < rate ? 0 : value / keep)));
}

function addRows(left: number[][], right: number[][]): number[][] {
  return left.map((row, rowIndex) => row.map((value, column) => value + right[rowIndex][column]));
}

function patchEmbed(model: ModelFile, pixels: ArrayLike<number>): number[][] {
  const { imageSize, patchSize, dim } = model.config;
  const grid = imageSize / patchSize;
  const plane = imageSize * imageSize;
  const tokens: number[][] = [];
  for (let row = 0; row < grid; row += 1) {
    for (let col = 0; col < grid; col += 1) {
      const token = model.patchB.slice();
      for (let out = 0; out < dim; out += 1) {
        let sum = model.patchB[out];
        let weight = out * 3 * patchSize * patchSize;
        for (let channel = 0; channel < 3; channel += 1) {
          for (let ky = 0; ky < patchSize; ky += 1) {
            for (let kx = 0; kx < patchSize; kx += 1) {
              const y = row * patchSize + ky;
              const x = col * patchSize + kx;
              sum += pixels[channel * plane + y * imageSize + x] * model.patchW[weight];
              weight += 1;
            }
          }
        }
        token[out] = sum;
      }
      tokens.push(token);
    }
  }
  return tokens;
}

function splitHeads(rows: number[][], offset: number, heads: number, headDim: number): number[][][] {
  const tokens = rows.length;
  const grouped = Array.from({ length: heads }, () =>
    Array.from({ length: tokens }, () => new Array<number>(headDim)),
  );
  for (let token = 0; token < tokens; token += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let dim = 0; dim < headDim; dim += 1) {
        grouped[head][token][dim] = rows[token][offset + head * headDim + dim];
      }
    }
  }
  return grouped;
}

function attend(
  query: number[][],
  key: number[][],
  value: number[][],
  mask: "full" | "causal",
): { scores: number[][]; weights: number[][]; out: number[][] } {
  const tokens = query.length;
  const headDim = query[0].length;
  const scale = 1 / Math.sqrt(headDim);
  const scores = query.map((qRow, qIndex) =>
    key.map((kRow, kIndex) => {
      if (mask === "causal" && kIndex > qIndex) return -Infinity;
      let dot = 0;
      for (let dim = 0; dim < headDim; dim += 1) dot += qRow[dim] * kRow[dim];
      return dot * scale;
    }),
  );
  const weights = softmaxRows(scores);
  const out = weights.map((row) => {
    const mixed = new Array<number>(headDim).fill(0);
    for (let token = 0; token < tokens; token += 1) {
      for (let dim = 0; dim < headDim; dim += 1) mixed[dim] += row[token] * value[token][dim];
    }
    return mixed;
  });
  return { scores, weights, out };
}

export function forward(model: ModelFile, pixels: ArrayLike<number>, options: ForwardOptions): Trace {
  const { dim, heads, mlpRatio, eps } = model.config;
  const headDim = dim / heads;
  const random = mulberry32(options.seed);
  const patchTokens = patchEmbed(model, pixels);
  const withClass = [model.cls.slice(), ...patchTokens];
  const tokenCount = withClass.length;
  const afterPos = withClass.map((row, index) => {
    const base = index * dim;
    return row.map((value, column) => value + model.pos[base + column]);
  });

  let hidden = afterPos;
  const blocks: BlockTrace[] = [];
  for (const block of model.blocks) {
    const ln1 = applyNorm(hidden, block.ln1, eps);
    const qkv = applyLinear(ln1, block.qkv, dim * 3, dim);
    const query = splitHeads(qkv, 0, heads, headDim);
    const key = splitHeads(qkv, dim, heads, headDim);
    const value = splitHeads(qkv, dim * 2, heads, headDim);
    const scores: number[][][] = [];
    const weights: number[][][] = [];
    const headOut: number[][][] = [];
    for (let head = 0; head < heads; head += 1) {
      const result = attend(query[head], key[head], value[head], options.mask);
      scores.push(result.scores);
      weights.push(result.weights);
      headOut.push(result.out);
    }
    const concat = Array.from({ length: tokenCount }, (_, token) => {
      const row = new Array<number>(dim);
      for (let head = 0; head < heads; head += 1) {
        for (let column = 0; column < headDim; column += 1) {
          row[head * headDim + column] = headOut[head][token][column];
        }
      }
      return row;
    });
    const proj = applyLinear(concat, block.proj, dim, dim);
    const droppedAttn = dropoutRows(proj, options.dropout, random);
    const resid1 = addRows(hidden, droppedAttn);
    const ln2 = applyNorm(resid1, block.ln2, eps);
    const mlpUp = applyLinear(ln2, block.fc1, dim * mlpRatio, dim);
    const activated = mlpUp.map((row) => row.map(gelu));
    const mlpDown = applyLinear(activated, block.fc2, dim, dim * mlpRatio);
    const droppedMlp = dropoutRows(mlpDown, options.dropout, random);
    const resid2 = addRows(resid1, droppedMlp);
    blocks.push({
      ln1,
      query,
      key,
      value,
      scores,
      weights,
      headOut,
      concat,
      proj,
      droppedAttn,
      resid1,
      ln2,
      mlpUp,
      activated,
      mlpDown,
      droppedMlp,
      resid2,
    });
    hidden = resid2;
  }

  const classVector = applyNorm([hidden[0]], model.norm, eps)[0];
  const logits = applyLinear([classVector], model.head, model.config.classes.length, dim)[0];
  return { patchTokens, afterPos, blocks, classVector, logits };
}

export function tokenNames(grid: number): { name: string; row: number | null; col: number | null }[] {
  const names: { name: string; row: number | null; col: number | null }[] = [{ name: "CLS", row: null, col: null }];
  for (let row = 0; row < grid; row += 1) {
    for (let col = 0; col < grid; col += 1) {
      names.push({ name: `${row},${col}`, row, col });
    }
  }
  return names;
}
