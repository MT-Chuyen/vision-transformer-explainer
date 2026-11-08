import { useMemo, type ReactNode } from "react";
import { forward, gelu, tokenNames, type ModelFile, type Trace } from "../model/forward";
import { probabilities } from "../model/sample";
import { Picture } from "./visual";

function dot(left: number[], right: number[]): number {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) sum += left[index] * right[index];
  return sum;
}

function cosine(left: number[], right: number[]): number {
  let product = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    product += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }
  const denom = Math.sqrt(leftNorm * rightNorm);
  return denom === 0 ? 0 : product / denom;
}

function softmax(row: number[]): number[] {
  let peak = -Infinity;
  for (const value of row) if (value > peak) peak = value;
  const exps = row.map((value) => (Number.isFinite(value) ? Math.exp(value - peak) : 0));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
}

function lengthOf(values: number[]): number {
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
}

function similarity(rows: number[][]): number[][] {
  return rows.map((left) => rows.map((right) => cosine(left, right)));
}

function meanOff(matrix: number[][]): number {
  let sum = 0;
  let count = 0;
  for (let row = 0; row < matrix.length; row += 1) {
    for (let column = 0; column < matrix.length; column += 1) {
      if (row === column) continue;
      sum += matrix[row][column];
      count += 1;
    }
  }
  return count === 0 ? 0 : sum / count;
}

export function Why({
  model,
  pixels,
  trace,
  block,
  head,
  token,
  mask,
}: {
  model: ModelFile;
  pixels: Float32Array;
  trace: Trace;
  block: number;
  head: number;
  token: number;
  mask: "full" | "causal";
}) {
  const view = useMemo(() => {
    const current = trace.blocks[block];
    const query = current.query[head][token];
    const keys = current.key[head];
    const raw = keys.map((key, index) => {
      if (mask === "causal" && index > token) return Number.NEGATIVE_INFINITY;
      return dot(query, key);
    });
    const scale = 1 / Math.sqrt(query.length);
    const plain = softmax(raw);
    const scaled = softmax(raw.map((value) => (Number.isFinite(value) ? value * scale : value)));
    const dim = model.config.dim;
    const placed = trace.patchTokens.map((row, index) => {
      const base = (index + 1) * dim;
      return row.map((value, column) => value + model.pos[base + column]);
    });
    const before = similarity(trace.patchTokens);
    const after = similarity(placed);
    const incoming = (block === 0 ? trace.afterPos : trace.blocks[block - 1].resid2)[token];
    const update = current.droppedAttn[token];
    const sum = current.resid1[token];
    const units = current.mlpUp[token];
    let negative = 0;
    let negativeGelu = 0;
    for (const value of units) {
      if (value < 0) {
        negative += 1;
        negativeGelu += gelu(value);
      }
    }
    return { plain, scaled, before, after, incoming, update, sum, units, negative, negativeGelu };
  }, [trace, block, head, token, mask, model]);

  const runs = useMemo(
    () => ({
      full: forward(model, pixels, { mask: "full", dropout: 0, seed: 1 }),
      causal: forward(model, pixels, { mask: "causal", dropout: 0, seed: 1 }),
    }),
    [model, pixels],
  );

  const wiring = useMemo(() => {
    const current = trace.blocks[block];
    const hidden = current.ln1;
    const content = softmax(hidden.map((row) => cosine(hidden[token], row)));
    const learned = current.weights[head][token];
    const width = current.value[head][0].length;
    const mix = (rows: number[][]) => {
      const mixed = new Array<number>(rows[0].length).fill(0);
      for (let time = 0; time < rows.length; time += 1) {
        for (let column = 0; column < mixed.length; column += 1) mixed[column] += learned[time] * rows[time][column];
      }
      return mixed;
    };
    const pre = block === 0 ? trace.afterPos : trace.blocks[block - 1].resid2;
    return {
      content,
      learned,
      fromValue: mix(current.value[head]),
      fromToken: mix(hidden.map((row) => row.slice(0, width))),
      pre: pre.map(lengthOf),
      post: hidden.map(lengthOf),
    };
  }, [trace, block, head, token]);

  const peak = Math.max(...view.plain, ...view.scaled, 1e-6);
  const { imageSize, patchSize } = model.config;
  const stream = lengthOf(view.incoming);
  const update = lengthOf(view.update);
  const shared = Math.max(...[...view.incoming, ...view.update, ...view.sum].map((value) => Math.abs(value)), 1e-6);

  return (
    <section data-testid="why" className="border-t border-slate-200 bg-white px-5 py-5">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">Why this operation</h2>
      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <Card title="softmax(q · k / √d)">
          <div className="grid grid-cols-2 gap-3">
            <Choice label="q · k" value={`${(Math.max(...view.plain) * 100).toFixed(0)}% on one token`}>
              <Picture pixels={pixels} imageSize={imageSize} patch={patchSize} overlay={{ weights: view.plain.slice(1), selected: token, hover: null }} grid={false} display={132} />
              <Bars values={view.plain} peak={peak} />
            </Choice>
            <Choice label="q · k / √d" value={`${(Math.max(...view.scaled) * 100).toFixed(0)}% on one token`} live>
              <Picture pixels={pixels} imageSize={imageSize} patch={patchSize} overlay={{ weights: view.scaled.slice(1), selected: token, hover: null }} grid={false} display={132} />
              <Bars values={view.scaled} peak={peak} />
            </Choice>
          </div>
        </Card>

        <Card title="patch + position">
          <div className="grid grid-cols-2 gap-3">
            <Choice label="patch only" value={`mean likeness ${meanOff(view.before).toFixed(2)}`}>
              <SimGrid matrix={view.before} />
            </Choice>
            <Choice label="patch + position" value={`mean likeness ${meanOff(view.after).toFixed(2)}`} live>
              <SimGrid matrix={view.after} />
            </Choice>
          </div>
        </Card>

        <Card title="h + f(h)">
          <div className="space-y-2">
            <Strip label="f(h)" values={view.update} peak={shared} note={update.toFixed(2)} />
            <Strip label="h" values={view.incoming} peak={shared} note={stream.toFixed(2)} />
            <Strip label="h + f(h)" values={view.sum} peak={shared} note={lengthOf(view.sum).toFixed(2)} live />
          </div>
          <p className="mt-2 text-[11px] leading-4 text-slate-500">
            f(h) lines up with h at {cosine(view.incoming, view.update).toFixed(2)}. h + f(h) lines up with h at {cosine(view.incoming, view.sum).toFixed(2)}.
          </p>
        </Card>

        <Card title="GELU, and 16 patches">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <Activation values={view.units} />
              <p className="mt-1 text-[11px] leading-4 text-slate-500">
                {view.negative} of {view.units.length} units are negative. GELU sums them to {view.negativeGelu.toFixed(1)}. ReLU sums them to 0.
              </p>
            </div>
            <div>
              <div className="grid grid-cols-2 items-end gap-2">
                <PixelMatrix />
                <div>
                  <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-indigo-700">17 × 17</div>
                  <SimGrid matrix={trace.blocks[block].weights[head]} cell={5} mode="weight" />
                </div>
              </div>
              <p className="mt-1 text-[11px] leading-4 text-slate-500">A pixel token would mix a 1024×1024 matrix. A patch token mixes the 17×17 matrix this head already has.</p>
            </div>
          </div>
        </Card>
      </div>

      <h2 className="mt-6 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">Why this wiring</h2>
      <div className="mt-3 grid gap-3 xl:grid-cols-3">
        <Card title="learned q · k">
          <QueryCompare
            pixels={pixels}
            imageSize={imageSize}
            patchSize={patchSize}
            token={token}
            content={wiring.content}
            learned={wiring.learned}
            fromToken={wiring.fromToken}
            fromValue={wiring.fromValue}
          />
        </Card>
        <Card title="LayerNorm">
          <Lengths before={wiring.pre} after={wiring.post} />
        </Card>
        <Card title="full attention">
          <MaskCompare
            model={model}
            pixels={pixels}
            imageSize={imageSize}
            patchSize={patchSize}
            full={runs.full}
            causal={runs.causal}
            block={block}
            head={head}
          />
        </Card>
      </div>
    </section>
  );
}

function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <article className="rounded-xl border border-slate-200 p-3">
      <h3 className="font-mono text-[12px] text-slate-800">{title}</h3>
      <div className="mt-2">{children}</div>
    </article>
  );
}

function Choice({ label, value, live, children }: { label: string; value: string; live?: boolean; children: ReactNode }) {
  return (
    <div>
      <div className={`text-[10px] font-semibold uppercase tracking-wide ${live ? "text-indigo-700" : "text-slate-400"}`}>{label}</div>
      <div className="text-[11px] tabular-nums text-slate-500">{value}</div>
      <div className="mt-1">{children}</div>
    </div>
  );
}

function Bars({ values, peak }: { values: number[]; peak: number }) {
  return (
    <div className="mt-1 flex h-8 items-end gap-px">
      {values.map((value, index) => (
        <span key={index} title={`${(value * 100).toFixed(1)}%`} className="min-w-0 flex-1 rounded-sm bg-indigo-500" style={{ height: `${Math.max(2, (value / peak) * 100)}%` }} />
      ))}
    </div>
  );
}

function SimGrid({ matrix, cell = 8, mode = "cosine" }: { matrix: number[][]; cell?: number; mode?: "cosine" | "weight" }) {
  return (
    <div className="inline-grid" style={{ gridTemplateColumns: `repeat(${matrix.length}, ${cell}px)`, gap: 1 }}>
      {matrix.map((row, rowIndex) =>
        row.map((value, column) => {
          const amount = mode === "weight" ? Math.min(1, Math.max(0, value) * 4) : Math.min(1, Math.max(0, (value - 0.15) / 0.85));
          return <span key={`${rowIndex}-${column}`} title={value.toFixed(2)} className="block" style={{ width: cell, height: cell, background: `rgba(67, 56, 202, ${0.08 + amount * 0.92})` }} />;
        }),
      )}
    </div>
  );
}

function Strip({ label, values, peak, note, live }: { label: string; values: number[]; peak: number; note: string; live?: boolean }) {
  return (
    <div className="grid grid-cols-[72px_1fr_36px] items-center gap-2">
      <span className={`font-mono text-[11px] ${live ? "font-semibold text-indigo-700" : "text-slate-500"}`}>{label}</span>
      <span className="flex gap-px overflow-hidden">
        {values.map((value, index) => {
          const amount = Math.min(1, Math.abs(value) / peak);
          return (
            <span
              key={index}
              className="h-3.5 w-1.5 shrink-0"
              style={{ background: value >= 0 ? `rgba(79, 70, 229, ${0.15 + amount * 0.85})` : `rgba(220, 38, 38, ${0.15 + amount * 0.85})` }}
            />
          );
        })}
      </span>
      <span className="text-right font-mono text-[10px] text-slate-400">{note}</span>
    </div>
  );
}

function Activation({ values }: { values: number[] }) {
  const width = 240;
  const height = 96;
  const x0 = -2.5;
  const x1 = 3;
  const yOf = (value: number) => height * 0.72 - value * 16;
  const xOf = (value: number) => ((value - x0) / (x1 - x0)) * width;
  const curve = (fn: (value: number) => number) =>
    Array.from({ length: 49 }, (_, index) => {
      const x = x0 + ((x1 - x0) * index) / 48;
      return `${xOf(x)},${yOf(fn(x))}`;
    }).join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
      <line x1="0" y1={yOf(0)} x2={width} y2={yOf(0)} stroke="#e2e8f0" />
      <polyline fill="none" stroke="#94a3b8" strokeWidth="1.5" points={curve((value) => Math.max(0, value))} />
      <polyline fill="none" stroke="#ea580c" strokeWidth="2" points={curve(gelu)} />
      {values.map((value, index) => {
        const x = Math.min(x1, Math.max(x0, value));
        return (
          <g key={index}>
            <circle cx={xOf(x)} cy={yOf(Math.max(0, value))} r="1.5" fill="#94a3b8" />
            <circle cx={xOf(x)} cy={yOf(gelu(value))} r="1.7" fill="#ea580c" />
          </g>
        );
      })}
    </svg>
  );
}

function argmax(values: number[]): number {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) if (values[index] > values[best]) best = index;
  return best;
}

function QueryCompare({
  pixels,
  imageSize,
  patchSize,
  token,
  content,
  learned,
  fromToken,
  fromValue,
}: {
  pixels: Float32Array;
  imageSize: number;
  patchSize: number;
  token: number;
  content: number[];
  learned: number[];
  fromToken: number[];
  fromValue: number[];
}) {
  const names = tokenNames(imageSize / patchSize);
  const peak = Math.max(...content, ...learned, 1e-6);
  const shared = Math.max(...[...fromToken, ...fromValue].map((value) => Math.abs(value)), 1e-6);
  return (
    <>
      <div className="grid grid-cols-2 gap-3">
        <Choice label="match the token" value={`${(Math.max(...content) * 100).toFixed(0)}% on ${names[argmax(content)].name}`}>
          <Picture pixels={pixels} imageSize={imageSize} patch={patchSize} overlay={{ weights: content.slice(1), selected: token, hover: null, scale: "unit" }} grid={false} display={120} />
          <Bars values={content} peak={peak} />
        </Choice>
        <Choice label="learned query" value={`${(Math.max(...learned) * 100).toFixed(0)}% on ${names[argmax(learned)].name}`} live>
          <Picture pixels={pixels} imageSize={imageSize} patch={patchSize} overlay={{ weights: learned.slice(1), selected: token, hover: null, scale: "unit" }} grid={false} display={120} />
          <Bars values={learned} peak={peak} />
        </Choice>
      </div>
      <div className="mt-3 space-y-1">
        <Strip label="mix h" values={fromToken} peak={shared} note="" />
        <Strip label="mix V" values={fromValue} peak={shared} note="" live />
      </div>
      <p className="mt-2 text-[11px] leading-4 text-slate-500">
        The value mixture lines up with mixing the token itself at {cosine(fromToken, fromValue).toFixed(2)}.
      </p>
    </>
  );
}

function Lengths({ before, after }: { before: number[]; after: number[] }) {
  const peak = Math.max(...before, ...after, 1e-6);
  const span = (values: number[]) => `${Math.min(...values).toFixed(2)}–${Math.max(...values).toFixed(2)}`;
  return (
    <>
      <LengthBars label="h" values={before} peak={peak} note={span(before)} />
      <LengthBars label="LayerNorm" values={after} peak={peak} note={span(after)} live />
      <p className="mt-2 text-[11px] leading-4 text-slate-500">
        The longest token is {(Math.max(...before) / Math.max(Math.min(...before), 1e-6)).toFixed(0)}× the shortest. After the norm the lengths sit together, so a long patch cannot win the dot product by size.
      </p>
    </>
  );
}

function LengthBars({ label, values, peak, note, live }: { label: string; values: number[]; peak: number; note: string; live?: boolean }) {
  return (
    <div className="mt-2">
      <div className="flex items-baseline justify-between">
        <span className={`font-mono text-[11px] ${live ? "font-semibold text-indigo-700" : "text-slate-500"}`}>{label}</span>
        <span className="font-mono text-[10px] text-slate-400">{note}</span>
      </div>
      <div className="mt-1 flex h-14 items-end gap-px">
        {values.map((value, index) => (
          <span key={index} title={value.toFixed(2)} className="min-w-0 flex-1 rounded-sm bg-indigo-500" style={{ height: `${Math.max(4, (value / peak) * 100)}%` }} />
        ))}
      </div>
    </div>
  );
}

function MaskCompare({
  model,
  pixels,
  imageSize,
  patchSize,
  full,
  causal,
  block,
  head,
}: {
  model: ModelFile;
  pixels: Float32Array;
  imageSize: number;
  patchSize: number;
  full: Trace;
  causal: Trace;
  block: number;
  head: number;
}) {
  const fullProbs = probabilities(full.logits, 1);
  const causalProbs = probabilities(causal.logits, 1);
  const fullTop = argmax(fullProbs);
  const causalTop = argmax(causalProbs);
  return (
    <>
    <div className="grid grid-cols-2 gap-3">
      <Choice label="causal mask" value={`${model.config.classes[causalTop]} ${(causalProbs[causalTop] * 100).toFixed(0)}%`}>
        <Picture
          pixels={pixels}
          imageSize={imageSize}
          patch={patchSize}
          overlay={{ weights: causal.blocks[block].weights[head][0].slice(1), selected: 0, hover: null, scale: "unit" }}
          grid={false}
          display={120}
        />
      </Choice>
      <Choice label="every patch" value={`${model.config.classes[fullTop]} ${(fullProbs[fullTop] * 100).toFixed(0)}%`} live>
        <Picture
          pixels={pixels}
          imageSize={imageSize}
          patch={patchSize}
          overlay={{ weights: full.blocks[block].weights[head][0].slice(1), selected: 0, hover: null, scale: "unit" }}
          grid={false}
          display={120}
        />
      </Choice>
    </div>
    <p className="mt-2 text-[11px] leading-4 text-slate-500">
      The class token is first in line. A causal mask lets it read only itself, and {model.config.classes[fullTop]} falls to {(causalProbs[fullTop] * 100).toFixed(0)}%.
    </p>
  </>
);
}

function PixelMatrix() {
  const lines = Array.from({ length: 32 }, (_, index) => index * 2);
  return (
    <div>
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-slate-400">1024 × 1024</div>
      <svg viewBox="0 0 64 64" className="h-16 w-16 rounded bg-indigo-50">
        {lines.map((position) => (
          <line key={`v${position}`} x1={position} y1="0" x2={position} y2="64" stroke="#4338ca" strokeWidth="0.7" />
        ))}
        {lines.map((position) => (
          <line key={`h${position}`} x1="0" y1={position} x2="64" y2={position} stroke="#4338ca" strokeWidth="0.7" />
        ))}
      </svg>
    </div>
  );
}
