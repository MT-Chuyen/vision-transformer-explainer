import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { popovers, type PopoverId, type StepId } from "../content";
import { gelu, tokenNames, type ModelFile, type Trace } from "../model/forward";
import { EffectMap, Heatmap, Picture, PopCard, ProbabilityBars, TokenStack, meanAbs, vectorStyle } from "./visual";

type Box = { x: number; y: number; w: number; h: number };

const FLOW = [
  { id: "embedding", color: "#c084fc" },
  { id: "qkv", color: "#60a5fa" },
  { id: "attention", color: "#818cf8" },
  { id: "mlp", color: "#fb923c" },
  { id: "more", color: "#94a3b8" },
  { id: "softmax", color: "#6366f1" },
  { id: "effect", color: "#e11d48" },
] as const;

export function Stage({
  model,
  pixels,
  trace,
  block,
  setBlock,
  head,
  setHead,
  token,
  setToken,
  hover,
  setHover,
  mask,
  setMask,
  dropout,
  setDropout,
  onRedrawDropout,
  expanded,
  setExpanded,
  popover,
  openPopover,
  closePopover,
  focus,
  probs,
  pool,
  sampled,
  playing,
  setPlaying,
  sourcePixels,
  drops,
  hidden,
  onTogglePatch,
  onClearPatches,
  watchedClass,
  watchedProb,
  removedProb,
}: {
  model: ModelFile;
  pixels: Float32Array;
  trace: Trace;
  block: number;
  setBlock: (value: number) => void;
  head: number;
  setHead: (value: number) => void;
  token: number;
  setToken: (value: number) => void;
  hover: number | null;
  setHover: (value: number | null) => void;
  mask: "full" | "causal";
  setMask: (value: "full" | "causal") => void;
  dropout: number;
  setDropout: (value: number) => void;
  onRedrawDropout: () => void;
  expanded: StepId | null;
  setExpanded: (value: StepId | null) => void;
  popover: { id: PopoverId; x: number; y: number } | null;
  openPopover: (id: PopoverId, event: MouseEvent) => void;
  closePopover: () => void;
  focus: string | null;
  probs: number[];
  pool: boolean[];
  sampled: number;
  playing: boolean;
  setPlaying: (value: boolean) => void;
  sourcePixels: Float32Array;
  drops: number[];
  hidden: number[];
  onTogglePatch: (patchIndex: number) => void;
  onClearPatches: () => void;
  watchedClass: string;
  watchedProb: number;
  removedProb: number | null;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [boxes, setBoxes] = useState<Record<string, Box>>({});
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });
  const [hoverStep, setHoverStep] = useState<string | null>(null);
  const { dim, heads, depth, classes, imageSize, patchSize } = model.config;
  const names = tokenNames(imageSize / patchSize);
  const labels = names.map((_, index) => (index === 0 ? "C" : String(index)));
  const current = trace.blocks[block];
  const best = probs.indexOf(Math.max(...probs));
  const activeFlow = hoverStep ?? focus;

  const overlay = useMemo(
    () => ({
      weights: current.weights[head][token].slice(1),
      selected: token,
      hover,
    }),
    [current, head, token, hover],
  );

  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    const measure = () => {
      const base = root.getBoundingClientRect();
      const next: Record<string, Box> = {};
      root.querySelectorAll<HTMLElement>("[data-flow]").forEach((node) => {
        const id = node.dataset.flow;
        if (!id) return;
        const rect = node.getBoundingClientRect();
        next[id] = {
          x: rect.left - base.left + root.scrollLeft,
          y: rect.top - base.top,
          w: rect.width,
          h: rect.height,
        };
      });
      setBoxes(next);
      setCanvasSize({ w: root.scrollWidth, h: root.scrollHeight });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    root.addEventListener("scroll", measure);
    return () => {
      observer.disconnect();
      root.removeEventListener("scroll", measure);
    };
  }, [expanded, block, head, mask, pixels]);

  const [sweeping, setSweeping] = useState(false);
  const keyToken = hover ?? token;
  const score = current.scores[head][token][keyToken];
  const weight = current.weights[head][token][keyToken];

  useEffect(() => {
    if (!sweeping) return;
    const id = window.setInterval(() => {
      setToken(token >= names.length - 1 ? 0 : token + 1);
    }, 480);
    return () => window.clearInterval(id);
  }, [sweeping, token, names.length, setToken]);

  function follow(next: number) {
    setSweeping(false);
    setToken(next);
  }

  return (
    <div ref={scroller} className="relative overflow-x-auto">
      <svg className="pointer-events-none absolute left-0 top-0" width={canvasSize.w} height={canvasSize.h}>
        {FLOW.slice(0, -1).map((step, index) => {
          const from = boxes[step.id];
          const to = boxes[FLOW[index + 1].id];
          if (!from || !to) return null;
          const x1 = from.x + from.w;
          const y1 = from.y + from.h * 0.42;
          const x2 = to.x;
          const y2 = to.y + to.h * 0.42;
          const mid = (x1 + x2) / 2;
          const lit = !activeFlow || activeFlow === "block"
            ? activeFlow === "block"
              ? ["qkv", "attention", "mlp"].includes(step.id) || ["qkv", "attention", "mlp"].includes(FLOW[index + 1].id)
              : true
            : activeFlow === step.id || activeFlow === FLOW[index + 1].id;
          return (
            <path
              key={step.id}
              d={`M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`}
              stroke={step.color}
              strokeWidth={22}
              fill="none"
              className={lit ? "flow-run" : undefined}
              opacity={lit ? 0.55 : 0.08}
            />
          );
        })}
      </svg>

      <div className="relative flex items-stretch gap-8 px-6 py-8" onMouseLeave={() => setHoverStep(null)}>
        <Frame
          id="embedding"
          title="Patch embedding"
          shape="(17, 32)"
          focus={focus}
          dim={expanded !== null && expanded !== "embedding"}
          wide={expanded === "embedding"}
          onToggle={() => setExpanded(expanded === "embedding" ? null : "embedding")}
          onEnter={() => setHoverStep("embedding")}
        >
          <Picture
            pixels={pixels}
            imageSize={imageSize}
            patch={patchSize}
            overlay={overlay}
            grid
            display={expanded === "embedding" ? 220 : 148}
            onPick={follow}
            onMove={follow}
          />
          <p data-testid="probe" className="mt-2 text-[11px] leading-4 text-slate-500">
            {names[token].name === "CLS" ? "CLS is selected. It is not a patch." : `Patch ${names[token].name} is selected.`} Move across the picture and the rest of the diagram follows. Darker violet means a larger share of this head. CLS gets {(current.weights[head][token][0] * 100).toFixed(1)}%.
          </p>
          <button type="button" className="mt-2 text-[11px] font-semibold text-indigo-600" onClick={() => setSweeping((value) => !value)}>
            {sweeping ? "Stop sweep" : "Sweep patches"}
          </button>
          <div className="mt-3">
            <TokenStack rows={trace.afterPos} selected={token} onSelect={setToken} tone="#c084fc" labels={labels} />
          </div>
          {expanded === "embedding" ? (
            <div className="mt-3 space-y-2">
              <Mini label="patch only, no position" rows={withClass(trace.patchTokens, model.cls)} tone="#a855f7" labels={labels} token={token} onSelect={setToken} />
              <Mini label="position that gets added" rows={rowsFrom(model.pos, dim)} tone="#f59e0b" labels={labels} token={token} onSelect={setToken} />
            </div>
          ) : null}
          <div className="mt-2 flex gap-3">
            <TextButton onClick={(event) => openPopover("patch", event)}>Patch matrix</TextButton>
            <TextButton onClick={(event) => openPopover("position", event)}>Position matrix</TextButton>
          </div>
        </Frame>

        <div
          data-focus="block"
          className={`flex shrink-0 flex-col rounded-2xl border border-dashed px-4 py-3 ${focus === "block" ? "border-amber-400 ring-2 ring-amber-400 ring-offset-2" : "border-slate-300"}`}
        >
          <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
              Transformer block {block + 1} / {depth}
            </h2>
            <TextButton onClick={(event) => openPopover("residual1", event)}>Residual</TextButton>
            <TextButton onClick={(event) => openPopover("ln1", event)}>Layer norm</TextButton>
            <TextButton onClick={(event) => openPopover("dropout", event)}>Dropout</TextButton>
            <label className="flex items-center gap-2 text-[11px] text-slate-500">
              p={dropout.toFixed(2)}
              <input type="range" min={0} max={0.9} step={0.05} value={dropout} onChange={(event) => setDropout(Number(event.target.value))} />
            </label>
            {dropout > 0 ? (
              <TextButton onClick={() => onRedrawDropout()}>New mask</TextButton>
            ) : null}
          </div>
          <div key={block} className="block-swap flex items-stretch gap-6">
            <Frame
              id="qkv"
              title="Q · K · V"
              shape="(17, 96) → 4 × (17, 8)"
              focus={focus}
              dim={expanded !== null && expanded !== "qkv"}
              wide={expanded === "qkv"}
              onToggle={() => setExpanded(expanded === "qkv" ? null : "qkv")}
              onEnter={() => setHoverStep("qkv")}
            >
              <HeadSwitch heads={heads} head={head} onHead={setHead} />
              <div className="mt-3 space-y-2">
                <Mini label="Query" rows={current.query[head]} tone="#3b82f6" labels={labels} token={token} onSelect={setToken} />
                <Mini label="Key" rows={current.key[head]} tone="#ef4444" labels={labels} token={token} onSelect={setToken} />
                <Mini label="Value" rows={current.value[head]} tone="#22c55e" labels={labels} token={token} onSelect={setToken} />
              </div>
              <div className="mt-2">
                <TextButton onClick={(event) => openPopover("qkv", event)}>QKV matrix</TextButton>
              </div>
            </Frame>

            <Frame
              id="attention"
              title="Multi-head self-attention"
              shape="(17, 17) × 4"
              focus={focus}
              dim={expanded !== null && expanded !== "attention"}
              wide={expanded === "attention"}
              onToggle={() => setExpanded(expanded === "attention" ? null : "attention")}
              onEnter={() => setHoverStep("attention")}
            >
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <HeadSwitch heads={heads} head={head} onHead={setHead} />
                <div className="flex gap-2 text-[11px] text-slate-500">
                  <label className="flex items-center gap-1">
                    <input type="radio" checked={mask === "full"} onChange={() => setMask("full")} />
                    Full
                  </label>
                  <label className="flex items-center gap-1">
                    <input type="radio" checked={mask === "causal"} onChange={() => setMask("causal")} />
                    Causal mask
                  </label>
                </div>
              </div>
              <div className="mb-3 flex flex-wrap gap-2">
                {current.weights.map((headsWeights, index) => (
                  <button key={index} type="button" onClick={() => setHead(index)} className="text-left">
                    <Picture
                      pixels={pixels}
                      imageSize={imageSize}
                      patch={patchSize}
                      overlay={{ weights: headsWeights[token].slice(1), selected: token, hover: null }}
                      grid={false}
                      display={72}
                      onPick={(next) => {
                        setHead(index);
                        setToken(next);
                      }}
                    />
                    <span className={`mt-1 block text-center text-[10px] ${head === index ? "font-semibold text-slate-900" : "text-slate-400"}`}>
                      {index + 1}
                    </span>
                  </button>
                ))}
              </div>
              <Heatmap
                rows={current.weights[head]}
                rowLabels={expanded === "attention" ? labels : undefined}
                colLabels={expanded === "attention" ? labels : undefined}
                selectedRow={token}
                selectedCol={hover}
                onHover={(row, col) => {
                  setHover(col);
                  if (row >= 0) follow(row);
                }}
                onPick={follow}
                cell={expanded === "attention" ? 20 : 9}
                domain="unit"
              />
              <p className="mt-2 max-w-[420px] text-[11px] leading-4 text-slate-500">
                {names[token].name} reads {names[keyToken].name} at {(weight * 100).toFixed(1)}%. Score before softmax: {Number.isFinite(score) ? score.toFixed(2) : "masked"}.
              </p>
              {expanded === "attention" ? (
                <div className="mt-3">
                  <div className="mb-1 text-[11px] text-slate-400">Head outputs of the selected token, concatenated</div>
                  <div className="flex gap-1">
                    {current.headOut.map((group, index) => (
                      <span key={index} className="h-16 w-3 rounded-sm" style={{ background: vectorStyle(group[token], ["#3b82f6", "#ef4444", "#22c55e", "#a855f7"][index]) }} />
                    ))}
                  </div>
                </div>
              ) : null}
              <div className="mt-2 flex flex-wrap gap-3">
                <TextButton onClick={(event) => openPopover("attention", event)}>Attention matrix</TextButton>
                <TextButton onClick={(event) => openPopover("concat", event)}>Concatenate</TextButton>
                <TextButton onClick={(event) => openPopover("proj", event)}>Projection</TextButton>
              </div>
            </Frame>

            <Frame
              id="mlp"
              title="MLP"
              shape="(17, 128) → (17, 32)"
              focus={focus}
              dim={expanded !== null && expanded !== "mlp"}
              wide={expanded === "mlp"}
              onToggle={() => setExpanded(expanded === "mlp" ? null : "mlp")}
              onEnter={() => setHoverStep("mlp")}
            >
              <div className="space-y-2">
                <Mini label="up, 128" rows={current.mlpUp} tone="#fdba74" labels={labels} token={token} onSelect={setToken} />
                <Mini label="GELU" rows={current.activated} tone="#f97316" labels={labels} token={token} onSelect={setToken} />
                <Mini label="down, 32" rows={current.mlpDown} tone="#ea580c" labels={labels} token={token} onSelect={setToken} />
              </div>
              <div className="mt-2 flex flex-wrap gap-3">
                <TextButton onClick={(event) => openPopover("mlp-up", event)}>Up matrix</TextButton>
                <TextButton onClick={(event) => openPopover("gelu", event)}>GELU</TextButton>
                <TextButton onClick={(event) => openPopover("mlp-down", event)}>Down matrix</TextButton>
                <TextButton onClick={(event) => openPopover("residual2", event)}>Residual</TextButton>
              </div>
            </Frame>
          </div>
        </div>

        <Frame
          id="more"
          title="Subsequent blocks"
          shape={`${Math.max(depth - block - 1, 0)} left`}
          focus={focus}
          dim={false}
          wide={false}
          onToggle={() => setPlaying(!playing)}
          onEnter={() => setHoverStep("more")}
        >
          <div className="flex h-full min-w-[168px] flex-col items-start justify-center gap-3 py-2">
            <div className="grid grid-cols-2 gap-1.5">
              {trace.blocks.map((item, index) => (
                <button key={index} type="button" onClick={() => setBlock(index)} className="text-left">
                  <Picture
                    pixels={pixels}
                    imageSize={imageSize}
                    patch={patchSize}
                    overlay={{ weights: item.weights[head][token].slice(1), selected: token, hover: null }}
                    grid={false}
                    display={76}
                  />
                  <span className={`mt-0.5 block text-center text-[10px] ${block === index ? "font-semibold text-slate-900" : "text-slate-400"}`}>
                    {index + 1}
                  </span>
                </button>
              ))}
            </div>
            <p className="font-serif text-4xl leading-none text-slate-800">{Math.max(depth - block - 1, 0)}</p>
            <p className="max-w-[160px] text-sm leading-5 text-slate-500">
              {block === depth - 1 ? "This is the last block. The class token goes to the linear layer." : "more identical blocks, each with its own weights."}
            </p>
            <div className="flex gap-2">
              <button type="button" className="rounded-md border border-slate-300 px-2 py-1 text-xs" onClick={() => setBlock(Math.max(0, block - 1))}>
                Previous
              </button>
              <button type="button" className="rounded-md border border-slate-300 px-2 py-1 text-xs" onClick={() => setBlock(Math.min(depth - 1, block + 1))}>
                Next
              </button>
            </div>
            <button type="button" className="text-xs font-semibold text-indigo-600" onClick={() => setPlaying(!playing)}>
              {playing ? "Stop" : "Play blocks"}
            </button>
          </div>
        </Frame>

        <Frame
          id="softmax"
          title="Linear + softmax"
          shape="(9,)"
          focus={focus}
          dim={expanded !== null && expanded !== "softmax"}
          wide={expanded === "softmax"}
          onToggle={() => setExpanded(expanded === "softmax" ? null : "softmax")}
          onEnter={() => setHoverStep("softmax")}
        >
          <ProbabilityBars labels={classes} probs={probs} pool={pool} sampled={sampled} best={best} />
          <ClassGrid classes={classes} probs={probs} best={best} />
          <p className="mt-2 max-w-[240px] text-[11px] leading-4 text-slate-500">
            Indigo bars are in the top-k or top-p pool. The amber outline is the drawn class. Temperature leaves the blocks as they are and only rescales these bars.
          </p>
          {expanded === "softmax" ? (
            <ul className="mt-2 space-y-0.5 font-mono text-[10px] text-slate-500">
              {classes.map((name, index) => (
                <li key={name}>
                  {name} logit {trace.logits[index].toFixed(2)}
                </li>
              ))}
            </ul>
          ) : null}
          <div className="mt-2 flex gap-3">
            <TextButton onClick={(event) => openPopover("classifier", event)}>Classifier matrix</TextButton>
            <TextButton onClick={(event) => openPopover("softmax", event)}>Softmax</TextButton>
          </div>
        </Frame>

        <Frame
          id="effect"
          title="Patch removed"
          shape="16 forwards"
          focus={focus}
          dim={expanded !== null && expanded !== "effect"}
          wide={expanded === "effect"}
          onToggle={() => setExpanded(expanded === "effect" ? null : "effect")}
          onEnter={() => setHoverStep("effect")}
        >
          <EffectMap
            pixels={sourcePixels}
            imageSize={imageSize}
            patch={patchSize}
            drops={drops}
            hidden={hidden}
            display={expanded === "effect" ? 220 : 168}
            onToggle={onTogglePatch}
          />
          <p className="mt-2 max-w-[220px] text-[11px] leading-4 text-slate-500">
            {removedProb === null
              ? `${watchedClass} is ${(watchedProb * 100).toFixed(0)}%. Red is how far that falls if the patch is blacked out.`
              : `${watchedClass} moves from ${(watchedProb * 100).toFixed(0)}% to ${(removedProb * 100).toFixed(0)}%.`}
          </p>
          {hidden.length ? (
            <button type="button" className="mt-2 text-[11px] font-semibold text-indigo-600" onClick={onClearPatches}>
              Restore patches
            </button>
          ) : null}
        </Frame>
      </div>

      {popover ? (
        <>
          <button type="button" aria-label="Close details" className="fixed inset-0 z-30 cursor-default" onClick={closePopover} />
          <PopCard
            x={popover.x}
            y={popover.y}
            title={popovers[popover.id].title}
            formula={popovers[popover.id].formula}
            body={popovers[popover.id].body}
            note={liveNote(popover.id, model, trace, block, head, token)}
            onClose={closePopover}
          >
            <PopoverBody id={popover.id} model={model} trace={trace} block={block} head={head} token={token} />
          </PopCard>
        </>
      ) : null}
    </div>
  );
}

function Frame({
  id,
  title,
  shape,
  focus,
  dim,
  wide,
  onToggle,
  onEnter,
  children,
}: {
  id: string;
  title: string;
  shape: string;
  focus: string | null;
  dim: boolean;
  wide: boolean;
  onToggle: () => void;
  onEnter: () => void;
  children: ReactNode;
}) {
  return (
    <section
      data-flow={id}
      data-focus={id}
      onMouseEnter={onEnter}
      className={`flex shrink-0 flex-col ${wide ? "min-w-[520px]" : ""} ${dim ? "opacity-40" : ""} ${focus === id ? "rounded-xl ring-2 ring-amber-400 ring-offset-2" : ""}`}
    >
      <button type="button" onClick={onToggle} className="mb-2 text-left">
        <span className="block text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400 hover:text-slate-700">{title}</span>
        <span className="font-mono text-[10px] text-slate-400">{shape}</span>
      </button>
      <div className="rounded-xl border border-slate-200 bg-white/95 p-3 shadow-sm">{children}</div>
    </section>
  );
}

function ClassGrid({ classes, probs, best }: { classes: string[]; probs: number[]; best: number }) {
  const paints = [
    [225, 29, 72],
    [22, 163, 74],
    [37, 99, 235],
  ];
  const marks = ["●", "■", "▲"];
  return (
    <div className="mt-3 grid w-[240px] grid-cols-3 gap-1">
      {probs.map((prob, index) => {
        const [red, green, blue] = paints[Math.floor(index / 3)];
        return (
          <div
            key={classes[index]}
            title={classes[index]}
            className={`flex h-9 items-center justify-center rounded-md text-[11px] font-semibold ${index === best ? "ring-2 ring-amber-400" : ""}`}
            style={{ background: `rgba(${red}, ${green}, ${blue}, ${0.18 + prob * 0.82})`, color: prob > 0.45 ? "#fff" : "#1e293b" }}
          >
            {marks[index % 3]} {(prob * 100).toFixed(0)}
          </div>
        );
      })}
    </div>
  );
}

function HeadSwitch({ heads, head, onHead }: { heads: number; head: number; onHead: (value: number) => void }) {
  return (
    <div className="flex gap-1">
      {Array.from({ length: heads }, (_, index) => (
        <button
          key={index}
          type="button"
          onClick={() => onHead(index)}
          className={`rounded-md px-2 py-0.5 text-[11px] font-semibold ${head === index ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-600"}`}
        >
          Head {index + 1}
        </button>
      ))}
    </div>
  );
}

function Mini({
  label,
  rows,
  tone,
  labels,
  token,
  onSelect,
}: {
  label: string;
  rows: number[][];
  tone: string;
  labels: string[];
  token: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
      <TokenStack rows={rows} selected={token} onSelect={onSelect} tone={tone} labels={labels} />
    </div>
  );
}

function TextButton({ children, onClick }: { children: ReactNode; onClick: (event: MouseEvent) => void }) {
  return (
    <button type="button" onClick={onClick} className="text-[11px] font-semibold text-indigo-600 hover:text-indigo-800">
      {children}
    </button>
  );
}

function withClass(patches: number[][], cls: number[]): number[][] {
  return [cls.slice(), ...patches];
}

function rowsFrom(flat: number[], width: number): number[][] {
  const rows: number[][] = [];
  for (let index = 0; index < flat.length; index += width) rows.push(flat.slice(index, index + width));
  return rows;
}

function matrix(flat: number[], columns: number): number[][] {
  return rowsFrom(flat, columns);
}

function liveNote(id: PopoverId, model: ModelFile, trace: Trace, block: number, head: number, token: number): string | undefined {
  const current = trace.blocks[block];
  const skip = block === 0 ? trace.afterPos : trace.blocks[block - 1].resid2;
  if (id === "residual1") {
    return `Mean absolute value of the skip is ${meanAbs(skip).toFixed(3)}. Mean absolute value of the attention add is ${meanAbs(current.droppedAttn).toFixed(3)}.`;
  }
  if (id === "residual2") {
    return `Mean absolute value of the stream is ${meanAbs(current.resid1).toFixed(3)}. Mean absolute value of the MLP add is ${meanAbs(current.droppedMlp).toFixed(3)}.`;
  }
  if (id === "attention") {
    const row = current.weights[head][token];
    const top = row.indexOf(Math.max(...row));
    return `Head ${head + 1}, token ${token}: the strongest key is token ${top} at ${(row[top] * 100).toFixed(1)}%.`;
  }
  if (id === "classifier") {
    return `Class token after the final norm, first four coordinates: ${trace.classVector.slice(0, 4).map((value) => value.toFixed(2)).join(", ")}.`;
  }
  if (id === "gelu") {
    const before = current.mlpUp[token][0];
    return `Selected token, first hidden unit: ${before.toFixed(3)} → GELU ${gelu(before).toFixed(3)}. Width ${model.config.dim * model.config.mlpRatio}.`;
  }
  return undefined;
}

function PopoverBody({
  id,
  model,
  trace,
  block,
  head,
  token,
}: {
  id: PopoverId;
  model: ModelFile;
  trace: Trace;
  block: number;
  head: number;
  token: number;
}) {
  const { dim, mlpRatio } = model.config;
  const current = trace.blocks[block];
  const weights = model.blocks[block];
  if (id === "patch") return <Heatmap rows={matrix(model.patchW, 3 * 8 * 8)} cell={3} domain="signed" />;
  if (id === "position") return <Heatmap rows={matrix(model.pos, dim)} cell={6} domain="signed" />;
  if (id === "ln1") return <Heatmap rows={current.ln1} cell={6} domain="signed" />;
  if (id === "qkv") return <Heatmap rows={matrix(weights.qkv.w, dim)} cell={4} domain="signed" />;
  if (id === "attention") return <Heatmap rows={current.weights[head]} cell={14} domain="unit" />;
  if (id === "concat") return <Heatmap rows={current.concat} cell={6} domain="signed" />;
  if (id === "proj") return <Heatmap rows={matrix(weights.proj.w, dim)} cell={6} domain="signed" />;
  if (id === "residual1") return <Heatmap rows={current.resid1} cell={6} domain="signed" />;
  if (id === "ln2") return <Heatmap rows={current.ln2} cell={6} domain="signed" />;
  if (id === "mlp-up") return <Heatmap rows={matrix(weights.fc1.w, dim)} cell={4} domain="signed" />;
  if (id === "mlp-down") return <Heatmap rows={matrix(weights.fc2.w, dim * mlpRatio)} cell={3} domain="signed" />;
  if (id === "residual2") return <Heatmap rows={current.resid2} cell={6} domain="signed" />;
  if (id === "classifier") return <Heatmap rows={matrix(model.head.w, dim)} cell={8} domain="signed" />;
  if (id === "softmax") return <p className="text-[11px] text-slate-500">The bars in the last column are this softmax. Logit {token === 0 ? "of the class token" : "source"} is fixed until temperature changes.</p>;
  if (id === "gelu") return <GeluChart />;
  if (id === "dropout") return <p className="text-[11px] text-slate-500">The slider on the block applies this mask to the attention projection and the MLP output.</p>;
  return null;
}

function GeluChart() {
  const points: string[] = [];
  for (let index = 0; index <= 48; index += 1) {
    const x = -3 + (6 * index) / 48;
    const y = gelu(x);
    points.push(`${(index / 48) * 160},${40 - ((y + 0.2) / 3.2) * 40}`);
  }
  return (
    <svg viewBox="0 0 160 48" className="h-16 w-full">
      <line x1="0" y1="34" x2="160" y2="34" stroke="#e2e8f0" />
      <polyline fill="none" stroke="#ea580c" strokeWidth="2" points={points.join(" ")} />
    </svg>
  );
}
