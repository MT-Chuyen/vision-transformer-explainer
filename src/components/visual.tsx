import { useEffect, useRef, type MouseEvent, type ReactNode } from "react";

export function meanAbs(rows: number[][]): number {
  let total = 0;
  let count = 0;
  for (const row of rows) {
    for (const value of row) {
      total += Math.abs(value);
      count += 1;
    }
  }
  return count === 0 ? 0 : total / count;
}

export function vectorStyle(values: number[], positive: string): string {
  const peak = Math.max(...values.map((value) => Math.abs(value)), 1e-6);
  const step = Math.max(1, Math.ceil(values.length / 28));
  const sampled: number[] = [];
  for (let index = 0; index < values.length; index += step) sampled.push(values[index]);
  const stops = sampled.map((value, index) => {
    const amount = 0.18 + (Math.abs(value) / peak) * 0.82;
    const color = value >= 0 ? positive : "#dc2626";
    const start = (index / sampled.length) * 100;
    const end = ((index + 1) / sampled.length) * 100;
    return `${withAlpha(color, amount)} ${start}% ${end}%`;
  });
  return `linear-gradient(to bottom, ${stops.join(", ")})`;
}

function withAlpha(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  const red = parseInt(value.slice(0, 2), 16);
  const green = parseInt(value.slice(2, 4), 16);
  const blue = parseInt(value.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

export function TokenStack({
  rows,
  selected,
  onSelect,
  tone,
  labels,
}: {
  rows: number[][];
  selected: number;
  onSelect: (index: number) => void;
  tone: string;
  labels: string[];
}) {
  return (
    <div className="flex items-end gap-[3px]">
      {rows.map((row, index) => (
        <button
          key={labels[index] ?? index}
          type="button"
          title={labels[index]}
          onClick={() => onSelect(index)}
          className={`flex w-[16px] flex-col items-center gap-1 ${selected === index ? "opacity-100" : "opacity-80 hover:opacity-100"}`}
        >
          <span
            className={`block w-[10px] rounded-sm transition-transform duration-200 ${selected === index ? "scale-110 outline outline-2 outline-offset-1 outline-slate-900" : ""}`}
            style={{ height: 112, background: vectorStyle(row, tone) }}
          />
          <span className="text-[8px] leading-none text-slate-400">{index === 0 ? "C" : index}</span>
        </button>
      ))}
    </div>
  );
}

export function Heatmap({
  rows,
  rowLabels,
  colLabels,
  selectedRow,
  selectedCol,
  onHover,
  onPick,
  cell = 16,
  domain = "auto",
}: {
  rows: number[][];
  rowLabels?: string[];
  colLabels?: string[];
  selectedRow?: number;
  selectedCol?: number | null;
  onHover?: (row: number, col: number | null) => void;
  onPick?: (row: number) => void;
  cell?: number;
  domain?: "auto" | "unit" | "signed";
}) {
  let low = 0;
  let high = 1;
  if (domain === "auto" || domain === "signed") {
    low = Infinity;
    high = -Infinity;
    for (const row of rows) {
      for (const value of row) {
        if (!Number.isFinite(value)) continue;
        low = Math.min(low, value);
        high = Math.max(high, value);
      }
    }
    if (!Number.isFinite(low) || high === low) {
      low = 0;
      high = 1;
    }
  }
  const width = rows[0]?.length ?? 0;
  const cells: ReactNode[] = [];
  if (colLabels) {
    if (rowLabels) cells.push(<span key="corner" />);
    colLabels.forEach((label, index) => {
      cells.push(
        <span key={`c-${index}`} className="truncate text-center text-[8px] text-slate-400">
          {label}
        </span>,
      );
    });
  }
  rows.forEach((row, rowIndex) => {
    if (rowLabels) {
      cells.push(
        <span key={`l-${rowIndex}`} className="self-center text-right text-[8px] text-slate-400">
          {rowLabels[rowIndex]}
        </span>,
      );
    }
    row.forEach((value, colIndex) => {
      const finite = Number.isFinite(value);
      let color = "#e5e7eb";
      if (finite && domain === "signed") {
        const peak = Math.max(Math.abs(low), Math.abs(high), 1e-6);
        const amount = Math.abs(value) / peak;
        color = value >= 0 ? `rgba(79, 70, 229, ${0.12 + amount * 0.88})` : `rgba(220, 38, 38, ${0.12 + amount * 0.88})`;
      } else if (finite) {
        const span = domain === "unit" ? 1 : high - low;
        const start = domain === "unit" ? 0 : low;
        const amount = Math.min(1, Math.max(0, (value - start) / span));
        color = `rgb(${Math.round(241 - amount * 160)} ${Math.round(245 - amount * 190)} ${Math.round(255 - amount * 40)})`;
      }
      const marked = selectedRow === rowIndex || selectedCol === colIndex;
      cells.push(
        <button
          key={`${rowIndex}-${colIndex}`}
          type="button"
          title={finite ? value.toFixed(3) : "masked"}
          onMouseEnter={() => onHover?.(rowIndex, colIndex)}
          onClick={() => onPick?.(rowIndex)}
          className={`relative ${marked ? "ring-1 ring-slate-900" : ""}`}
          style={{ width: cell, height: cell, background: color }}
        />,
      );
    });
  });
  return (
    <div className="overflow-auto" onMouseLeave={() => onHover?.(-1, null)}>
      <div
        className="grid"
        style={{
          gridTemplateColumns: `${rowLabels ? "28px " : ""}repeat(${width}, ${cell}px)`,
          gap: 1,
        }}
      >
        {cells}
      </div>
    </div>
  );
}

export function ProbabilityBars({
  labels,
  probs,
  pool,
  sampled,
  best,
}: {
  labels: string[];
  probs: number[];
  pool: boolean[];
  sampled: number;
  best: number;
}) {
  return (
    <div className="flex w-[240px] flex-col gap-1.5">
      {labels.map((label, index) => (
        <div key={label} className="grid grid-cols-[88px_1fr_36px] items-center gap-1 text-[11px]">
          <span className={index === best ? "font-semibold text-slate-900" : "text-slate-500"}>{label}</span>
          <span className="h-2 overflow-hidden rounded-full bg-slate-100">
            <span
              className={`block h-full rounded-full transition-[width] duration-300 ${pool[index] ? "bg-indigo-500" : "bg-slate-300"} ${sampled === index ? "outline outline-2 outline-offset-1 outline-amber-500" : ""}`}
              style={{ width: `${Math.max(2, probs[index] * 100)}%` }}
            />
          </span>
          <span className="text-right tabular-nums text-slate-500">{(probs[index] * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

type Overlay = {
  weights: number[] | null;
  selected: number;
  hover: number | null;
  scale?: "peak" | "unit";
};

export function drawScene(
  canvas: HTMLCanvasElement,
  pixels: ArrayLike<number>,
  imageSize: number,
  patch: number,
  overlay: Overlay | null,
  grid: boolean,
) {
  const context = canvas.getContext("2d");
  if (!context) return;
  const off = document.createElement("canvas");
  off.width = imageSize;
  off.height = imageSize;
  const source = off.getContext("2d");
  if (!source) return;
  const image = source.createImageData(imageSize, imageSize);
  const plane = imageSize * imageSize;
  for (let y = 0; y < imageSize; y += 1) {
    for (let x = 0; x < imageSize; x += 1) {
      const offset = (y * imageSize + x) * 4;
      image.data[offset] = Math.round(pixels[y * imageSize + x] * 255);
      image.data[offset + 1] = Math.round(pixels[plane + y * imageSize + x] * 255);
      image.data[offset + 2] = Math.round(pixels[2 * plane + y * imageSize + x] * 255);
      image.data[offset + 3] = 255;
    }
  }
  source.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(off, 0, 0, canvas.width, canvas.height);
  const cell = canvas.width / (imageSize / patch);
  if (overlay?.weights) {
    const gridSize = imageSize / patch;
    const peak = overlay.scale === "unit" ? 1 : Math.max(...overlay.weights, 1e-6);
    for (let index = 0; index < overlay.weights.length; index += 1) {
      const row = Math.floor(index / gridSize);
      const col = index % gridSize;
      const amount = overlay.weights[index] / peak;
      const ink = overlay.scale === "unit" ? Math.min(0.78, amount * 1.7) : amount * 0.62;
      context.fillStyle = `rgba(79, 70, 229, ${ink})`;
      context.fillRect(col * cell, row * cell, cell, cell);
    }
  }
  if (grid) {
    context.strokeStyle = "rgba(15, 23, 42, 0.35)";
    context.lineWidth = 1;
    const gridSize = imageSize / patch;
    for (let index = 1; index < gridSize; index += 1) {
      context.beginPath();
      context.moveTo(index * cell + 0.5, 0);
      context.lineTo(index * cell + 0.5, canvas.height);
      context.stroke();
      context.beginPath();
      context.moveTo(0, index * cell + 0.5);
      context.lineTo(canvas.width, index * cell + 0.5);
      context.stroke();
    }
  }
  if (overlay && overlay.selected > 0) {
    const gridSize = imageSize / patch;
    const index = overlay.selected - 1;
    const row = Math.floor(index / gridSize);
    const col = index % gridSize;
    context.strokeStyle = "#f59e0b";
    context.lineWidth = 3;
    context.strokeRect(col * cell + 1.5, row * cell + 1.5, cell - 3, cell - 3);
  }
  if (overlay?.hover && overlay.hover > 0) {
    const gridSize = imageSize / patch;
    const index = overlay.hover - 1;
    const row = Math.floor(index / gridSize);
    const col = index % gridSize;
    context.strokeStyle = "#ffffff";
    context.lineWidth = 2;
    context.strokeRect(col * cell + 4, row * cell + 4, cell - 8, cell - 8);
  }
}

export function EffectMap({
  pixels,
  imageSize,
  patch,
  drops,
  hidden,
  display,
  onToggle,
}: {
  pixels: ArrayLike<number>;
  imageSize: number;
  patch: number;
  drops: number[];
  hidden: number[];
  display: number;
  onToggle: (patchIndex: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    drawScene(canvas, pixels, imageSize, patch, null, true);
    const context = canvas.getContext("2d");
    if (!context) return;
    const gridSize = imageSize / patch;
    const cell = canvas.width / gridSize;
    const peak = Math.max(...drops.map((value) => Math.max(0, value)), 0.02);
    drops.forEach((drop, index) => {
      const row = Math.floor(index / gridSize);
      const col = index % gridSize;
      const removed = hidden.includes(index);
      if (drop > 0) {
        context.fillStyle = `rgba(190, 18, 60, ${(drop / peak) * 0.72})`;
        context.fillRect(col * cell, row * cell, cell, cell);
      }
      if (removed) {
        context.strokeStyle = "#0f172a";
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(col * cell + 4, row * cell + 4);
        context.lineTo((col + 1) * cell - 4, (row + 1) * cell - 4);
        context.moveTo((col + 1) * cell - 4, row * cell + 4);
        context.lineTo(col * cell + 4, (row + 1) * cell - 4);
        context.stroke();
      }
    });
  }, [pixels, imageSize, patch, drops, hidden, display]);
  return (
    <canvas
      ref={ref}
      width={display}
      height={display}
      className="rounded-md bg-slate-900"
      style={{ width: display, height: display, imageRendering: "pixelated" }}
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const x = Math.min(imageSize - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * imageSize)));
        const y = Math.min(imageSize - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * imageSize)));
        const col = Math.floor(x / patch);
        const row = Math.floor(y / patch);
        onToggle(row * (imageSize / patch) + col);
      }}
    />
  );
}

export function Picture({
  pixels,
  imageSize,
  patch,
  overlay,
  grid,
  display,
  onPick,
  onMove,
}: {
  pixels: ArrayLike<number>;
  imageSize: number;
  patch: number;
  overlay: Overlay | null;
  grid: boolean;
  display: number;
  onPick?: (token: number) => void;
  onMove?: (token: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    drawScene(ref.current, pixels, imageSize, patch, overlay, grid);
  }, [pixels, imageSize, patch, overlay, grid, display]);
  return (
    <canvas
      ref={ref}
      width={display}
      height={display}
      className={`rounded-md bg-slate-900 ${onMove ? "cursor-crosshair" : ""}`}
      style={{ width: display, height: display, imageRendering: "pixelated" }}
      onClick={(event) => {
        const token = patchToken(event, imageSize, patch);
        if (token !== null) onPick?.(token);
      }}
      onPointerMove={(event) => {
        if (!onMove) return;
        const token = patchToken(event, imageSize, patch);
        if (token !== null) onMove(token);
      }}
    />
  );
}

function patchToken(event: MouseEvent<HTMLCanvasElement>, imageSize: number, patch: number): number | null {
  const rect = event.currentTarget.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return null;
  const x = Math.min(imageSize - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * imageSize)));
  const y = Math.min(imageSize - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * imageSize)));
  const col = Math.floor(x / patch);
  const row = Math.floor(y / patch);
  return 1 + row * (imageSize / patch) + col;
}

export function PopCard({
  x,
  y,
  title,
  formula,
  body,
  note,
  onClose,
  children,
}: {
  x: number;
  y: number;
  title: string;
  formula: string;
  body: string;
  note?: string;
  onClose: () => void;
  children?: React.ReactNode;
}) {
  const left = Math.min(x, window.innerWidth - 380);
  const top = Math.min(y, window.innerHeight - 80);
  return (
    <div
      className="fixed z-40 w-[360px] rounded-xl border border-slate-200 bg-white p-4 shadow-2xl"
      style={{ left: Math.max(8, left), top: Math.max(8, top) }}
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-serif text-lg leading-tight text-slate-900">{title}</h3>
        <button type="button" onClick={onClose} className="text-sm text-slate-400 hover:text-slate-700">
          Close
        </button>
      </div>
      <p className="mt-2 overflow-x-auto rounded-md bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-700">{formula}</p>
      <p className="mt-2 text-[13px] leading-5 text-slate-600">{body}</p>
      {note ? <p className="mt-2 text-[12px] leading-5 text-indigo-700">{note}</p> : null}
      {children ? <div className="mt-3">{children}</div> : null}
    </div>
  );
}
