import { useMemo } from "react";
import { architectureSwaps, type SwapRow } from "../model/swap";
import type { ModelFile } from "../model/forward";

export function Swap({ model, pixels }: { model: ModelFile; pixels: Float32Array }) {
  const study = useMemo(() => architectureSwaps(model, pixels), [model, pixels]);
  const peak = Math.max(...study.rows.flatMap((row) => row.gradients), 1e-8);
  const floor = Math.log10(Math.max(Math.min(...study.rows.flatMap((row) => row.gradients)), 1e-4));
  const ceil = Math.log10(peak);
  return (
    <section data-testid="swap" className="border-t border-slate-200 bg-white px-5 py-5">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">If this piece is swapped</h2>
      <p className="mt-1 max-w-3xl text-[11px] leading-4 text-slate-500">
        The true class is {study.trueClass}. Each row keeps the trained weights and the attention output, then changes one piece. The bars are the log size of the gradient of that class logit, at the class token, from the input on the left to the last block on the right.
      </p>
      <div className="mt-3 grid grid-cols-[28px_150px_160px_1fr] items-end gap-x-3 text-[10px] uppercase tracking-wide text-slate-400">
        <span />
        <span>Change</span>
        <span>Result</span>
        <span className="grid grid-cols-5 text-center">
          <span>in</span>
          <span>1</span>
          <span>2</span>
          <span>3</span>
          <span>4</span>
        </span>
      </div>
      <div className="mt-1 space-y-2">
        {study.rows.map((row) => (
          <SwapLine key={row.label} row={row} trueClass={study.trueClass} floor={floor} ceil={ceil} />
        ))}
      </div>
    </section>
  );
}

function SwapLine({ row, trueClass, floor, ceil }: { row: SwapRow; trueClass: string; floor: number; ceil: number }) {
  const held = row.predicted === trueClass;
  return (
    <div className={`grid grid-cols-[28px_150px_160px_1fr] items-center gap-x-3 rounded-lg px-1 py-1 ${row.model ? "bg-indigo-50" : ""}`}>
      <span className={`h-2.5 w-2.5 rounded-full ${row.model ? "bg-indigo-600" : held ? "bg-slate-300" : "bg-rose-500"}`} />
      <span className={`font-mono text-[12px] ${row.model ? "font-semibold text-indigo-800" : "text-slate-700"}`}>{row.label}</span>
      <span>
        <span className={`block text-[12px] ${held ? "text-slate-800" : "font-semibold text-rose-700"}`}>{row.predicted}</span>
        <span className="mt-1 block h-1.5 overflow-hidden rounded-full bg-slate-100">
          <span className={`block h-full rounded-full ${held ? "bg-indigo-500" : "bg-rose-500"}`} style={{ width: `${Math.max(2, row.trueProb * 100)}%` }} />
        </span>
        <span className="font-mono text-[10px] text-slate-400">{(row.trueProb * 100).toFixed(0)}% of {trueClass}</span>
      </span>
      <span className="grid h-12 grid-cols-5 items-end gap-1">
        {row.gradients.map((value, index) => {
          const amount = (Math.log10(Math.max(value, 1e-4)) - floor) / Math.max(ceil - floor, 1e-6);
          return <span key={index} title={value.toExponential(2)} className={`block rounded-sm ${amount > 0.85 ? "bg-rose-500" : "bg-indigo-500"}`} style={{ height: `${Math.max(6, amount * 100)}%` }} />;
        })}
      </span>
    </div>
  );
}
