import type { GuidePage } from "../content";

export function Textbook({
  page,
  index,
  total,
  onPrev,
  onNext,
  onClose,
}: {
  page: GuidePage;
  index: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  return (
    <aside className="fixed bottom-4 right-4 top-20 z-20 flex w-[340px] flex-col rounded-2xl border border-slate-200 bg-white shadow-2xl">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
          Textbook {index + 1} / {total}
        </p>
        <button type="button" onClick={onClose} className="text-sm text-slate-400 hover:text-slate-700">
          Close
        </button>
      </div>
      <div className="flex-1 overflow-auto px-4 py-4">
        <h2 className="font-serif text-2xl leading-tight text-slate-900">{page.title}</h2>
        <p className="mt-3 text-[14px] leading-6 text-slate-600">{page.body}</p>
      </div>
      <div className="flex justify-between border-t border-slate-100 px-4 py-3">
        <button type="button" onClick={onPrev} disabled={index === 0} className="text-sm font-semibold text-slate-600 disabled:opacity-30">
          Back
        </button>
        <button type="button" onClick={onNext} disabled={index === total - 1} className="text-sm font-semibold text-indigo-600 disabled:opacity-30">
          Next
        </button>
      </div>
    </aside>
  );
}
