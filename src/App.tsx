import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { article, guide, type PopoverId, type StepId } from "./content";
import { Controls } from "./components/Controls";
import { Stage } from "./components/Stage";
import { Textbook } from "./components/Textbook";
import { Swap } from "./components/Swap";
import { Why } from "./components/Why";
import { blankPatches, patchDrops } from "./model/effect";
import { forward, type ModelFile } from "./model/forward";
import { probabilities, sampleClass, type Sampling } from "./model/sample";

const SHAPES: [string, string][] = [
  ["Image", "3 × 32 × 32"],
  ["Patches", "16 × 192"],
  ["With CLS and position", "17 × 32"],
  ["Q, K, V", "17 × 96"],
  ["One head", "17 × 8, then a 17 × 17 matrix"],
  ["MLP hidden", "17 × 128"],
  ["Class logits", "9"],
];

export function App() {
  const [model, setModel] = useState<ModelFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pixels, setPixels] = useState<Float32Array | null>(null);
  const [exampleId, setExampleId] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.8);
  const [sampling, setSampling] = useState<Sampling>({ type: "top-k", value: 5 });
  const [sampleSeed, setSampleSeed] = useState(1);
  const [dropoutSeed, setDropoutSeed] = useState(1);
  const [block, setBlock] = useState(0);
  const [head, setHead] = useState(0);
  const [token, setToken] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [mask, setMask] = useState<"full" | "causal">("full");
  const [dropout, setDropout] = useState(0);
  const [expanded, setExpanded] = useState<StepId | null>(null);
  const [popover, setPopover] = useState<{ id: PopoverId; x: number; y: number } | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [guideOpen, setGuideOpen] = useState(false);
  const [guideIndex, setGuideIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hidden, setHidden] = useState<number[]>([]);

  useEffect(() => {
    let gone = false;
    fetch(`${import.meta.env.BASE_URL}model.json`)
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status));
        return response.json() as Promise<ModelFile>;
      })
      .then((loaded) => {
        if (gone) return;
        setModel(loaded);
        setPixels(Float32Array.from(loaded.examples[0].pixels));
        setExampleId(loaded.examples[0].id);
      })
      .catch(() => {
        if (!gone) setError("The weights did not load.");
      });
    return () => {
      gone = true;
    };
  }, []);

  useEffect(() => {
    if (!playing || !model) return;
    const id = window.setInterval(() => {
      setBlock((current) => (current + 1) % model.config.depth);
    }, 900);
    return () => window.clearInterval(id);
  }, [playing, model]);

  useEffect(() => {
    if (!focus) return;
    document.querySelector<HTMLElement>(`[data-focus="${focus}"]`)?.scrollIntoView({
      behavior: "smooth",
      inline: "center",
      block: "nearest",
    });
  }, [focus, guideIndex]);

  const shown = useMemo(() => {
    if (!model || !pixels) return null;
    return blankPatches(pixels, model.config.imageSize, model.config.patchSize, hidden);
  }, [model, pixels, hidden]);

  const trace = useMemo(() => {
    if (!model || !shown) return null;
    return forward(model, shown, { mask, dropout, seed: dropoutSeed });
  }, [model, shown, mask, dropout, dropoutSeed]);

  const effect = useMemo(() => (model && pixels ? patchDrops(model, pixels) : null), [model, pixels]);

  const removedProb = useMemo(() => {
    if (!model || !shown || !effect || !hidden.length) return null;
    const next = probabilities(forward(model, shown, { mask: "full", dropout: 0, seed: 1 }).logits, 1);
    return next[effect.classIndex];
  }, [model, shown, effect, hidden]);

  const probs = useMemo(() => (trace ? probabilities(trace.logits, temperature) : []), [trace, temperature]);
  const draw = useMemo(() => (probs.length ? sampleClass(probs, sampling, sampleSeed) : null), [probs, sampling, sampleSeed]);

  function showGuide(index: number) {
    const page = guide[index];
    setGuideIndex(index);
    setFocus(page.focus);
    setExpanded(page.expand);
  }

  function openPopover(id: PopoverId, event: MouseEvent) {
    const rect = event.currentTarget.getBoundingClientRect();
    setPopover({ id, x: rect.left, y: rect.bottom + 8 });
  }

  if (error) {
    return <p className="p-8 text-sm text-red-700">{error}</p>;
  }
  if (!model || !pixels || !shown || !trace || !draw || !effect) {
    return <p className="p-8 text-sm text-slate-500">Loading the Vision Transformer…</p>;
  }

  const best = probs.indexOf(Math.max(...probs));

  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between bg-[#172554] px-5 py-3 text-white">
        <div>
          <h1 className="text-lg font-semibold">Vision Transformer Explainer</h1>
          <p className="text-xs text-blue-100">
            {model.config.depth} blocks · {model.config.heads} heads · dim {model.config.dim} · validation {(model.config.valAccuracy * 100).toFixed(1)}%
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            setGuideOpen((open) => !open);
            if (!guideOpen) showGuide(guideIndex);
          }}
          className="rounded-md bg-white/10 px-3 py-1.5 text-sm font-semibold hover:bg-white/20"
        >
          Textbook
        </button>
      </header>

      <Controls
        model={model}
        exampleId={exampleId}
        pixels={pixels}
        onExample={(id) => {
          const example = model.examples.find((item) => item.id === id);
          if (!example) return;
          setPixels(Float32Array.from(example.pixels));
          setExampleId(id);
          setSampleSeed(1);
          setToken(0);
          setHidden([]);
        }}
        onPaint={(x, y, rgb) => {
          setPixels((current) => {
            if (!current) return current;
            const next = current.slice();
            const size = model.config.imageSize;
            for (let dy = -1; dy <= 1; dy += 1) {
              for (let dx = -1; dx <= 1; dx += 1) {
                const yy = y + dy;
                const xx = x + dx;
                if (yy < 0 || xx < 0 || yy >= size || xx >= size) continue;
                const offset = yy * size + xx;
                next[offset] = rgb[0];
                next[size * size + offset] = rgb[1];
                next[2 * size * size + offset] = rgb[2];
              }
            }
            return next;
          });
          setExampleId(null);
          setHidden([]);
        }}
        onUpload={(next) => {
          setPixels(next);
          setExampleId(null);
          setSampleSeed(1);
          setHidden([]);
        }}
        temperature={temperature}
        onTemperature={setTemperature}
        sampling={sampling}
        onSampling={setSampling}
        sampled={model.config.classes[draw.index]}
        best={model.config.classes[best]}
        onResample={() => setSampleSeed((seed) => seed + 1)}
      />

      <Stage
        model={model}
        pixels={shown}
        trace={trace}
        block={block}
        setBlock={setBlock}
        head={head}
        setHead={setHead}
        token={token}
        setToken={setToken}
        hover={hover}
        setHover={setHover}
        mask={mask}
        setMask={setMask}
        dropout={dropout}
        setDropout={setDropout}
        onRedrawDropout={() => setDropoutSeed((seed) => seed + 1)}
        expanded={expanded}
        setExpanded={setExpanded}
        popover={popover}
        openPopover={openPopover}
        closePopover={() => setPopover(null)}
        focus={focus}
        probs={probs}
        pool={draw.pool}
        sampled={draw.index}
        playing={playing}
        setPlaying={setPlaying}
        sourcePixels={pixels}
        drops={effect.drops}
        hidden={hidden}
        onTogglePatch={(patchIndex) =>
          setHidden((current) => (current.includes(patchIndex) ? current.filter((item) => item !== patchIndex) : [...current, patchIndex]))
        }
        onClearPatches={() => setHidden([])}
        watchedClass={model.config.classes[effect.classIndex]}
        watchedProb={effect.probability}
        removedProb={removedProb}
      />

      <Why model={model} pixels={shown} trace={trace} block={block} head={head} token={token} mask={mask} />
      <Swap model={model} pixels={shown} />

      <article className="mx-auto max-w-3xl px-6 py-12 text-slate-700">
        {article.map((section) => (
          <section key={section.heading} className="mb-8">
            <h2 className="font-serif text-3xl text-slate-900">{section.heading}</h2>
            {section.paragraphs.map((paragraph) => (
              <p key={paragraph.slice(0, 24)} className="mt-3 text-[15px] leading-7">
                {paragraph}
              </p>
            ))}
          </section>
        ))}
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-300 text-slate-500">
              <th className="py-2 font-medium">Tensor</th>
              <th className="py-2 font-medium">Shape in this model</th>
            </tr>
          </thead>
          <tbody>
            {SHAPES.map(([name, shape]) => (
              <tr key={name} className="border-b border-slate-200">
                <td className="py-2">{name}</td>
                <td className="py-2 font-mono text-[13px]">{shape}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-6 text-sm leading-6 text-slate-500">
          The architecture follows Dosovitskiy et al., An Image is Worth 16×16 Words (
          <a className="text-indigo-700 underline" href="https://arxiv.org/abs/2010.11929">
            arXiv:2010.11929
          </a>
          ). The walkthrough beside it is the same one used for GPT-2 in the Transformer Explainer: embeddings, QKV, attention, MLP, stacked blocks, logits, temperature, and sampling.
        </p>
      </article>

      {guideOpen ? (
        <Textbook
          page={guide[guideIndex]}
          index={guideIndex}
          total={guide.length}
          onPrev={() => showGuide(Math.max(0, guideIndex - 1))}
          onNext={() => showGuide(Math.min(guide.length - 1, guideIndex + 1))}
          onClose={() => {
            setGuideOpen(false);
            setFocus(null);
          }}
        />
      ) : null}
    </div>
  );
}
