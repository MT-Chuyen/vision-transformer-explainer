import { useEffect, useRef, useState, type MutableRefObject, type PointerEvent } from "react";
import type { ModelFile } from "../model/forward";
import type { Sampling } from "../model/sample";
import { Picture } from "./visual";

const TEMPERATURES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

const PAINTS: { name: string; rgb: [number, number, number] }[] = [
  { name: "Red", rgb: [1, 0, 0] },
  { name: "Green", rgb: [0, 1, 0] },
  { name: "Blue", rgb: [0, 0, 1] },
  { name: "Yellow", rgb: [1, 1, 0] },
  { name: "White", rgb: [1, 1, 1] },
  { name: "Eraser", rgb: [0, 0, 0] },
];

export function Controls({
  model,
  exampleId,
  pixels,
  onExample,
  onPaint,
  onUpload,
  temperature,
  onTemperature,
  sampling,
  onSampling,
  sampled,
  best,
  onResample,
}: {
  model: ModelFile;
  exampleId: string | null;
  pixels: Float32Array;
  onExample: (id: string) => void;
  onPaint: (x: number, y: number, rgb: [number, number, number]) => void;
  onUpload: (pixels: Float32Array) => void;
  temperature: number;
  onTemperature: (value: number) => void;
  sampling: Sampling;
  onSampling: (value: Sampling) => void;
  sampled: string;
  best: string;
  onResample: () => void;
}) {
  const [painting, setPainting] = useState(false);
  const [color, setColor] = useState(0);
  const drawing = useRef(false);
  const temperatureIndex = Math.max(0, TEMPERATURES.indexOf(temperature));
  const { imageSize, patchSize } = model.config;

  return (
    <section data-focus="controls" className="border-b border-slate-200 bg-white px-5 py-3">
      <div className="flex flex-wrap items-end gap-6">
        <div className="min-w-[280px] flex-1">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">Image</div>
          <div className="flex flex-wrap gap-1.5">
            {model.examples.map((example) => (
              <ExampleButton
                key={example.id}
                pixels={example.pixels}
                imageSize={imageSize}
                patch={patchSize}
                label={example.label}
                active={example.id === exampleId}
                onClick={() => onExample(example.id)}
              />
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setPainting((value) => !value)}
              className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            >
              {painting ? "Hide canvas" : "Paint"}
            </button>
            <label className="cursor-pointer rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50">
              Upload
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (!file) return;
                  readImage(file, imageSize, onUpload);
                  event.target.value = "";
                }}
              />
            </label>
          </div>
        </div>

        <label className="w-40 text-xs text-slate-500">
          <span className="mb-1 block font-semibold uppercase tracking-[0.14em] text-slate-400">Temperature</span>
          <input
            type="range"
            min={0}
            max={TEMPERATURES.length - 1}
            step={1}
            value={temperatureIndex}
            onChange={(event) => onTemperature(TEMPERATURES[Number(event.target.value)])}
            className="w-full"
          />
          <span className="tabular-nums">{temperature.toFixed(1)}</span>
        </label>

        <div className="w-48 text-xs text-slate-500">
          <div className="mb-1 font-semibold uppercase tracking-[0.14em] text-slate-400">Sampling</div>
          <div className="flex gap-3">
            {(["top-k", "top-p"] as const).map((type) => (
              <label key={type} className="flex items-center gap-1">
                <input
                  type="radio"
                  name="sampling"
                  checked={sampling.type === type}
                  onChange={() => onSampling({ type, value: type === "top-k" ? 5 : 0.9 })}
                />
                {type}
              </label>
            ))}
          </div>
          <input
            className="mt-1 w-full"
            type="range"
            min={sampling.type === "top-k" ? 1 : 0.1}
            max={sampling.type === "top-k" ? model.config.classes.length : 1}
            step={sampling.type === "top-k" ? 1 : 0.05}
            value={sampling.value}
            onChange={(event) => onSampling({ ...sampling, value: Number(event.target.value) })}
          />
          <span className="tabular-nums">
            {sampling.type === "top-k" ? "k" : "p"}={sampling.type === "top-k" ? sampling.value : sampling.value.toFixed(2)}
          </span>
        </div>

        <div className="ml-auto min-w-[180px]">
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">Sampled class</div>
          <div className="font-serif text-3xl leading-none text-indigo-700">{sampled}</div>
          <div className="mt-1 text-xs text-slate-500">most probable: {best}</div>
          <button
            type="button"
            onClick={onResample}
            className="mt-2 rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500"
          >
            Sample again
          </button>
        </div>
      </div>

      {painting ? (
        <div className="mt-3 flex flex-wrap items-start gap-4 border-t border-slate-100 pt-3">
          <Painter
            pixels={pixels}
            imageSize={imageSize}
            drawing={drawing}
            color={PAINTS[color].rgb}
            onPaint={onPaint}
          />
          <div className="flex flex-col gap-1">
            {PAINTS.map((paint, index) => (
              <button
                key={paint.name}
                type="button"
                onClick={() => setColor(index)}
                className={`flex items-center gap-2 rounded-md px-2 py-1 text-xs ${color === index ? "bg-slate-100 font-semibold" : ""}`}
              >
                <span className="h-3 w-3 rounded-sm border border-slate-300" style={{ background: `rgb(${paint.rgb.map((channel) => channel * 255).join(" ")})` }} />
                {paint.name}
              </button>
            ))}
            <p className="mt-2 max-w-[220px] text-[11px] leading-4 text-slate-400">
              The canvas is the model input, 32×32. Painting replaces the example and runs the same forward pass.
            </p>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function ExampleButton({
  pixels,
  imageSize,
  patch,
  label,
  active,
  onClick,
}: {
  pixels: number[];
  imageSize: number;
  patch: number;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      title={label}
      onClick={onClick}
      className={`rounded-md p-0.5 ${active ? "ring-2 ring-indigo-500" : "ring-1 ring-slate-200 hover:ring-slate-400"}`}
    >
      <Picture pixels={pixels} imageSize={imageSize} patch={patch} overlay={null} grid={false} display={36} />
    </button>
  );
}

function drawPaint(canvas: HTMLCanvasElement, pixels: Float32Array) {
  const size = 32;
  const off = document.createElement("canvas");
  off.width = size;
  off.height = size;
  const source = off.getContext("2d");
  const context = canvas.getContext("2d");
  if (!source || !context) return;
  const image = source.createImageData(size, size);
  const plane = size * size;
  for (let index = 0; index < plane; index += 1) {
    image.data[index * 4] = Math.round(pixels[index] * 255);
    image.data[index * 4 + 1] = Math.round(pixels[plane + index] * 255);
    image.data[index * 4 + 2] = Math.round(pixels[2 * plane + index] * 255);
    image.data[index * 4 + 3] = 255;
  }
  source.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(off, 0, 0, canvas.width, canvas.height);
}

function Painter({
  pixels,
  imageSize,
  drawing,
  color,
  onPaint,
}: {
  pixels: Float32Array;
  imageSize: number;
  drawing: MutableRefObject<boolean>;
  color: [number, number, number];
  onPaint: (x: number, y: number, rgb: [number, number, number]) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) drawPaint(ref.current, pixels);
  }, [pixels]);
  return (
    <canvas
      ref={ref}
      width={256}
      height={256}
      className="cursor-crosshair rounded-md border border-slate-300 bg-black"
      style={{ imageRendering: "pixelated" }}
      onPointerDown={(event) => {
        drawing.current = true;
        event.currentTarget.setPointerCapture(event.pointerId);
        stamp(event, imageSize, color, onPaint);
      }}
      onPointerMove={(event) => {
        if (!drawing.current) return;
        stamp(event, imageSize, color, onPaint);
      }}
      onPointerUp={() => {
        drawing.current = false;
      }}
    />
  );
}

function stamp(
  event: PointerEvent<HTMLCanvasElement>,
  imageSize: number,
  rgb: [number, number, number],
  onPaint: (x: number, y: number, rgb: [number, number, number]) => void,
) {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = Math.min(imageSize - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * imageSize)));
  const y = Math.min(imageSize - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * imageSize)));
  onPaint(x, y, rgb);
}

function readImage(file: File, imageSize: number, onUpload: (pixels: Float32Array) => void) {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = imageSize;
    canvas.height = imageSize;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(image, 0, 0, imageSize, imageSize);
    const data = context.getImageData(0, 0, imageSize, imageSize).data;
    const pixels = new Float32Array(3 * imageSize * imageSize);
    const plane = imageSize * imageSize;
    for (let index = 0; index < plane; index += 1) {
      pixels[index] = data[index * 4] / 255;
      pixels[plane + index] = data[index * 4 + 1] / 255;
      pixels[2 * plane + index] = data[index * 4 + 2] / 255;
    }
    onUpload(pixels);
    URL.revokeObjectURL(url);
  };
  image.src = url;
}

