import { forward, type ModelFile } from "./forward";
import { probabilities } from "./sample";

export function blankPatches(pixels: Float32Array, imageSize: number, patchSize: number, patches: number[]): Float32Array {
  if (!patches.length) return pixels;
  const next = pixels.slice();
  const grid = imageSize / patchSize;
  const plane = imageSize * imageSize;
  for (const patch of patches) {
    const row = Math.floor(patch / grid);
    const col = patch % grid;
    for (let channel = 0; channel < 3; channel += 1) {
      for (let y = 0; y < patchSize; y += 1) {
        for (let x = 0; x < patchSize; x += 1) {
          const yy = row * patchSize + y;
          const xx = col * patchSize + x;
          next[channel * plane + yy * imageSize + xx] = 0;
        }
      }
    }
  }
  return next;
}

export function patchDrops(model: ModelFile, pixels: Float32Array): { drops: number[]; classIndex: number; probability: number } {
  const { imageSize, patchSize } = model.config;
  const grid = imageSize / patchSize;
  const base = probabilities(forward(model, pixels, { mask: "full", dropout: 0, seed: 1 }).logits, 1);
  let classIndex = 0;
  for (let index = 1; index < base.length; index += 1) {
    if (base[index] > base[classIndex]) classIndex = index;
  }
  const drops: number[] = [];
  for (let patch = 0; patch < grid * grid; patch += 1) {
    const altered = blankPatches(pixels, imageSize, patchSize, [patch]);
    const next = probabilities(forward(model, altered, { mask: "full", dropout: 0, seed: 1 }).logits, 1);
    drops.push(base[classIndex] - next[classIndex]);
  }
  return { drops, classIndex, probability: base[classIndex] };
}
