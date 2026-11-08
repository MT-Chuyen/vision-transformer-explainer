import { readFileSync } from "node:fs";
import { forward, type ModelFile } from "../src/model/forward.ts";

const file = new URL("../public/model.json", import.meta.url);
const model = JSON.parse(readFileSync(file, "utf8")) as ModelFile;
let failed = 0;

for (const example of model.examples) {
  const trace = forward(model, example.pixels, { mask: "full", dropout: 0, seed: 1 });
  const gap = Math.max(...trace.logits.map((value, index) => Math.abs(value - example.logits[index])));
  const got = trace.logits.indexOf(Math.max(...trace.logits));
  const want = example.logits.indexOf(Math.max(...example.logits));
  if (gap > 1e-4 || got !== want) {
    failed += 1;
    console.error(example.id, "gap", gap, "got", got, "want", want);
  }
  for (const block of trace.blocks) {
    for (const head of block.weights) {
      for (const row of head) {
        const sum = row.reduce((total, value) => total + value, 0);
        if (Math.abs(sum - 1) > 1e-5) {
          failed += 1;
          console.error(example.id, "attention row", sum);
        }
      }
    }
  }
}

const first = model.examples[0];
const trace = forward(model, first.pixels, { mask: "full", dropout: 0, seed: 1 });
if (first.debug) {
  const after = trace.afterPos.flat();
  const afterGap = Math.max(...after.map((value, index) => Math.abs(value - first.debug!.afterPos[index])));
  const attn = trace.blocks[0].weights[0].flat();
  const attnGap = Math.max(...attn.map((value, index) => Math.abs(value - first.debug!.attnHead0[index])));
  if (afterGap > 1e-4 || attnGap > 1e-4) {
    failed += 1;
    console.error("debug gap", afterGap, attnGap);
  } else {
    console.log(`embeddings and head-0 attention match (gaps ${afterGap.toExponential(1)}, ${attnGap.toExponential(1)})`);
  }
}

if (failed > 0) {
  console.error(`${failed} checks failed`);
  process.exit(1);
}
console.log(`logits match on ${model.examples.length} images`);
