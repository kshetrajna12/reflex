// reflex in the browser: the same System One readout as the Python engine, on transformers.js.
// Pure inference module (no DOM) so it can be tested in Node and used by the page.
//
//   const engine = await loadEngine({ device: "webgpu", onProgress });
//   const resp = await engine.answer({ state, questions }, { image, temperature });
//
// Per question we build "state + question + options" exactly like src/reflex/prompt.py,
// run ONE forward pass (no generation), keep only the label-token logits (A/B/C or
// Yes/No), temperature-scale and softmax. That distribution is the answer.

export const MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX-OPT";
export const IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>";
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const SYSTEM_PROMPT =
  "You are a System One decision model. You read the State and answer each Question " +
  "by choosing exactly one of the listed options. You never explain. You answer with " +
  "the single option label only.";

export function renderText(x) {
  if (x == null) return "";
  if (typeof x === "string") return x;
  return JSON.stringify(x, null, 2);
}

// Replace {"type":"image","source":...} objects by the placeholder; return how many were found.
export function splitImages(state) {
  let n = 0;
  const walk = (x) => {
    if (x && typeof x === "object" && !Array.isArray(x) && x.type === "image") { n += 1; return IMAGE_PLACEHOLDER; }
    if (Array.isArray(x)) return x.map(walk);
    if (x && typeof x === "object") return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, walk(v)]));
    return x;
  };
  return [walk(state), n];
}

export function prefix(state) {
  return `<|im_start|>system\n${SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n# State\n${renderText(state)}\n\n`;
}

function optionsBlock(instructions, labelled, ask) {
  const lines = [`# Question\n${renderText(instructions)}\n`, "# Options"];
  for (const [label, desc] of labelled) lines.push(desc ? `${label}. ${desc}` : `${label}.`);
  lines.push(`\n${ask}\n`);
  return lines.join("\n") + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

// -> { kind, labels, keys, text }
export function buildBranch(q) {
  if (q.type === "noul") {
    const t = renderText(q.criteria?.true) || "The statement is true.";
    const f = renderText(q.criteria?.false) || "The statement is false.";
    return { kind: "noul", labels: ["Yes", "No"], keys: [true, false],
      text: optionsBlock(q.instructions, [["Yes", t], ["No", f]], "Respond with only Yes or No.") };
  }
  if (q.type === "choice") {
    const keys = Object.keys(q.criteria || {});
    if (keys.length < 2 || keys.length > 26) throw new Error("choice needs 2..26 options");
    const labels = keys.map((_, i) => LETTERS[i]);
    const labelled = keys.map((k, i) => { const d = renderText(q.criteria[k]); return [labels[i], d ? `${k}: ${d}` : k]; });
    return { kind: "choice", labels, keys, text: optionsBlock(q.instructions, labelled, "Respond with only the letter of the best option.") };
  }
  if (q.type === "score") {
    const levels = q.criteria || [];
    if (levels.length < 2 || levels.length > 10) throw new Error("score needs 2..10 levels");
    const keys = levels.map((_, i) => i);
    const labels = keys.map((i) => LETTERS[i]);
    const labelled = keys.map((i) => [labels[i], `(level ${i} of ${levels.length - 1}) ${renderText(levels[i])}`]);
    return { kind: "score", labels, keys, text: optionsBlock(q.instructions, labelled, "Respond with only the letter of the level that best matches.") };
  }
  throw new Error(`unknown question type ${q.type}`);
}

export function softmax(z, T = 1) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp((v - m) / T));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function confidence(p) {
  const n = p.length;
  if (n <= 1) return 1;
  const h = -p.reduce((a, v) => a + (v > 1e-12 ? v * Math.log(v) : 0), 0);
  return Math.max(0, Math.min(1, 1 - h / Math.log(n)));
}

export function toAnswer(kind, keys, p, q) {
  if (kind === "noul") return { type: "noul", noul: p[0] };
  if (kind === "choice") {
    const probabilities = Object.fromEntries(keys.map((k, i) => [k, p[i]]));
    const best = keys[p.indexOf(Math.max(...p))];
    return { type: "choice", choice: best, probabilities, confidence: confidence(p) };
  }
  const score = p.reduce((a, v, i) => a + v * i, 0);
  return { type: "score", score,
    legend: Object.fromEntries(keys.map((i) => [String(i), renderText(q.criteria[i])])),
    probabilities: Object.fromEntries(keys.map((i) => [String(i), p[i]])), confidence: confidence(p) };
}

export async function loadEngine({ transformers, device = "webgpu", modelId = MODEL_ID, onProgress } = {}) {
  const { AutoProcessor, Qwen3_5ForConditionalGeneration } = transformers;
  const processor = await AutoProcessor.from_pretrained(modelId, { progress_callback: onProgress });
  const model = await Qwen3_5ForConditionalGeneration.from_pretrained(modelId, {
    dtype: { embed_tokens: "q4", vision_encoder: "fp16", decoder_model_merged: "q4" },
    device,
    progress_callback: onProgress,
  });
  const tok = processor.tokenizer;
  const labelIds = new Map();
  const labelId = (l) => {
    if (!labelIds.has(l)) {
      const ids = tok.encode(l, { add_special_tokens: false });
      if (ids.length !== 1) throw new Error(`label ${l} is not a single token`);
      labelIds.set(l, ids[0]);
    }
    return labelIds.get(l);
  };

  // One forward pass over "state + branch"; returns restricted logits in label order.
  async function branchLogits(text, image, labels) {
    const inputs = await processor(text, image ?? null);
    const out = await model.forward(inputs);
    const [, L, V] = out.logits.dims;
    const last = out.logits.data.subarray((L - 1) * V, L * V);
    const z = labels.map((l) => Number(last[labelId(l)]));
    return { z, tokens: L };
  }

  async function answer(req, { image = null, temperature = 1.0, onQuestion } = {}) {
    const [stateText, nImages] = splitImages(req.state);
    if (nImages > 1) throw new Error("the browser demo supports one image per request");
    if (nImages === 1 && !image) throw new Error("state references an image but none was provided");
    const pre = prefix(stateText);
    const answers = {};
    let tokens = 0;
    const t0 = performance.now();
    for (const [qid, q] of Object.entries(req.questions)) {
      const br = buildBranch(q);
      const { z, tokens: L } = await branchLogits(pre + br.text, nImages ? image : null, br.labels);
      tokens += L;
      answers[qid] = toAnswer(br.kind, br.keys, softmax(z, temperature), q);
      onQuestion?.(qid, answers[qid]);
    }
    return { model: modelId, answers, usage: { input_tokens: tokens, questions: Object.keys(req.questions).length, ms: performance.now() - t0 } };
  }

  return { processor, model, answer, device, modelId };
}
