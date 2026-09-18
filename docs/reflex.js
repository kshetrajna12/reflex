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

// dtype presets for the decoder. q4f16 (4-bit weights, fp16 activations) is the default:
// measured 1.4x faster than q4 on WebGPU. q4 (fp32 activations) is the fallback for
// backends without fp16.
export const DTYPES = {
  q4: { embed_tokens: "q4", vision_encoder: "fp16", decoder_model_merged: "q4" },
  q4f16: { embed_tokens: "q4f16", vision_encoder: "fp16", decoder_model_merged: "q4f16" },
  fp16: { embed_tokens: "fp16", vision_encoder: "fp16", decoder_model_merged: "fp16" },
};

export async function loadEngine({ transformers, device = "webgpu", modelId = MODEL_ID, dtype = "q4f16", onProgress } = {}) {
  const { AutoProcessor, AutoModelForImageTextToText, cat, Tensor } = transformers;
  const processor = await AutoProcessor.from_pretrained(modelId, { progress_callback: onProgress });
  const model = await AutoModelForImageTextToText.from_pretrained(modelId, {
    dtype: DTYPES[dtype] ?? dtype,
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

  // Sequential: one forward pass per branch (kept for comparison / fallback).
  async function branchLogits(text, image, labels) {
    const inputs = await processor(text, image ?? null);
    const out = await model.forward(inputs);
    const [, L, V] = out.logits.dims;
    const last = out.logits.data.subarray((L - 1) * V, L * V);
    return { z: labels.map((l) => Number(last[labelId(l)])), tokens: L };
  }

  // Batched: every branch is one row of a right-padded batch, ONE forward pass for the
  // whole request. Same idea as the Python engine's "batched" strategy. The image (if
  // any) is preprocessed once and its patches repeated per row so the vision encoder
  // runs as one batched call too.
  async function batchLogits(texts, image, labelsPerRow) {
    const B = texts.length;
    let imageInputs = {};
    if (image) {
      // preprocess once, repeat the patch tensor per row (one batched vision-encoder call)
      const one = await processor.image_processor(image);
      imageInputs = {
        pixel_values: cat(Array(B).fill(one.pixel_values), 0),
        image_grid_thw: cat(Array(B).fill(one.image_grid_thw), 0),
      };
      const merge = processor.image_processor.config.merge_size ** 2;
      const n = Number(one.image_grid_thw.tolist()[0].reduce((a, b) => a * b, 1n)) / merge;
      texts = texts.map((t) => t.replace("<|image_pad|>", "<|image_pad|>".repeat(n)));
    }
    // Left-pad so the last position of every row is its real last token, then ask the
    // graph for logits at that one position only (num_logits_to_keep = 1). Without this
    // the output projection runs over every token of every row against a 248k vocab and
    // hundreds of MB of logits are copied back from the GPU per request.
    tok.padding_side = "left";
    const textInputs = tok(texts, { padding: true, truncation: false });
    const out = await model.forward({ ...textInputs, ...imageInputs, num_logits_to_keep: new Tensor("int64", [1n], []) });
    const [, K, V] = out.logits.dims;
    const mask = textInputs.attention_mask.data;
    const L = textInputs.attention_mask.dims[1];
    const rows = [];
    let tokens = 0;
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < L; i++) if (Number(mask[b * L + i])) tokens++;
      const last = out.logits.data.subarray((b * K + K - 1) * V, (b * K + K) * V);
      rows.push(labelsPerRow[b].map((l) => Number(last[labelId(l)])));
    }
    return { rows, tokens };
  }

  // ---- prefix sharing -------------------------------------------------------------
  // Run the state once and keep the decoder cache (attention KV + linear-attention
  // conv/recurrent states); then run each question branch alone, continuing from that
  // cache. The ONNX GroupQueryAttention op refuses a *batched* multi-token continuation
  // from a cache ("batch_size must be 1 when sequence_length > 1 and past context is
  // given"), so unlike the Python engine's batched strategy this is one small forward
  // per question. Each forward only sees the branch tokens (no state re-read, no
  // padding, logits at one position), so a request costs state + sum(branches) tokens.

  function disposeAll(obj) {
    for (const t of Object.values(obj)) t?.dispose?.();
  }

  function cacheFromOutputs(out) {
    const cache = {};
    for (const name of Object.keys(out)) {
      if (!name.startsWith("present")) continue;
      cache[name.replace("present_conv", "past_conv").replace("present_recurrent", "past_recurrent").replace("present_ssm", "past_ssm").replace("present", "past_key_values")] = out[name];
    }
    return cache;
  }

  async function encodePrefix(prefixText, image) {
    let imageInputs = {};
    let text = prefixText;
    if (image) {
      const one = await processor.image_processor(image);
      imageInputs = { pixel_values: one.pixel_values, image_grid_thw: one.image_grid_thw };
      const merge = processor.image_processor.config.merge_size ** 2;
      const n = Number(one.image_grid_thw.tolist()[0].reduce((a, b) => a * b, 1n)) / merge;
      text = text.replace("<|image_pad|>", "<|image_pad|>".repeat(n));
    }
    const enc = tok(text);
    const [position_ids] = model.get_rope_index(enc.input_ids, imageInputs.image_grid_thw ?? null, null, enc.attention_mask);
    const out = await model.forward({ ...enc, ...imageInputs, position_ids, num_logits_to_keep: new Tensor("int64", [1n], []) });
    return { ids: enc.input_ids, cache: cacheFromOutputs(out), grid: imageInputs.image_grid_thw ?? null, dispose() { disposeAll(this.cache); } };
  }

  async function sharedLogits(pre, branchTexts, labelsPerRow) {
    const P = pre.ids.dims[1];
    const rows = [];
    let tokens = P;
    for (let b = 0; b < branchTexts.length; b++) {
      const enc = tok(branchTexts[b]);
      const L = enc.input_ids.dims[1];
      const full_ids = cat([pre.ids, enc.input_ids], 1);
      const full_mask = new Tensor("int64", new BigInt64Array(P + L).fill(1n), [1, P + L]);
      const [full_pos] = model.get_rope_index(full_ids, pre.grid, null, full_mask);
      const out = await model.forward({
        input_ids: enc.input_ids, attention_mask: full_mask, position_ids: full_pos.slice(null, null, [P, null]),
        past_key_values: pre.cache, num_logits_to_keep: new Tensor("int64", [1n], []),
      });
      const V = out.logits.dims[2];
      const last = out.logits.data.subarray(0, V);
      rows.push(labelsPerRow[b].map((l) => Number(last[labelId(l)])));
      disposeAll(cacheFromOutputs(out)); // GPU-resident present.* outputs would leak otherwise
      tokens += L;
    }
    return { rows, tokens };
  }

  const stateCache = new Map(); // prefix text (+image id) -> encoded prefix, tiny LRU
  const STATE_CACHE_MAX = 4;

  async function answer(req, { image = null, temperature = 1.0, onQuestion, share = true, batch = true } = {}) {
    const [stateText, nImages] = splitImages(req.state);
    if (nImages > 1) throw new Error("the browser demo supports one image per request");
    if (nImages === 1 && !image) throw new Error("state references an image but none was provided");
    const pre = prefix(stateText);
    const img = nImages ? image : null;
    const entries = Object.entries(req.questions).map(([qid, q]) => ({ qid, q, br: buildBranch(q) }));
    const answers = {};
    let usage;
    const t0 = performance.now();
    if (share) {
      const key = pre + (img ? `|${img.width}x${img.height}|${img.data.length}` : "");
      let encoded = stateCache.get(key);
      const hit = !!encoded;
      if (!encoded) {
        encoded = await encodePrefix(pre, img);
        stateCache.set(key, encoded);
        if (stateCache.size > STATE_CACHE_MAX) {
          const oldest = stateCache.keys().next().value;
          stateCache.get(oldest).dispose();
          stateCache.delete(oldest);
        }
      }
      const tPrefix = performance.now();
      const { rows, tokens } = await sharedLogits(encoded, entries.map((e) => e.br.text), entries.map((e) => e.br.labels));
      entries.forEach((e, i) => { answers[e.qid] = toAnswer(e.br.kind, e.br.keys, softmax(rows[i], temperature), e.q); onQuestion?.(e.qid, answers[e.qid]); });
      usage = { input_tokens: hit ? tokens - encoded.ids.dims[1] : tokens, state_tokens: encoded.ids.dims[1], state_cache_hit: hit, forwards: entries.length + (hit ? 0 : 1), prefix_ms: tPrefix - t0 };
    } else if (batch) {
      const { rows, tokens } = await batchLogits(entries.map((e) => pre + e.br.text), img, entries.map((e) => e.br.labels));
      entries.forEach((e, i) => { answers[e.qid] = toAnswer(e.br.kind, e.br.keys, softmax(rows[i], temperature), e.q); onQuestion?.(e.qid, answers[e.qid]); });
      usage = { input_tokens: tokens, forwards: 1 };
    } else {
      let tokens = 0;
      for (const e of entries) {
        const { z, tokens: L } = await branchLogits(pre + e.br.text, img, e.br.labels);
        tokens += L;
        answers[e.qid] = toAnswer(e.br.kind, e.br.keys, softmax(z, temperature), e.q);
        onQuestion?.(e.qid, answers[e.qid]);
      }
      usage = { input_tokens: tokens, forwards: entries.length };
    }
    return { model: modelId, answers, usage: { ...usage, questions: entries.length, ms: performance.now() - t0 } };
  }

  return { processor, model, answer, device, modelId, dtype };
}
