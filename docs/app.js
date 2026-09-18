import * as transformers from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
import { loadEngine, MODEL_ID, DTYPES } from "./reflex.js";

// ?dtype=q4|q4f16|fp16 and ?model=<hub id> let you A/B speed without redeploying.
const params = new URLSearchParams(location.search);
const DTYPE = DTYPES[params.get("dtype")] ? params.get("dtype") : "q4f16";
const MODEL = params.get("model") || MODEL_ID;

const $ = (id) => document.getElementById(id);
const PRESETS = {
  ticket: {
    label: "Support ticket (text)",
    state: {
      ticket: { subject: "Payouts failing", body: "My payouts have failed three times this week and nobody has replied to my two emails. I have contractors waiting on this money. If it's not fixed by Friday I'm moving to a competitor." },
      customer: { plan: "pro", tenure_months: 27 },
    },
    questions: {
      queue: { type: "choice", instructions: "Which team should handle this ticket?", criteria: { payments: "Payouts, refunds, invoices, failed charges", account: "Login, profile, permissions, 2FA", other: "Anything else" } },
      escalate: { type: "noul", instructions: "Should this ticket be escalated to a human manager right away?" },
      urgency: { type: "score", instructions: "How urgent is this ticket?", criteria: ["Low: can wait several days", "Medium: should be handled today", "High: money or access is blocked right now"] },
      refund_requested: { type: "noul", instructions: "Does the customer explicitly ask for a refund?" },
    },
  },
  photo: {
    label: "Photo triage (image)",
    state: { photo: { type: "image", source: "(the picked image)" }, caption: "a tabby cat sitting on grass" },
    questions: {
      subject: { type: "choice", instructions: "What is the main subject of the photo?", criteria: { person: null, animal: null, food: null, vehicle: null, landscape: "outdoor scenery, nature, cityscape", document: "text, screenshot, receipt", product: "an object for sale, packaging", other: null } },
      contains_text: { type: "noul", instructions: "Does the image contain readable text?" },
      matches_caption: { type: "noul", instructions: "Does the `caption` in the state accurately describe the photo?" },
      quality: { type: "score", instructions: "How good is the technical image quality?", criteria: ["Unusable: extremely blurry, dark, or corrupted", "Poor: noticeable blur, noise, or bad exposure", "Acceptable: minor flaws", "Good: sharp and well exposed"] },
    },
  },
  shapes: {
    label: "Shapes test image (image)",
    state: { photo: { type: "image", source: "(the picked image)" }, note: "A synthetic test image with simple shapes." },
    questions: {
      red_shape: { type: "choice", instructions: "What shape is the red object in the photo?", criteria: { circle: null, square: null, triangle: null } },
      has_blue: { type: "noul", instructions: "Is there a blue shape in the photo?" },
      clutter: { type: "score", instructions: "How cluttered is the photo?", criteria: ["empty or one or two simple shapes", "several objects", "very busy scene"] },
    },
  },
};

let engine = null;
let image = null; // transformers RawImage

function setStatus(text, pct, mode = "idle") {
  $("status").textContent = text;
  $("bar").style.width = pct == null ? "0%" : `${Math.round(pct)}%`;
  $("dot").className = `dot ${mode}`;
}

function loadPreset(name) {
  const p = PRESETS[name];
  $("state").value = JSON.stringify(p.state, null, 2);
  $("questions").value = JSON.stringify(p.questions, null, 2);
  if (name === "shapes") drawShapes();
}

async function drawShapes() {
  const c = document.createElement("canvas");
  c.width = c.height = 448;
  const g = c.getContext("2d");
  g.fillStyle = "white"; g.fillRect(0, 0, 448, 448);
  g.fillStyle = "red"; g.beginPath(); g.arc(160, 160, 100, 0, Math.PI * 2); g.fill();
  g.fillStyle = "blue"; g.fillRect(280, 280, 140, 140);
  await setImageFromBlob(await new Promise((r) => c.toBlob(r, "image/png")));
}

async function setImageFromBlob(blob) {
  const url = URL.createObjectURL(blob);
  $("preview").src = url;
  $("preview").hidden = false;
  let img = await transformers.RawImage.read(url);
  const longest = Math.max(img.width, img.height);
  const maxEdge = 512; // ~256 image tokens; keeps the demo snappy
  if (longest > maxEdge) {
    const s = maxEdge / longest;
    img = await img.resize(Math.round(img.width * s), Math.round(img.height * s));
  }
  image = img;
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function bar(label, p, best) {
  const pct = (p * 100).toFixed(1);
  return `<div class="row${best ? " best" : ""}"><span class="lab" title="${esc(label)}">${esc(label)}</span><span class="track"><span class="fill" data-w="${pct}"></span></span><span class="pct">${pct}%</span></div>`;
}

function renderAnswer(qid, a) {
  let body = "", verdict = "";
  if (a.type === "noul") {
    body = bar("yes", a.noul, a.noul >= 0.5) + bar("no", 1 - a.noul, a.noul < 0.5);
    verdict = `<b>${a.noul >= 0.5 ? "yes" : "no"}</b> · P(yes) = ${a.noul.toFixed(2)}`;
  } else if (a.type === "choice") {
    body = Object.entries(a.probabilities).map(([k, p]) => bar(k, p, k === a.choice)).join("");
    verdict = `<b>${esc(a.choice)}</b> · confidence ${a.confidence.toFixed(2)}`;
  } else {
    const n = Object.keys(a.probabilities).length - 1;
    const top = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0][0];
    body = Object.entries(a.probabilities).map(([k, p]) => bar(`${k} · ${a.legend[k]}`, p, k === top)).join("");
    verdict = `<b>${a.score.toFixed(2)}</b> on a 0–${n} scale · confidence ${a.confidence.toFixed(2)}`;
  }
  const el = document.createElement("div");
  el.className = "answer";
  el.innerHTML = `<div><h3>${esc(qid)}</h3><span class="kind">${a.type}</span><div class="verdict">${verdict}</div></div><div>${body}</div>`;
  const empty = $("answers").querySelector(".empty"); if (empty) empty.remove();
  $("answers").appendChild(el);
  requestAnimationFrame(() => el.querySelectorAll(".fill").forEach((f) => { f.style.width = f.dataset.w + "%"; }));
}

async function ensureEngine() {
  if (engine) return engine;
  const hasWebGPU = !!navigator.gpu;
  const device = hasWebGPU ? "webgpu" : "wasm";
  if (!hasWebGPU) $("gpu-warning").hidden = false;
  const seen = new Map();
  setStatus(`Loading ${MODEL} (${DTYPE}) on ${device}…`, 0, "busy");
  engine = await loadEngine({
    transformers, device, modelId: MODEL, dtype: DTYPE,
    onProgress: (e) => {
      if (e.status === "progress") seen.set(e.file, [e.loaded, e.total]);
      let l = 0, t = 0;
      for (const [a, b] of seen.values()) { l += a; t += b; }
      if (t) setStatus(`Downloading model… ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB`, (100 * l) / t, "busy");
      if (e.status === "ready") setStatus("Compiling…", 100, "busy");
    },
  });
  setStatus(`Ready: ${MODEL.split("/").pop()} · ${DTYPE} · ${device}. The first run compiles shaders and is slower.`, 100, "ready");
  $("run").disabled = false;
  $("load").disabled = true;
  return engine;
}

async function run() {
  $("answers").innerHTML = "";
  $("usage").hidden = true;
  let state, questions;
  try {
    const s = $("state").value.trim();
    state = s.startsWith("{") || s.startsWith("[") ? JSON.parse(s) : s;
    questions = JSON.parse($("questions").value);
  } catch (e) { setStatus(`Bad JSON: ${e.message}`, null, "idle"); return; }
  const temperature = Number($("temp").value);
  $("run").disabled = true;
  try {
    const eng = await ensureEngine();
    setStatus("Thinking: one forward pass per question, no text generated…", 100, "busy");
    const resp = await eng.answer({ state, questions }, { image, temperature, onQuestion: renderAnswer });
    const u = resp.usage;
    const stateNote = u.state_cache_hit ? `state cached (${u.state_tokens} tokens skipped, ${u.forwards} small passes)` : `${u.forwards} batched pass, state not cached yet`;
    $("usage").textContent = `${u.questions} questions · ${stateNote} · ${u.input_tokens} tokens run · ${u.ms.toFixed(0)} ms total · ${DTYPE} · temperature ${temperature}`;
    $("usage").hidden = false;
    setStatus("Done.", 100, "ready");
  } catch (e) {
    setStatus(`Error: ${e.message}`, null, "idle");
    console.error(e);
  } finally {
    $("run").disabled = false;
  }
}

for (const [k, p] of Object.entries(PRESETS)) {
  const o = document.createElement("option"); o.value = k; o.textContent = p.label; $("preset").appendChild(o);
}
$("preset").addEventListener("change", (e) => loadPreset(e.target.value));
$("file").addEventListener("change", async (e) => { if (e.target.files[0]) await setImageFromBlob(e.target.files[0]); });
$("temp").addEventListener("input", () => { $("temp-val").textContent = $("temp").value; });
$("run").addEventListener("click", run);
$("load").addEventListener("click", () => ensureEngine().catch((e) => setStatus(`Error: ${e.message}`, null, "idle")));
loadPreset("ticket");
if (!navigator.gpu) $("gpu-warning").hidden = false;
