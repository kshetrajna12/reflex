import * as transformers from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
import { loadEngine, MODEL_ID, DTYPES } from "./reflex.js";
import { fetchPR, reviewPR } from "./pr.js";

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
  if ($("lowpower").checked) engine.setPolicy({ lowPower: true });
  setStatus(`Ready: ${MODEL.split("/").pop()} · ${DTYPE} · ${device}. The first run compiles shaders and is slower.`, 100, "ready");
  $("run").disabled = false;
  $("pr-run").disabled = false;
  $("pr-status").textContent = "Ready.";
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
    const stateNote = u.path === "shared"
      ? (u.state_cache_hit ? `state cached (${u.state_tokens} tokens skipped, ${u.forwards} small passes)` : `state encoded once + ${u.questions} small passes (low power)`)
      : `1 batched pass, state not cached yet`;
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
$("lowpower").addEventListener("change", () => { const v = $("lowpower").checked; engine?.setPolicy({ lowPower: v ? true : "auto" }); try { localStorage.setItem("reflex.lowpower", v ? "1" : "0"); } catch {} });
try { if (localStorage.getItem("reflex.lowpower") === "1") $("lowpower").checked = true; } catch {}
$("run").addEventListener("click", run);
$("load").addEventListener("click", () => ensureEngine().catch((e) => setStatus(`Error: ${e.message}`, null, "idle")));
loadPreset("ticket");
if (!navigator.gpu) $("gpu-warning").hidden = false;


// ---------------------------------------------------------------- PR triage section
function prStatus(text, pct, mode = "idle") {
  $("pr-status").textContent = text;
  $("pr-bar").style.width = pct == null ? "0%" : `${Math.round(100 * pct)}%`;
  $("pr-dot").className = `dot ${mode}`;
}
const pct = (p) => `${(100 * p).toFixed(0)}%`;

function renderPR(out) {
  const s = out.summary;
  const kinds = Object.entries(s.kind_probs).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k} ${pct(v)}`).join(" · ");
  const flags = [];
  if (s.needs_tests) flags.push("needs tests: behaviour changes, no test files touched");
  if (s.sensitive) flags.push(`${s.sensitive} hunk(s) in sensitive areas`);
  if (s.weakens_errors) flags.push(`${s.weakens_errors} hunk(s) weaken error handling`);
  if (s.debug) flags.push(`${s.debug} hunk(s) with debug leftovers`);
  if (s.breaking_change > 0.5) flags.push("likely breaking change");
  const row = (r) => `<tr class="${r.is_test ? "test" : ""}"><td class="p">${pct(r.p_needs_eyes)}</td><td class="f">${esc(r.file)}<br><span class="meta">${esc(r.header.slice(0, 40))} +${r.added}/−${r.removed}</span></td><td>${["sensitive_area", "removes_error_handling", "public_api_change", "leftover_debug"].filter((k) => r[k] > 0.5).map((k) => k.replaceAll("_", " ")).join(", ") || "<span class='meta'>—</span>"}</td></tr>`;
  const list = [...s.escalate, ...s.escalate_tests];
  $("pr-out").innerHTML = `
    <h3 style="margin:16px 0 4px;font:500 22px/1.2 var(--serif)"><a href="${out.pr.url}" target="_blank" rel="noopener">#${out.pr.number}</a> ${esc(out.pr.title)}</h3>
    <div class="prhead">
      <div class="stat"><div class="k">kind</div><div class="v">${esc(s.kind)}<small>${kinds}</small></div></div>
      <div class="stat"><div class="k">risk</div><div class="v">${s.risk.toFixed(2)}<small>of 3</small></div></div>
      <div class="stat"><div class="k">breaking</div><div class="v">${pct(s.breaking_change)}</div></div>
      <div class="stat"><div class="k">needs migration</div><div class="v">${pct(s.needs_migration)}</div></div>
      <div class="stat"><div class="k">description matches</div><div class="v">${pct(s.description_matches)}</div></div>
      <div class="stat"><div class="k">hunks</div><div class="v">${s.reviewed}<small>of ${s.total_hunks}, ${s.logic_hunks} change behaviour</small></div></div>
    </div>
    <div>${flags.length ? flags.map((f) => `<span class="flag">${esc(f)}</span>`).join("") : "<span class='meta'>no flags</span>"}</div>
    <div class="cardtitle" style="margin-top:18px">Hunks to hand to a reviewer (P(careful read) ≥ 50%)</div>
    ${list.length ? `<div class="tablewrap"><table class="hunks"><thead><tr><th>P</th><th>hunk</th><th>why</th></tr></thead><tbody>${list.map(row).join("")}</tbody></table></div>` : "<p class='meta'>nothing; every hunk looks routine</p>"}
    <p class="meta" style="margin-top:10px">Test hunks are greyed. A 0.8B model judging code: treat these as a triage order, not a verdict, and calibrate on your own PR history before trusting thresholds.</p>`;
}

async function runPR() {
  const repo = $("pr-repo").value.trim();
  const number = Number($("pr-number").value);
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo) || !number) { prStatus("Enter owner/repo and a PR number.", null, "idle"); return; }
  $("pr-run").disabled = true;
  $("pr-out").innerHTML = "";
  try {
    const eng = await ensureEngine();
    prStatus("Fetching from GitHub…", 0, "busy");
    const pr = await fetchPR(repo, number);
    const t0 = performance.now();
    const out = await reviewPR(eng, pr, { maxHunks: Number($("pr-max").value), temperature: Number($("temp").value), onProgress: (t, p) => prStatus(t, p, "busy") });
    renderPR(out);
    prStatus(`Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`, 1, "ready");
  } catch (e) {
    prStatus(`Error: ${e.message}`, null, "idle");
    console.error(e);
  } finally {
    $("pr-run").disabled = false;
  }
}
$("pr-run").addEventListener("click", runPR);
$("pr-max").addEventListener("input", () => { $("pr-max-val").textContent = $("pr-max").value; });
