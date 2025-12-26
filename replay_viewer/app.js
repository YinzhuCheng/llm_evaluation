/* LLM QA Playback Viewer (static, no build step).
 * - Import results_*.jsonl (one JSON per line)
 * - Navigate (prev/next + long-press auto skip)
 * - Jump to ID / index
 * - Show 3 cards (Question / LLM Response / Expert)
 * - MathJax render; optional LLM fix with undo, with CORS-safe manual fallback
 * - Export/Import annotations.json; Print to PDF with print CSS
 */

const $ = (sel) => document.querySelector(sel);

function nowIso() {
  return new Date().toISOString();
}

function safeJsonParse(line) {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

function normalizeId(v) {
  const s = (v ?? "").toString().trim();
  return s;
}

function textOrEmpty(v) {
  if (v === null || v === undefined) return "";
  return typeof v === "string" ? v : JSON.stringify(v, null, 2);
}

function escapeHtml(s) {
  return (s ?? "")
    .toString()
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function isTruthyBool(v) {
  if (v === true) return true;
  if (v === false) return false;
  return null;
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function downloadJson(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function tryCopy(text) {
  const s = text ?? "";
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(s);
  // fallback
  const ta = document.createElement("textarea");
  ta.value = s;
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
  return Promise.resolve();
}

// ----------------------------
// State
// ----------------------------

const state = {
  records: /** @type {Array<any>} */ ([]),
  byId: /** @type {Map<string, number>} */ (new Map()),
  idx: 0,
  runLabel: "—",

  // annotation model:
  // annotations = {
  //   meta: { created_at, updated_at, source },
  //   records: {
  //     [id]: {
  //       edited: { question?: string, llm?: string },
  //       expert?: string,
  //       image_zoom?: number,
  //       math_fixes: { stack: Array<{field, prev, next, at, mode, notes}> },
  //       fixed: { question?: string, llm?: string, expert?: string }
  //     }
  //   },
  //   ui: { colors, image_base }
  // }
  annotations: null,

  // undo stack for math fixes (global, last operation)
  fixUndo: /** @type {Array<any>} */ ([]),
};

function defaultAnnotations() {
  return {
    meta: {
      created_at: nowIso(),
      updated_at: nowIso(),
      source: "",
      viewer: "replay_viewer_v1",
    },
    records: {},
    ui: {
      colors: {
        question: $("#colorQuestion").value,
        llm: $("#colorLLM").value,
        expert: $("#colorExpert").value,
      },
      image_base: $("#imageBase").value,
      use_fixed: $("#chkUseFixed").checked,
    },
  };
}

function ensureAnnotations() {
  if (!state.annotations) state.annotations = defaultAnnotations();
  return state.annotations;
}

function touchAnnUpdated() {
  const ann = ensureAnnotations();
  ann.meta.updated_at = nowIso();
}

function annForCurrent() {
  const ann = ensureAnnotations();
  const rec = state.records[state.idx];
  const id = normalizeId(rec?.id);
  if (!id) return null;
  if (!ann.records[id]) {
    ann.records[id] = { edited: {}, expert: "", image_zoom: 100, fixed: {}, fix_stack: [] };
  }
  return ann.records[id];
}

function uiSetCssVars() {
  const q = $("#colorQuestion").value;
  const l = $("#colorLLM").value;
  const e = $("#colorExpert").value;
  document.documentElement.style.setProperty("--color-question", q);
  document.documentElement.style.setProperty("--color-llm", l);
  document.documentElement.style.setProperty("--color-expert", e);

  const ann = ensureAnnotations();
  ann.ui.colors = { question: q, llm: l, expert: e };
  touchAnnUpdated();
}

function uiSetImageBase() {
  const base = ($("#imageBase").value || "images").trim();
  const ann = ensureAnnotations();
  ann.ui.image_base = base;
  touchAnnUpdated();
}

function uiSetUseFixed() {
  const ann = ensureAnnotations();
  ann.ui.use_fixed = $("#chkUseFixed").checked;
  touchAnnUpdated();
  renderCurrent();
}

function getUseFixed() {
  const ann = ensureAnnotations();
  return !!ann.ui.use_fixed;
}

// ----------------------------
// Parsing JSONL
// ----------------------------

async function loadJsonlFile(file) {
  const text = await file.text();
  const lines = text.split(/\r?\n/);
  const records = [];
  let bad = 0;
  for (const ln of lines) {
    const s = ln.trim();
    if (!s) continue;
    const obj = safeJsonParse(s);
    if (!obj || typeof obj !== "object") {
      bad += 1;
      continue;
    }
    records.push(obj);
  }

  // Build id->index map (first occurrence)
  const byId = new Map();
  for (let i = 0; i < records.length; i++) {
    const id = normalizeId(records[i].id);
    if (id && !byId.has(id)) byId.set(id, i);
  }

  state.records = records;
  state.byId = byId;
  state.idx = 0;
  state.runLabel = file.name;

  const ann = ensureAnnotations();
  ann.meta.source = file.name;
  touchAnnUpdated();

  $("#loadedCount").textContent = String(records.length);
  $("#runLabel").textContent = state.runLabel;

  if (bad) setMathStatus(`Loaded with ${bad} malformed JSON lines skipped.`);
  else setMathStatus(`Loaded ${records.length} records.`);

  renderCurrent();
}

// ----------------------------
// Rendering
// ----------------------------

function getModelName(rec) {
  // Try a few common locations.
  const mc = rec?.model_call;
  const req = mc?.request;
  const model = req?.model ?? rec?.model ?? "";
  const provider = rec?.model_call?.request ? "api" : (rec?.model_provider ?? "");
  const out = (model || provider || "").toString().trim();
  return out || "—";
}

function formatCorrectBadge(rec) {
  const jc = rec?.judge_correct;
  const v = isTruthyBool(jc);
  const badge = $("#metaCorrectBadge");
  badge.classList.remove("correct", "incorrect", "unknown");

  if (v === true) {
    badge.textContent = "Correct (judge_correct=true)";
    badge.classList.add("badge", "correct");
    return;
  }
  if (v === false) {
    badge.textContent = "Incorrect (judge_correct=false)";
    badge.classList.add("badge", "incorrect");
    return;
  }
  badge.textContent = "Unknown (no judge result)";
  badge.classList.add("badge", "unknown");
}

function getEffectiveText(rec, annRec) {
  const useFixed = getUseFixed();
  const fixed = annRec?.fixed || {};
  const edited = annRec?.edited || {};

  const origQuestion = textOrEmpty(rec?.question);
  const origLLM = textOrEmpty(rec?.model_text);

  const question = edited.question?.trim()
    ? edited.question
    : useFixed && fixed.question?.trim()
      ? fixed.question
      : origQuestion;
  const llm = edited.llm?.trim()
    ? edited.llm
    : useFixed && fixed.llm?.trim()
      ? fixed.llm
      : origLLM;
  const expert = useFixed && fixed.expert?.trim() ? fixed.expert : (annRec?.expert || "");
  return { question, llm, expert, origQuestion, origLLM };
}

function renderOptions(rec) {
  const opts = rec?.options;
  const container = $("#questionOptions");
  if (!Array.isArray(opts) || opts.length === 0) {
    container.textContent = "";
    return;
  }
  const lines = opts.map((x) => (x ?? "").toString());
  container.innerHTML = `<div class="label">Options</div><div class="content">${escapeHtml(lines.join("\n"))}</div>`;
}

function renderMeta(rec) {
  const idx = state.idx + 1;
  const total = state.records.length;
  $("#metaIdx").textContent = `${idx}/${total}`;
  $("#metaId").textContent = normalizeId(rec?.id) || "—";
  $("#metaModel").textContent = getModelName(rec);
  formatCorrectBadge(rec);

  const parts = [];
  if (rec?.question_type_inferred) parts.push(`Type: ${rec.question_type_inferred}`);
  if (rec?.question_type_raw) parts.push(`Raw type: ${rec.question_type_raw}`);
  if (rec?.gold !== undefined) parts.push(`Gold: ${textOrEmpty(rec.gold)}`);
  if (rec?.pred !== undefined) parts.push(`Pred: ${textOrEmpty(rec.pred)}`);
  if (rec?.answer_json_ok !== undefined) parts.push(`Answer JSON OK: ${String(rec.answer_json_ok)}`);
  if (rec?.majority_vote_n !== undefined) parts.push(`Majority vote: ${String(rec.majority_vote_n)}`);
  $("#questionMeta").textContent = parts.join(" · ");

  const llmParts = [];
  if (rec?.latency_ms !== undefined) llmParts.push(`Model latency: ${String(rec.latency_ms)}ms`);
  if (rec?.judge_latency_ms !== undefined) llmParts.push(`Judge latency: ${String(rec.judge_latency_ms ?? 0)}ms`);
  if (rec?.total_latency_ms !== undefined) llmParts.push(`Total: ${String(rec.total_latency_ms ?? 0)}ms`);
  $("#llmSummary").textContent = llmParts.join(" · ");

  // Judge summary (readable)
  const j = rec?.judge;
  const jj = j?.judge_json;
  const judgeLines = [];
  if (jj?.verdict) judgeLines.push(`Verdict: ${String(jj.verdict)}`);
  if (jj?.extracted_answer_normalized) judgeLines.push(`Extracted (normalized): ${String(jj.extracted_answer_normalized)}`);
  if (jj?.extracted_answer && !jj?.extracted_answer_normalized) judgeLines.push(`Extracted: ${String(jj.extracted_answer)}`);
  if (jj?.reason) judgeLines.push(`Reason: ${String(jj.reason)}`);
  if (judgeLines.length === 0 && j?.judge_text) judgeLines.push(`Judge text: ${String(j.judge_text)}`);
  $("#judgeSummary").textContent = judgeLines.join("\n");
}

function renderImage(rec, annRec) {
  const img = $("#questionImage");
  const hint = $("#imageHint");
  const base = ($("#imageBase").value || "images").trim().replace(/\/+$/, "");
  const id = normalizeId(rec?.id);
  if (!id) {
    img.style.display = "none";
    hint.textContent = "";
    return;
  }

  const zoom = clamp(Number(annRec?.image_zoom ?? 100), 10, 300);
  $("#imgZoom").value = String(zoom);
  $("#imgZoomLabel").textContent = `${zoom}%`;
  img.style.transform = `scale(${zoom / 100})`;

  // try common extensions
  const exts = ["png", "jpg", "jpeg", "webp", "gif"];
  let k = 0;
  img.onerror = () => {
    k += 1;
    if (k >= exts.length) {
      img.style.display = "none";
      hint.textContent = `No image found for id=${id} under ${base}/`;
      img.onerror = null;
      return;
    }
    img.src = `${base}/${encodeURIComponent(id)}.${exts[k]}`;
  };

  img.onload = () => {
    img.style.display = "block";
    hint.textContent = `${base}/${id}.* (zoom adjustable)`;
  };

  img.src = `${base}/${encodeURIComponent(id)}.${exts[k]}`;
}

function setMathStatus(msg) {
  $("#mathStatus").textContent = msg || "";
}

function detectMathErrors() {
  const root = document.querySelector(".viewer");
  if (!root) return 0;
  // MathJax errors are often rendered with class mjx-merror
  const errs = root.querySelectorAll(".mjx-merror");
  return errs.length;
}

async function typesetMath() {
  // Degrade gracefully if MathJax is unavailable
  if (!window.MathJax?.typesetPromise) {
    setMathStatus("MathJax not loaded (still usable).");
    return;
  }
  try {
    await window.MathJax.typesetPromise();
    const n = detectMathErrors();
    if (n) setMathStatus(`MathJax: ${n} rendering error(s) detected.`);
    else setMathStatus("MathJax: OK.");
  } catch (e) {
    setMathStatus(`MathJax typeset failed: ${e?.message || e}`);
  }
}

function renderCurrent() {
  const rec = state.records[state.idx];
  if (!rec) {
    $("#questionText").textContent = "Import a results_*.jsonl file to begin.";
    $("#llmText").textContent = "";
    $("#expertText").value = "";
    $("#editQuestion").value = "";
    $("#editLLM").value = "";
    $("#questionOptions").textContent = "";
    $("#questionMeta").textContent = "";
    $("#llmSummary").textContent = "";
    $("#metaIdx").textContent = "—";
    $("#metaId").textContent = "—";
    $("#metaModel").textContent = "—";
    $("#metaCorrectBadge").textContent = "—";
    return;
  }

  const annRec = annForCurrent();
  const eff = getEffectiveText(rec, annRec);
  renderMeta(rec);

  $("#questionText").textContent = eff.question || "";
  renderOptions(rec);

  $("#llmText").textContent = eff.llm || "";

  // keep edit boxes in sync with stored edits (but do not override active typing)
  $("#editQuestion").value = annRec?.edited?.question ?? "";
  $("#editLLM").value = annRec?.edited?.llm ?? "";
  $("#expertText").value = annRec?.expert ?? "";

  renderImage(rec, annRec);
  typesetMath();
}

// ----------------------------
// Navigation (including long press)
// ----------------------------

function gotoIndex(i) {
  if (!state.records.length) return;
  const next = clamp(i, 0, state.records.length - 1);
  state.idx = next;
  renderCurrent();
}

function gotoId(id) {
  const key = normalizeId(id);
  if (!key) return;
  const i = state.byId.get(key);
  if (i === undefined) {
    setMathStatus(`ID not found: ${key}`);
    return;
  }
  gotoIndex(i);
}

function step(delta) {
  gotoIndex(state.idx + delta);
}

function setupLongPress(btn, delta) {
  let tHold = null;
  let tRepeat = null;
  const start = () => {
    step(delta);
    tHold = setTimeout(() => {
      tRepeat = setInterval(() => step(delta), 80);
    }, 380);
  };
  const stop = () => {
    if (tHold) clearTimeout(tHold);
    if (tRepeat) clearInterval(tRepeat);
    tHold = null;
    tRepeat = null;
  };

  btn.addEventListener("mousedown", (e) => {
    e.preventDefault();
    start();
  });
  btn.addEventListener("mouseup", stop);
  btn.addEventListener("mouseleave", stop);
  btn.addEventListener("touchstart", (e) => {
    e.preventDefault();
    start();
  }, { passive: false });
  btn.addEventListener("touchend", stop);
  btn.addEventListener("touchcancel", stop);
}

// ----------------------------
// Edits & annotations
// ----------------------------

function saveEditsFromUI() {
  const annRec = annForCurrent();
  if (!annRec) return;
  annRec.edited.question = ($("#editQuestion").value || "").toString();
  annRec.edited.llm = ($("#editLLM").value || "").toString();
  annRec.expert = ($("#expertText").value || "").toString();
  touchAnnUpdated();
}

function clearLocalEdits() {
  state.annotations = defaultAnnotations();
  setMathStatus("Cleared local edits.");
  renderCurrent();
}

function importAnnotationsFile(file) {
  file.text().then((txt) => {
    const obj = safeJsonParse(txt);
    if (!obj || typeof obj !== "object") {
      setMathStatus("Invalid annotations.json");
      return;
    }
    state.annotations = obj;
    // restore UI settings if present
    const colors = obj?.ui?.colors;
    if (colors?.question) $("#colorQuestion").value = colors.question;
    if (colors?.llm) $("#colorLLM").value = colors.llm;
    if (colors?.expert) $("#colorExpert").value = colors.expert;
    if (obj?.ui?.image_base) $("#imageBase").value = obj.ui.image_base;
    if (typeof obj?.ui?.use_fixed === "boolean") $("#chkUseFixed").checked = obj.ui.use_fixed;
    uiSetCssVars();
    uiSetImageBase();
    setMathStatus("Imported annotations.json");
    renderCurrent();
  });
}

// ----------------------------
// Math fix (LLM)
// ----------------------------

function buildFixPrompt({ question, llm, expert }) {
  return [
    "You are a careful editor. Your task is to fix MathJax/LaTeX rendering issues in the provided text.",
    "Constraints:",
    "- Preserve the original meaning. Do NOT change any non-math content unless required to fix math markup.",
    "- Make math MathJax-friendly. Use $...$ for inline math and $$...$$ for display math when appropriate.",
    "- Keep code blocks as-is (do not introduce or remove triple backticks).",
    "- Remove stray control characters. Avoid invisible Unicode that breaks rendering.",
    "- Output ONLY the fixed plain text. No markdown wrapper. No commentary.",
    "",
    "TEXT:",
    question || llm || expert || "",
  ].join("\n");
}

function chooseFixTarget(rec, annRec) {
  // Prefer the fields that currently have MathJax errors. If none detected, default to LLM response.
  // In practice, this is heuristic; user can re-run multiple times.
  const eff = getEffectiveText(rec, annRec);
  // quick heuristic: if question contains backslashes or $ or \[ treat it as candidate
  const candidates = [
    { field: "question", text: eff.question },
    { field: "llm", text: eff.llm },
    { field: "expert", text: eff.expert },
  ];
  // pick the one with most latex markers
  let best = candidates[1];
  let bestScore = -1;
  for (const c of candidates) {
    const s = c.text || "";
    const score = (s.match(/\\\[|\\\(|\$\$|\$/g) || []).length;
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

async function llmFixCurrent() {
  const rec = state.records[state.idx];
  const annRec = annForCurrent();
  if (!rec || !annRec) return;

  // we always save edits first to not lose typing
  saveEditsFromUI();

  const target = chooseFixTarget(rec, annRec);
  const original = (target.text || "").toString();
  if (!original.trim()) {
    setMathStatus("Nothing to fix (empty text).");
    return;
  }

  const endpoint = ($("#llmEndpoint").value || "").trim();
  const model = ($("#llmModel").value || "").trim();
  const apiKey = ($("#llmApiKey").value || "").trim();
  const mode = $("#llmMode").value;

  const prompt = buildFixPrompt({ [target.field]: original });
  const reqBody = {
    model,
    temperature: 0,
    messages: [
      { role: "system", content: "You fix MathJax/LaTeX formatting issues." },
      { role: "user", content: prompt },
    ],
  };

  const applyFixed = (fixedText, notes) => {
    const fixed = (fixedText || "").toString();
    if (!fixed.trim()) {
      setMathStatus("Fix returned empty text; not applied.");
      return;
    }
    const prev = annRec.fixed?.[target.field] ?? "";
    annRec.fixed = annRec.fixed || {};
    annRec.fixed[target.field] = fixed;
    annRec.fix_stack = annRec.fix_stack || [];
    annRec.fix_stack.push({
      field: target.field,
      prev,
      next: fixed,
      at: nowIso(),
      mode,
      notes: notes || "",
    });
    state.fixUndo.push({ id: normalizeId(rec.id), field: target.field });
    touchAnnUpdated();
    setMathStatus(`Applied fix to ${target.field}.`);
    renderCurrent();
  };

  if (mode === "manual") {
    const curl = [
      "curl -sS " + JSON.stringify(endpoint) + " \\",
      "  -H " + JSON.stringify("Content-Type: application/json") + " \\",
      "  -H " + JSON.stringify(`Authorization: Bearer ${apiKey || "<YOUR_KEY>"}`) + " \\",
      "  -d " + JSON.stringify(JSON.stringify(reqBody)),
    ].join("\n");
    $("#manualRequest").value = curl;
    $("#manualPaste").value = "";
    $("#dlgManual").showModal();

    $("#btnApplyManual").onclick = () => {
      const pasted = ($("#manualPaste").value || "").toString();
      // If user pasted a JSON response, try to extract the content.
      const maybe = safeJsonParse(pasted.trim());
      const extracted = maybe?.choices?.[0]?.message?.content;
      applyFixed(extracted ?? pasted, "manual_paste");
    };
    return;
  }

  if (!endpoint || !model || !apiKey) {
    setMathStatus("Missing LLM settings (endpoint/model/api key).");
    return;
  }

  setMathStatus("Calling LLM to fix math…");
  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(reqBody),
    });
    const data = await resp.json().catch(() => null);
    if (!resp.ok) {
      const msg = data?.error?.message || `${resp.status} ${resp.statusText}`;
      setMathStatus(`LLM call failed: ${msg} (try Manual mode if CORS).`);
      return;
    }
    const out = data?.choices?.[0]?.message?.content ?? "";
    applyFixed(out, "direct_fetch");
  } catch (e) {
    setMathStatus(`LLM call blocked (likely CORS): ${e?.message || e}. Use Manual mode.`);
  }
}

function undoLastFix() {
  const rec = state.records[state.idx];
  const annRec = annForCurrent();
  if (!rec || !annRec) return;

  const stack = annRec.fix_stack || [];
  if (!stack.length) {
    setMathStatus("Nothing to undo.");
    return;
  }
  const last = stack.pop();
  annRec.fixed = annRec.fixed || {};
  annRec.fixed[last.field] = last.prev || "";
  touchAnnUpdated();
  setMathStatus(`Undid fix on ${last.field}.`);
  renderCurrent();
}

// ----------------------------
// Wire up UI
// ----------------------------

function init() {
  uiSetCssVars();

  $("#fileJsonl").addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) loadJsonlFile(f);
  });

  $("#fileAnn").addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) importAnnotationsFile(f);
  });

  $("#btnExportAnn").addEventListener("click", () => {
    saveEditsFromUI();
    const ann = ensureAnnotations();
    // persist UI state
    ann.ui.colors = {
      question: $("#colorQuestion").value,
      llm: $("#colorLLM").value,
      expert: $("#colorExpert").value,
    };
    ann.ui.image_base = $("#imageBase").value;
    ann.ui.use_fixed = $("#chkUseFixed").checked;
    touchAnnUpdated();
    downloadJson("annotations.json", ann);
  });

  $("#btnClearAnn").addEventListener("click", () => clearLocalEdits());

  $("#btnPrint").addEventListener("click", () => window.print());

  setupLongPress($("#btnPrev"), -1);
  setupLongPress($("#btnNext"), +1);

  $("#btnJump").addEventListener("click", () => gotoId($("#jumpId").value));
  $("#btnJumpIdx").addEventListener("click", () => {
    const n = Number($("#jumpIdx").value);
    if (!Number.isFinite(n) || n < 1) return;
    gotoIndex(n - 1);
  });

  $("#jumpId").addEventListener("keydown", (e) => {
    if (e.key === "Enter") gotoId($("#jumpId").value);
  });
  $("#jumpIdx").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const n = Number($("#jumpIdx").value);
      if (Number.isFinite(n) && n >= 1) gotoIndex(n - 1);
    }
  });

  // Colors
  $("#colorQuestion").addEventListener("input", uiSetCssVars);
  $("#colorLLM").addEventListener("input", uiSetCssVars);
  $("#colorExpert").addEventListener("input", uiSetCssVars);

  // Image base
  $("#imageBase").addEventListener("change", () => {
    uiSetImageBase();
    renderCurrent();
  });

  // Image zoom
  $("#imgZoom").addEventListener("input", () => {
    const annRec = annForCurrent();
    if (!annRec) return;
    const v = clamp(Number($("#imgZoom").value), 10, 300);
    annRec.image_zoom = v;
    $("#imgZoomLabel").textContent = `${v}%`;
    touchAnnUpdated();
    renderImage(state.records[state.idx], annRec);
  });

  // Edits auto-save (lightweight, local)
  $("#editQuestion").addEventListener("input", () => {
    const annRec = annForCurrent();
    if (!annRec) return;
    annRec.edited.question = $("#editQuestion").value;
    touchAnnUpdated();
    renderCurrent();
  });
  $("#editLLM").addEventListener("input", () => {
    const annRec = annForCurrent();
    if (!annRec) return;
    annRec.edited.llm = $("#editLLM").value;
    touchAnnUpdated();
    renderCurrent();
  });
  $("#btnSaveExpert").addEventListener("click", () => {
    const annRec = annForCurrent();
    if (!annRec) return;
    annRec.expert = $("#expertText").value || "";
    touchAnnUpdated();
    setMathStatus("Saved expert commentary.");
    renderCurrent();
  });

  $("#chkUseFixed").addEventListener("change", uiSetUseFixed);

  $("#btnFixMath").addEventListener("click", llmFixCurrent);
  $("#btnUndoFix").addEventListener("click", undoLastFix);

  $("#btnCopyLLM").addEventListener("click", () => {
    const rec = state.records[state.idx];
    const annRec = annForCurrent();
    if (!rec || !annRec) return;
    const eff = getEffectiveText(rec, annRec);
    tryCopy(eff.llm || "").then(() => setMathStatus("Copied LLM response."));
  });

  // Keyboard shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.target && ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(+1);
  });

  renderCurrent();
  setMathStatus("Ready. Import a results_*.jsonl to start.");
}

init();

