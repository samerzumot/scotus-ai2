const els = {
  form: document.getElementById("predict-form"),
  brief: document.getElementById("brief"),
  fileDrop: document.getElementById("file-drop"),
  fileName: document.getElementById("file-name"),
  fileHint: document.getElementById("file-hint"),
  side: document.getElementById("side"),
  caseHint: document.getElementById("case_hint"),
  transcriptUrl: document.getElementById("transcript_url"),
  status: document.getElementById("status"),
  skeleton: document.getElementById("skeleton"),
  toast: document.getElementById("toast"),
  fallbackWarning: document.getElementById("fallback-warning"),
  fallbackMessage: document.getElementById("fallback-message"),
  results: document.getElementById("results"),
  overall: document.getElementById("overall"),
  overallWinner: document.getElementById("overall-winner"),
  overallExplanation: document.getElementById("overall-explanation"),
  confidenceBar: document.getElementById("confidence-bar"),
  confidenceText: document.getElementById("confidence-text"),
  bench: document.getElementById("bench"),
  questions: document.getElementById("questions"),
  backtest: document.getElementById("backtest"),
  retrieval: document.getElementById("retrieval"),
  submitBtn: document.getElementById("submit-btn"),
};

let selectedFile = null;

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function toast(msg, durationMs = 3000) {
  const toastEl = els.toast;
  const messageEl = toastEl.querySelector(".toast__message");
  if (messageEl) {
    messageEl.textContent = msg;
  } else {
    toastEl.textContent = msg;
  }
  toastEl.classList.remove("is-hidden");
  window.clearTimeout(toast._t);
  toast._t = window.setTimeout(() => toastEl.classList.add("is-hidden"), durationMs);
}

function setLoading(isLoading, label) {
  if (els.skeleton) {
    els.skeleton.classList.toggle("is-hidden", !isLoading);
  }
  
  // Only hide results when loading starts, don't hide when loading ends
  if (isLoading && els.results) {
    els.results.classList.add("is-hidden");
  }
  
  if (els.status) {
    els.status.classList.toggle("is-loading", isLoading);
    const statusText = els.status.querySelector(".status__text");
    if (statusText) {
      statusText.textContent = label || (isLoading ? "Analyzing..." : "Ready");
    } else {
      els.status.textContent = label || (isLoading ? "Analyzing..." : "Ready");
    }
  }
  
  if (els.submitBtn) {
    els.submitBtn.disabled = isLoading;
    const btnText = els.submitBtn.querySelector(".btn__text");
    if (btnText) {
      btnText.textContent = isLoading ? "Analyzing..." : "Analyze Brief";
    }
  }
}

function isPdfFile(file) {
  if (!file) return false;
  const name = String(file.name || "").toLowerCase();
  const type = String(file.type || "").toLowerCase();
  return name.endsWith(".pdf") || type === "application/pdf";
}

function setSelectedFile(file) {
  selectedFile = file || null;
  if (els.fileName) {
    els.fileName.textContent = selectedFile ? selectedFile.name : "Click to upload or drag and drop";
  }
}

function voteClass(vote) {
  if (vote === "PETITIONER") return "is-petitioner";
  if (vote === "RESPONDENT") return "is-respondent";
  return "is-uncertain";
}

function voteColor(vote) {
  if (vote === "PETITIONER") return "#2563eb";
  if (vote === "RESPONDENT") return "#dc2626";
  return "#f59e0b";
}

function initFilePicker() {
  try {
    if (window.self !== window.top) {
      if (els.fileHint) {
        els.fileHint.textContent = "If the file picker doesn't open, open this app in your browser at http://localhost:8000";
      }
    }
  } catch (_e) {
    if (els.fileHint) {
      els.fileHint.textContent = "If the file picker doesn't open, open this app in your browser at http://localhost:8000";
    }
  }

  const openDialog = () => {
    try {
      els.brief.click();
    } catch (e) {
      toast("File picker blocked. Open the app in a real browser and try again.");
    }
  };

  if (els.fileDrop) {
    els.fileDrop.addEventListener("click", (e) => {
      if (e.target && e.target.id === "file-name") return;
      openDialog();
    });

    els.fileDrop.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openDialog();
      }
    });

    els.fileDrop.addEventListener("dragover", (e) => {
      e.preventDefault();
      els.fileDrop.classList.add("is-dragover");
    });

    els.fileDrop.addEventListener("dragleave", () => {
      els.fileDrop.classList.remove("is-dragover");
    });

    els.fileDrop.addEventListener("drop", (e) => {
      e.preventDefault();
      els.fileDrop.classList.remove("is-dragover");
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const f = files[0];
        if (isPdfFile(f)) {
          setSelectedFile(f);
        } else {
          toast("Please drop a PDF file.");
        }
      }
    });
  }

  if (els.brief) {
    els.brief.addEventListener("change", (e) => {
      const f = e.target.files?.[0];
      if (f && isPdfFile(f)) {
        setSelectedFile(f);
      } else if (f) {
        toast("Please choose a PDF file.");
        setSelectedFile(null);
      }
    });
  }
}

function renderPrediction(payload) {
  console.log("renderPrediction called with:", payload);
  const pred = payload?.prediction;
  if (!pred) {
    console.error("No prediction data in payload:", payload);
    toast("Error: No prediction data received");
    return;
  }

  // Show results section - this is critical!
  if (els.results) {
    els.results.classList.remove("is-hidden");
    console.log("Results section shown");
  } else {
    console.error("Results element not found! Check HTML for id='results'");
    toast("Error: Results container not found");
    return;
  }

  // Always check for warnings in payload
  const warnings = payload?.warnings || [];
  const overall = pred.overall || {};
  const model = pred.model || {};
  const isFallback = model.provider === "fallback" || (overall.why || "").includes("FALLBACK") || (overall.why || "").includes("⚠️");
  
  // Collect all warning messages
  const allWarnings = [];
  if (isFallback) {
    const msg = overall.why || "Google model not configured. Set GOOGLE_AI_KEY and GOOGLE_PREDICT_MODEL in env.local, then restart the server.";
    allWarnings.push(msg);
  }
  // Add any warnings from payload
  warnings.forEach(w => {
    if (w && !allWarnings.includes(w)) {
      allWarnings.push(w);
    }
  });
  
  // Always show warnings if any exist
  if (allWarnings.length > 0) {
    const warningText = allWarnings.join(" ");
    console.warn("⚠️ FALLBACK WARNING:", warningText);
    toast(`⚠️ WARNING: ${warningText}`, 12000);
    els.fallbackWarning.classList.remove("is-hidden");
    if (els.fallbackMessage) {
      els.fallbackMessage.textContent = allWarnings.join("\n\n");
    }
  } else {
    els.fallbackWarning.classList.add("is-hidden");
  }
  
  // Render overall prediction
  if (els.overallWinner) {
    const winner = overall.predicted_winner || "UNCERTAIN";
    els.overallWinner.textContent = winner;
    els.overallWinner.style.color = voteColor(winner);
  }
  
  if (els.confidenceBar && els.confidenceText) {
    const confidence = Math.round(100 * (overall.confidence || 0));
    els.confidenceBar.style.width = `${confidence}%`;
    els.confidenceText.textContent = `${confidence}%`;
  }
  
  if (els.overallExplanation) {
    const explanation = overall.why || overall.swing_justice 
      ? `${overall.why || ""}${overall.swing_justice ? ` Swing Justice: ${overall.swing_justice}` : ""}`
      : "—";
    els.overallExplanation.textContent = explanation;
  }

  // Render votes
  if (els.bench) {
    els.bench.innerHTML = "";
    (pred.votes || []).forEach((v) => {
      const card = document.createElement("div");
      card.className = `justice-card ${voteClass(v.vote)}`;
      card.innerHTML = `
        <div class="justice-card__name">${escapeHtml(v.justice_name || v.justice_id)}</div>
        <div class="justice-card__vote" style="color: ${voteColor(v.vote)}">${escapeHtml(v.vote || "UNCERTAIN")}</div>
        <div class="justice-card__confidence">Confidence: ${Math.round(100 * (v.confidence || 0))}%</div>
        ${v.rationale ? `<div class="justice-card__rationale">${escapeHtml(v.rationale)}</div>` : ""}
      `;
      els.bench.appendChild(card);
    });
  }

  // Render questions with citations
  if (els.questions) {
    els.questions.innerHTML = "";
    (pred.questions || []).forEach((q) => {
      const card = document.createElement("div");
      card.className = "question-card";
      
      // Extract and format citations
      const questionText = q.question || "";
      const citations = extractCitations(questionText);
      const questionWithCitations = formatQuestionWithCitations(questionText, citations);
      
      card.innerHTML = `
        <div class="question-card__header">
          <span class="question-card__justice">${escapeHtml(q.justice_name || q.justice_id)}</span>
        </div>
        <div class="question-card__text">${questionWithCitations}</div>
        ${citations.length > 0 ? `<div class="question-card__citations">📚 Citations: ${citations.map(c => `<span class="citation">${escapeHtml(c)}</span>`).join(", ")}</div>` : ""}
        ${q.what_it_tests ? `<div class="question-card__meta">${escapeHtml(q.what_it_tests)}</div>` : ""}
      `;
      els.questions.appendChild(card);
    });
  }
  
  // Helper functions for citations
  function extractCitations(text) {
    const citations = [];
    // Pattern: "Case Name v. Case Name"
    const casePattern = /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/g;
    let match;
    while ((match = casePattern.exec(text)) !== null) {
      if (match[1].length > 5 && match[1].length < 150) {
        citations.push(match[1]);
      }
    }
    // Pattern: "### U.S. ###"
    const scotusPattern = /\b(\d{1,4}\s+U\.S\.\s+\d{1,4})/g;
    while ((match = scotusPattern.exec(text)) !== null) {
      citations.push(match[1]);
    }
    return [...new Set(citations)]; // Remove duplicates
  }
  
  function formatQuestionWithCitations(text, citations) {
    let formatted = escapeHtml(text);
    // Highlight citations in the text
    citations.forEach(citation => {
      const regex = new RegExp(`(${citation.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      formatted = formatted.replace(regex, '<strong class="citation-highlight">$1</strong>');
    });
    return formatted;
  }

  // Render retrieved cases with citations
  const r = pred.retrieved_cases || [];
  if (els.retrieval) {
    if (r.length) {
      els.retrieval.innerHTML = r.map((x) => {
        const tags = (x.tags || []).map(t => `<span class="retrieval-item__tag">${escapeHtml(t)}</span>`).join("");
        // Format case name as citation
        const caseName = escapeHtml(x.case_name || x.case_id);
        const caseCitation = formatCaseCitation(x);
        return `
          <div class="retrieval-item">
            <div class="retrieval-item__name">
              <strong>${caseName}</strong>
              ${caseCitation ? `<span class="case-citation">${caseCitation}</span>` : ""}
            </div>
            <div class="retrieval-item__meta">
              ${x.term ? `<span>Term ${escapeHtml(String(x.term))}</span>` : ""}
              ${x.outcome ? `<span>${escapeHtml(x.outcome)}</span>` : ""}
              ${tags ? `<div style="margin-top: 4px;">${tags}</div>` : ""}
            </div>
          </div>
        `;
      }).join("");
    } else {
      els.retrieval.textContent = "—";
    }
  }
  
  // Helper to format case citation
  function formatCaseCitation(caseRef) {
    if (!caseRef.case_name) return "";
    // If case name already looks like a citation, return it
    if (caseRef.case_name.includes("v.") || caseRef.case_name.includes("U.S.")) {
      return caseRef.case_name;
    }
    // Otherwise, just return the case name as-is
    return "";
  }

  // Render backtest
  const bt = payload?.backtest;
  if (els.backtest) {
    if (!bt) {
      els.backtest.textContent = "—";
    } else {
      const found = bt.transcript_found ? "found" : "not found";
      const url = bt.transcript_url ? bt.transcript_url : "(none)";
      const score = bt.questions_score_pct ?? 0;
      const autoDetected = bt.transcript_auto_detected ? " (auto-detected)" : "";
      const explanation = bt.explanation || "";
      
      let scoreClass = "backtest-score";
      if (score >= 70) scoreClass += " backtest-score--excellent";
      else if (score >= 50) scoreClass += " backtest-score--good";
      else if (score >= 30) scoreClass += " backtest-score--moderate";
      else scoreClass += " backtest-score--poor";
      
      els.backtest.innerHTML = `
        <div class="${scoreClass}">
          <div class="backtest-score__value">${score}%</div>
          <div class="backtest-score__label">Question Match Score</div>
        </div>
        <div style="font-size: 12px; color: var(--color-text-secondary); margin-top: 8px;">
          <strong>Transcript:</strong> ${escapeHtml(url)}${escapeHtml(autoDetected)} (${escapeHtml(found)})
        </div>
        ${
          explanation
            ? `<div class="backtest-explanation" style="margin-top: 12px;">${escapeHtml(explanation)}</div>`
            : ""
        }
        ${
          (bt.matches || []).length
            ? `<details style="margin-top: 12px;">
                <summary style="cursor: pointer; font-weight: 600; margin-bottom: 8px;">Top Question Matches</summary>
                <div style="margin-top: 8px; display: flex; flex-direction: column; gap: 8px;">
                  ${bt.matches
                    .slice(0, 6)
                    .map(
                      (m) =>
                        `<div style="padding: 8px; background: var(--color-bg); border-radius: 8px;">
                          <div style="font-weight: 600; margin-bottom: 4px; color: var(--color-primary);">
                            ${m.justice_name ? `Justice ${escapeHtml(m.justice_name)}` : "Question"} Match
                          </div>
                          <div style="font-weight: 600; margin-bottom: 4px; font-size: 12px;">Predicted:</div>
                          <div style="font-size: 13px; margin-bottom: 8px; line-height: 1.5;">
                            ${formatQuestionWithCitations(m.predicted || "", m.predicted_citations || [])}
                          </div>
                          ${(m.predicted_citations || []).length > 0 ? `<div style="font-size: 11px; color: var(--color-text-secondary); margin-bottom: 8px; font-style: italic;">📚 Citations: ${(m.predicted_citations || []).map(c => escapeHtml(c)).join(", ")}</div>` : ""}
                          <div style="font-weight: 600; margin-bottom: 4px; font-size: 12px;">Best Actual Match:</div>
                          <div style="font-size: 13px; margin-bottom: 4px; line-height: 1.5;">
                            ${formatQuestionWithCitations(m.best_actual || "", m.actual_citations || [])}
                          </div>
                          ${(m.actual_citations || []).length > 0 ? `<div style="font-size: 11px; color: var(--color-text-secondary); margin-bottom: 8px; font-style: italic;">📚 Citations: ${(m.actual_citations || []).map(c => escapeHtml(c)).join(", ")}</div>` : ""}
                          <div style="font-size: 12px; color: var(--color-text-secondary); margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--color-border);">
                            Similarity: ${Math.round(100 * (m.similarity || 0))}%
                          </div>
                        </div>`
                    )
                    .join("")}
                </div>
              </details>`
            : ""
        }
      `;
    }
  }
}

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = selectedFile || els.brief.files?.[0];
  if (!file) return toast("Please select a PDF brief first.");
  if (!isPdfFile(file)) return toast("Please choose a PDF file.");

  const fd = new FormData();
  fd.append("brief", file);
  fd.append("side", els.side.value || "UNKNOWN");
  fd.append("case_hint", (els.caseHint?.value || "").trim());
  fd.append("transcript_url", (els.transcriptUrl?.value || "").trim());
  fd.append("run_backtest", "true"); // Always run backtest if transcript available

  setLoading(true, "Uploading and analyzing brief...");
  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "Prediction failed.");
    }
    
    console.log("Received data:", data); // Debug log
    
    if (!data.data || !data.data.prediction) {
      throw new Error("Invalid response format: missing prediction data");
    }
    
    renderPrediction(data.data);
    setLoading(false, "Analysis complete");
    toast("Analysis complete!", 2000);
    
    // Scroll to results
    if (els.results) {
      els.results.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    console.error("Error:", err);
    toast(String(err.message || err));
    setLoading(false, "Error occurred");
  }
});

// Hot reload for development
async function startHotReload() {
  const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);
  if (!isLocal) return;

  let token = null;
  const pollMs = 900;

  async function tick() {
    try {
      const res = await fetch("/__hotreload", { cache: "no-store" });
      if (!res.ok) throw new Error("no-hotreload");
      const data = await res.json();
      if (!data || !data.ok) throw new Error("no-hotreload");
      const next = data.token ?? 0;
      if (token === null) token = next;
      else if (String(next) !== String(token)) {
        window.location.reload();
        return;
      }
    } catch (_e) {
      // Server may be restarting; ignore and keep polling.
    }
    window.setTimeout(tick, pollMs);
  }

  window.setTimeout(tick, pollMs);
}

startHotReload();
initFilePicker();
