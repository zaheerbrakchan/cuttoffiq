const questionEl = document.getElementById("question");
const askBtn = document.getElementById("askBtn");
const loadingEl = document.getElementById("loading");
const resultEl = document.getElementById("result");
const errorBoxEl = document.getElementById("errorBox");
const userQueryEl = document.getElementById("userQuery");
const answerTextEl = document.getElementById("answerText");
const sqlBlockEl = document.getElementById("sqlBlock");
const tableBodyEl = document.getElementById("tableBody");
const toggleSqlBtn = document.getElementById("toggleSqlBtn");
const copySqlBtn = document.getElementById("copySqlBtn");
const suggestionsEl = document.getElementById("suggestions");

let lastSql = "";

function formatFee(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return `INR ${num.toLocaleString("en-IN")}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderTable(rows) {
  tableBodyEl.innerHTML = "";
  if (!rows || rows.length === 0) {
    tableBodyEl.innerHTML = `
      <tr><td colspan="8" class="py-4 text-slate-500">No matching rows found. Try widening your filters.</td></tr>
    `;
    return;
  }

  rows.forEach((row) => {
    tableBodyEl.innerHTML += `
      <tr class="border-b border-slate-100">
        <td class="py-2 pr-4">${escapeHtml(row.college_name || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.state || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.category || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.air_rank || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.score || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(formatFee(row.fee))}</td>
        <td class="py-2 pr-4">${escapeHtml(row.course || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.round || "-")}</td>
      </tr>
    `;
  });
}

async function askQuestion() {
  const question = questionEl.value.trim();
  if (!question) return;

  errorBoxEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  loadingEl.classList.remove("hidden");
  askBtn.disabled = true;
  askBtn.classList.add("opacity-60", "cursor-not-allowed");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Request failed.");
    }

    const data = await response.json();
    lastSql = data.sql || "";
    userQueryEl.textContent = question;
    answerTextEl.textContent = data.answer || "No explanation generated.";
    sqlBlockEl.textContent = lastSql;
    renderTable(data.data || []);
    resultEl.classList.remove("hidden");
  } catch (error) {
    errorBoxEl.textContent = error.message || "Something went wrong.";
    errorBoxEl.classList.remove("hidden");
  } finally {
    loadingEl.classList.add("hidden");
    askBtn.disabled = false;
    askBtn.classList.remove("opacity-60", "cursor-not-allowed");
  }
}

async function loadSuggestions() {
  try {
    const res = await fetch("/suggestions");
    const payload = await res.json();
    const items = payload.suggestions || [];
    suggestionsEl.innerHTML = "";

    items.forEach((q) => {
      const btn = document.createElement("button");
      btn.className = "text-sm px-3 py-1 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700";
      btn.textContent = q;
      btn.addEventListener("click", () => {
        questionEl.value = q;
        askQuestion();
      });
      suggestionsEl.appendChild(btn);
    });
  } catch (_) {
    // Suggestions are optional.
  }
}

askBtn.addEventListener("click", askQuestion);
questionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") askQuestion();
});

toggleSqlBtn.addEventListener("click", () => {
  sqlBlockEl.classList.toggle("hidden");
});

copySqlBtn.addEventListener("click", async () => {
  if (!lastSql) return;
  try {
    await navigator.clipboard.writeText(lastSql);
    copySqlBtn.textContent = "Copied!";
    setTimeout(() => {
      copySqlBtn.textContent = "Copy SQL";
    }, 1000);
  } catch (_) {
    // Clipboard can fail in some browser contexts.
  }
});

loadSuggestions();
