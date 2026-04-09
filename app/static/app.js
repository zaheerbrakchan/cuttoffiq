const questionEl = document.getElementById("question");
const askBtn = document.getElementById("askBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const loadingEl = document.getElementById("loading");
const dataPanelEl = document.getElementById("dataPanel");
const errorBoxEl = document.getElementById("errorBox");
const chatThreadEl = document.getElementById("chatThread");
const sqlBlockEl = document.getElementById("sqlBlock");
const tableBodyEl = document.getElementById("tableBody");
const toggleSqlBtn = document.getElementById("toggleSqlBtn");
const copySqlBtn = document.getElementById("copySqlBtn");
const suggestionsEl = document.getElementById("suggestions");

let lastSql = "";
let currentMessages = [];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderChat(messages) {
  if (!chatThreadEl) return;
  chatThreadEl.innerHTML = "";

  if (!messages.length) {
    const p = document.createElement("p");
    p.className = "text-sm text-slate-500 text-center py-10 px-2";
    p.innerHTML =
      "Start a conversation — the assistant may ask for your <strong>category</strong>, <strong>state or MCC</strong>, and <strong>college type</strong> before searching cutoffs.";
    chatThreadEl.appendChild(p);
    return;
  }

  messages.forEach((m) => {
    const isUser = m.role === "user";
    const wrap = document.createElement("div");
    wrap.className = `flex ${isUser ? "justify-end" : "justify-start"}`;
    const bubble = document.createElement("div");
    bubble.className = [
      "max-w-[92%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm",
      isUser
        ? "bg-indigo-600 text-white rounded-br-md"
        : "bg-white border border-slate-200 text-slate-800 rounded-bl-md",
    ].join(" ");
    if (!isUser && m.needsClarification) {
      bubble.classList.add("border-amber-300", "bg-amber-50/90");
    }
    bubble.innerHTML = `<p class="text-[10px] uppercase tracking-wide opacity-70 mb-1">${isUser ? "You" : "Counsellor"}</p><div class="whitespace-pre-wrap">${escapeHtml(m.content)}</div>`;
    wrap.appendChild(bubble);
    chatThreadEl.appendChild(wrap);
  });

  chatThreadEl.scrollTop = chatThreadEl.scrollHeight;
}

function renderTable(rows, suppressed) {
  tableBodyEl.innerHTML = "";
  if (suppressed) {
    tableBodyEl.innerHTML = `
      <tr><td colspan="9" class="py-4 text-slate-500">Cutoff results appear here after a full search (category, state or MCC, and college type).</td></tr>
    `;
    return;
  }
  if (!rows || rows.length === 0) {
    tableBodyEl.innerHTML = `
      <tr><td colspan="9" class="py-4 text-slate-500">No matching rows found. Try widening your filters.</td></tr>
    `;
    return;
  }

  rows.forEach((row) => {
    tableBodyEl.innerHTML += `
      <tr class="border-b border-slate-100">
        <td class="py-2 pr-4">${escapeHtml(row.college_name || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.state || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.college_type || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.category || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.air_rank || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.score || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.quota || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.course || "-")}</td>
        <td class="py-2 pr-4">${escapeHtml(row.round || "-")}</td>
      </tr>
    `;
  });
}

async function askQuestion() {
  const text = questionEl.value.trim();
  if (!text) return;

  errorBoxEl.classList.add("hidden");
  loadingEl.classList.remove("hidden");
  askBtn.disabled = true;
  askBtn.classList.add("opacity-60", "cursor-not-allowed");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text,
        messages: currentMessages,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Request failed.");
    }

    const data = await response.json();
    lastSql = data.sql || "";
    const needsClarification = data.needs_clarification === true;
    const showDataTables = Boolean(lastSql);

    const assistantText = data.answer || "No explanation generated.";
    const nextMessages = [
      ...currentMessages,
      { role: "user", content: text },
      {
        role: "assistant",
        content: assistantText,
        ...(needsClarification ? { needsClarification: true } : {}),
      },
    ];
    currentMessages = nextMessages.slice(-8);
    renderChat(currentMessages);

    questionEl.value = "";

    if (dataPanelEl) {
      dataPanelEl.classList.toggle("hidden", !showDataTables);
    }
    sqlBlockEl.textContent =
      lastSql || "— (no query run — share category, state/MCC, and college type first)";
    sqlBlockEl.classList.add("hidden");

    renderTable(data.data || [], !showDataTables);
  } catch (error) {
    errorBoxEl.textContent = error.message || "Something went wrong.";
    errorBoxEl.classList.remove("hidden");
  } finally {
    loadingEl.classList.add("hidden");
    askBtn.disabled = false;
    askBtn.classList.remove("opacity-60", "cursor-not-allowed");
    questionEl.focus();
  }
}

async function clearChat() {
  try {
    await fetch("/chat/clear", { method: "POST" });
  } catch (_) {
    // Best effort clear.
  }
  currentMessages = [];
  lastSql = "";
  renderChat([]);
  if (dataPanelEl) dataPanelEl.classList.add("hidden");
  sqlBlockEl.textContent = "";
  renderTable([], true);
  questionEl.focus();
}

async function loadServerChatContext() {
  try {
    const res = await fetch("/chat/context");
    if (!res.ok) return;
    const payload = await res.json();
    const rows = Array.isArray(payload.recent_chats) ? payload.recent_chats : [];
    currentMessages = rows
      .filter(
        (m) =>
          m &&
          (m.role === "user" || m.role === "assistant") &&
          typeof m.content === "string"
      )
      .slice(-8);
    renderChat(currentMessages);
  } catch (_) {
    // Fallback to empty chat if context endpoint fails.
    currentMessages = [];
    renderChat([]);
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
      btn.type = "button";
      btn.className =
        "text-sm px-3 py-1 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700";
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
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion();
  }
});

clearChatBtn.addEventListener("click", clearChat);

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
loadServerChatContext();
