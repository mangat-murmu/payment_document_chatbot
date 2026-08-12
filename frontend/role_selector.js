const headers = () => ({
  "Content-Type": "application/json",
  "X-User-Role": document.querySelector("#role").value,
  "X-User-Id": "browser-demo",
});
const thread = document.querySelector("#thread");
const query = document.querySelector("#query");
const welcome = document.querySelector("#welcome");
const roleSelect = document.querySelector("#role");
const suggestions = document.querySelector(".suggestions");
let streamFollowEnabled = false;
let streamActive = false;
let chatId = null;
let lastResponseId = null;

const roleSuggestions = {
  product_lead: [
    {
      title: "UPI performance",
      subtitle: "Find transaction success trends",
      query: "What's the success rate of UPI transactions this month?",
    },
    {
      title: "Failure reasons",
      subtitle: "Rank top transaction drop-offs",
      query: "Which UPI failure reasons are hurting conversion the most?",
    },
    {
      title: "Merchant mix",
      subtitle: "Compare categories and value",
      query: "Which merchant categories have the highest UPI payment volume?",
    },
    {
      title: "Risk patterns",
      subtitle: "Spot risky transaction segments",
      query: "Are there suspicious transaction patterns by bank, state, or merchant category?",
    },
  ],
  tech_lead: [
    {
      title: "Integration health",
      subtitle: "Review API errors and latency",
      query: "Are there any API integration failures today?",
    },
    {
      title: "Latency outliers",
      subtitle: "Find slow banks and operations",
      query: "Which bank API operations have the highest latency this month?",
    },
    {
      title: "Incident traces",
      subtitle: "Inspect severe failures",
      query: "Show recent SEV1 or SEV2 bank integration failures with trace context.",
    },
    {
      title: "Reconciliation",
      subtitle: "Check mismatches and pending items",
      query: "Which bank integration logs show reconciliation mismatches?",
    },
  ],
  compliance_lead: [
    {
      title: "Circular impact",
      subtitle: "Summarize obligations",
      query: "Which recent UPI circulars create new compliance obligations for us?",
    },
    {
      title: "Implementation dates",
      subtitle: "Find deadlines and scope",
      query: "What compliance deadlines are mentioned in the uploaded audit documents?",
    },
    {
      title: "Control gaps",
      subtitle: "Review required safeguards",
      query: "What controls are required for UPI user information and API usage?",
    },
    {
      title: "Transaction limits",
      subtitle: "Check regulatory changes",
      query: "What UPI transaction limit changes should product and operations know about?",
    },
  ],
  bank_alliance_lead: [
    {
      title: "Bank partnership SLA",
      subtitle: "Review availability and exceptions",
      query: "How is our SLA performance with Bank X?",
    },
    {
      title: "Commercial terms",
      subtitle: "Compare fees and platform costs",
      query: "Compare commercial terms across bank partnership agreements.",
    },
    {
      title: "Escalations",
      subtitle: "Find obligations and timelines",
      query: "What escalation or incident-response obligations do our bank SLAs define?",
    },
    {
      title: "Bank reliability",
      subtitle: "Match logs against SLA promises",
      query: "Which bank partners have operational issues that may affect SLA commitments?",
    },
  ],
};

function setChatUrl(id) {
  const path = `/chats/${id}`;
  if (window.location.pathname !== path) {
    window.history.pushState({}, "", path);
  }
}
function resetChat(updateUrl = true) {
  chatId = null;
  lastResponseId = null;
  thread.innerHTML = "";
  welcome.hidden = false;
  thread.append(welcome);
  if (updateUrl && window.location.pathname !== "/") {
    window.history.pushState({}, "", "/");
  }
  loadRecentChats();
}

function isNearPageBottom() {
  return (
    window.innerHeight + window.scrollY >=
    document.documentElement.scrollHeight - 96
  );
}
function scrollToBottom() {
  window.scrollTo({
    top: document.documentElement.scrollHeight,
    behavior: "auto",
  });
}
function renderSuggestions() {
  const items = roleSuggestions[roleSelect.value] || roleSuggestions.product_lead;
  suggestions.replaceChildren(
    ...items.map((item) => {
      const button = document.createElement("button");
      button.className = "suggestion";
      button.type = "button";
      button.dataset.query = item.query;
      button.append(document.createTextNode(item.title));

      const subtitle = document.createElement("span");
      subtitle.textContent = item.subtitle;
      button.append(subtitle);
      return button;
    }),
  );
}
window.addEventListener(
  "scroll",
  () => {
    if (streamActive) streamFollowEnabled = isNearPageBottom();
  },
  { passive: true },
);

function followMessage(message) {
  if (streamFollowEnabled) scrollToBottom();
}

function escapeHtml(value) {
  return String(value).replace(
    /[&<>'"]/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[
        char
      ],
  );
}
function renderMarkdown(value) {
  if (!window.marked || !window.DOMPurify) return escapeHtml(value);
  const rendered = window.marked.parse(String(value), {
    breaks: true,
    gfm: true,
  });
  return window.DOMPurify.sanitize(rendered, {
    ADD_ATTR: ["target"],
  });
}
function citationMeta(citation) {
  if (citation.citation_type === "query") {
    return citation.document_type ? citation.document_type.replaceAll("_", " ") : "SQL";
  }
  const pageNumbers = citation.page_numbers || [];
  if (pageNumbers.length) return `[${pageNumbers.join(", ")}]`;
  if (citation.page_number) return `[${citation.page_number}]`;

  const indexNumbers =
    citation.index_numbers || citation.chunk_indexes || [];
  if (indexNumbers.length) return `[#${indexNumbers.join(", #")}]`;
  const chunkIndex = citation.chunk_index ?? citation.index_number ?? citation.version;
  if (chunkIndex !== null && chunkIndex !== undefined) return `[#${chunkIndex}]`;
  return "";
}
function uniqueCitations(citations = []) {
  const byDocument = new Map();
  citations.forEach((citation) => {
    const key =
      citation.citation_type === "query"
        ? `query:${citation.tool_name}:${citation.query}`
        : `document:${citation.document_id}`;
    if (!byDocument.has(key)) {
      byDocument.set(key, {
        ...citation,
        page_numbers: [],
        index_numbers: [],
        chunk_indexes: [],
      });
    }
    if (citation.citation_type === "query") return;
    const grouped = byDocument.get(key);
    grouped.score = Math.max(grouped.score || 0, citation.score || 0);
    const pages = citation.page_numbers || [citation.page_number];
    pages
      .filter((page) => page !== null && page !== undefined)
      .forEach((page) => {
        if (!grouped.page_numbers.includes(page)) grouped.page_numbers.push(page);
      });
    const indexes =
      citation.index_numbers ||
      citation.chunk_indexes ||
      [citation.index_number ?? citation.chunk_index ?? citation.version];
    indexes
      .filter((index) => index !== null && index !== undefined)
      .forEach((index) => {
        if (!grouped.index_numbers.includes(index)) grouped.index_numbers.push(index);
        if (!grouped.chunk_indexes.includes(index)) grouped.chunk_indexes.push(index);
      });
  });
  return [...byDocument.values()].map((citation) => ({
    ...citation,
    page_numbers: citation.page_numbers.sort((left, right) => left - right),
    index_numbers: citation.index_numbers.sort((left, right) => left - right),
    chunk_indexes: citation.chunk_indexes.sort((left, right) => left - right),
  }));
}
function renderCitations(citations = []) {
  if (!citations.length) return "";
  return `<div class="citations">${uniqueCitations(citations)
    .map((citation) => {
      const title = escapeHtml(
        citation.citation_type === "query"
          ? citation.query || citation.title || "SQL query"
          : citation.title || `Document ${citation.document_id}`,
      );
      const meta = citationMeta(citation);
      const download =
        citation.citation_type === "query"
          ? ""
          : `<a class="citation-download" href="/api/documents/${encodeURIComponent(citation.document_id)}/download" download title="Download source" aria-label="Download ${title}">↓</a>`;
      return `<span class="citation${citation.citation_type === "query" ? " is-query" : ""}" title="${title}"><span class="citation-title">${title}</span>${meta ? `<span class="citation-meta">${escapeHtml(meta)}</span>` : ""}${download}</span>`;
    })
    .join("")}</div>`;
}
function addMessage(kind, content, citations = [], shouldScroll = true) {
  welcome.hidden = true;
  const row = document.createElement("article");
  row.className = "message";
  const source = renderCitations(citations);
  const assistantIcon =
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 2.8c.6 5.3 3.1 7.8 8.4 8.4-5.3.6-7.8 3.1-8.4 8.4-.6-5.3-3.1-7.8-8.4-8.4 5.3-.6 7.8-3.1 8.4-8.4Z"/><path d="M18.7 15.9c.2 1.7 1 2.5 2.7 2.7-1.7.2-2.5 1-2.7 2.7-.2-1.7-1-2.5-2.7-2.7 1.7-.2 2.5-1 2.7-2.7Z"/></svg>';
  const bodyClass = kind === "assistant" ? "markdown" : "plain";
  const body = kind === "assistant" ? renderMarkdown(content) : escapeHtml(content);
  row.innerHTML = `<div class="avatar ${kind === "user" ? "user" : ""}">${kind === "user" ? "You" : assistantIcon}</div><div class="message-content"><div class="message-meta">${kind === "user" ? "You" : "PayDoc AI"}</div><div class="message-body ${bodyClass}">${body}</div>${source}</div>`;
  thread.append(row);
  if (shouldScroll) row.scrollIntoView({ behavior: "smooth", block: "end" });
  return row;
}
function messageText(content) {
  if (typeof content === "string") return content;
  return content.text || JSON.stringify(content);
}
function renderRecentChats(chats) {
  const list = document.querySelector("#recent-chats");
  list.replaceChildren();
  if (!chats.length) {
    const empty = document.createElement("span");
    empty.className = "subtle";
    empty.textContent = "No chats yet";
    list.append(empty);
    return;
  }
  chats.forEach((chat) => {
    const row = document.createElement("div");
    row.className = `recent-chat-row${chat.id === chatId ? " is-active" : ""}`;
    const button = document.createElement("button");
    button.className = "recent-chat";
    button.type = "button";
    button.title = chat.title || "Untitled chat";
    const title = document.createElement("span");
    title.className = "recent-chat-title";
    title.textContent = chat.title || "Untitled chat";
    button.append(title);
    button.onclick = () => loadChat(chat.id);
    const deleteButton = document.createElement("button");
    deleteButton.className = "delete-chat";
    deleteButton.type = "button";
    deleteButton.textContent = "×";
    deleteButton.title = `Delete ${chat.title || "chat"}`;
    deleteButton.setAttribute("aria-label", `Delete ${chat.title || "chat"}`);
    deleteButton.onclick = () => deleteChat(chat.id, chat.title);
    row.append(button, deleteButton);
    list.append(row);
  });
}
async function loadRecentChats() {
  try {
    const response = await fetch("/api/chats?limit=20");
    if (!response.ok) throw new Error("Could not load chat history.");
    const data = await response.json();
    renderRecentChats(data.chats || []);
  } catch {
    renderRecentChats([]);
  }
}
document.querySelector("#toggle-recent-chats").onclick = (event) => {
  const list = document.querySelector("#recent-chats");
  const collapsed = !list.hidden;
  list.hidden = collapsed;
  event.currentTarget.setAttribute("aria-expanded", String(!collapsed));
};
async function deleteChat(id, title) {
  if (streamActive) return;
  if (!window.confirm(`Delete ${title || "this chat"}?`)) return;

  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new Error("Could not delete this chat.");
    if (chatId === id) resetChat();
    else await loadRecentChats();
  } catch (error) {
    window.alert(error.message);
  }
}
async function loadChat(id, updateUrl = true) {
  if (streamActive) return;
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(id)}/messages`);
    if (!response.ok) throw new Error("Could not load this chat.");
    const data = await response.json();
    const messages = data.messages || [];
    chatId = id;
    if (updateUrl) setChatUrl(id);
    const latestAssistant = [...messages]
      .reverse()
      .find((message) => message.role === "assistant");
    lastResponseId = latestAssistant ? latestAssistant.id : null;
    thread.innerHTML = "";
    if (!messages.length) {
      welcome.hidden = false;
      thread.append(welcome);
    } else {
      messages.forEach((message) => {
        const citations = message.meta_data?.citations || [];
        addMessage(message.role, messageText(message.content), citations, false);
      });
      scrollToBottom();
    }
    await loadRecentChats();
  } catch {
    addMessage("assistant", "I could not load this chat. Please try again.");
  }
}
async function ask() {
  const text = query.value.trim();
  if (!text) return;
  streamFollowEnabled = true;
  streamActive = true;
  addMessage("user", text);
  scrollToBottom();
  query.value = "";
  document.querySelector("#ask").disabled = true;
  const pending = addMessage("assistant", "");
  const content = pending.querySelector(".message-body");
  let buffer = "";
  try {
    const r = await fetch("/api/chats", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({
        chat_id: chatId,
        input: { text },
        last_response_id: lastResponseId,
        stream: true,
      }),
    });
    if (!r.ok || !r.body) throw new Error("stream unavailable");
    const reader = r.body.getReader(),
      decoder = new TextDecoder();
    let remainder = "",
      citations = [];
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      remainder += decoder.decode(value, { stream: true });
      const records = remainder.split("\n\n");
      remainder = records.pop();
      for (const record of records) {
        if (!record.startsWith("data: ")) continue;
        const event = JSON.parse(record.slice(6));
        if (event.chat_id) {
          chatId = event.chat_id;
          setChatUrl(chatId);
        }
        if (event.response_id) lastResponseId = event.response_id;
        if (event.delta) {
          buffer += event.delta;
          content.innerHTML = renderMarkdown(buffer);
          followMessage(pending);
        }
        if (event.done) citations = event.citations || [];
      }
    }
    pending
      .querySelector(".message-content")
      .insertAdjacentHTML("beforeend", renderCitations(citations));
    await loadRecentChats();
  } catch {
    pending.remove();
    addMessage("assistant", "I could not reach the service. Please try again.");
  } finally {
    streamActive = false;
    document.querySelector("#ask").disabled = false;
    query.focus();
  }
}
document.querySelector("#ask").onclick = ask;
query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
});
suggestions.addEventListener("click", (event) => {
  const button = event.target.closest(".suggestion");
  if (!button) return;
  query.value = button.dataset.query;
  ask();
});
roleSelect.addEventListener("change", renderSuggestions);
document.querySelector("#new-chat").onclick = () => {
  resetChat();
  query.focus();
};
document.querySelector("#save").onclick = async () => {
  const value = query.value.trim();
  if (!value) return;
  await fetch("/api/searches", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ query: value }),
  });
};
window.addEventListener("popstate", () => {
  const match = window.location.pathname.match(/^\/chats\/(\d+)$/);
  if (match) loadChat(Number(match[1]), false);
  else resetChat(false);
});
(async () => {
  renderSuggestions();
  const match = window.location.pathname.match(/^\/chats\/(\d+)$/);
  if (match) await loadChat(Number(match[1]), false);
  else await loadRecentChats();
})();
