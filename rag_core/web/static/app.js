/* ──────────────────────────────────────────────────────────
   Advanced RAG Chat — Application Logic
   ────────────────────────────────────────────────────────── */

(function () {
  "use strict";

  // ── DOM References ────────────────────────────────────
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const newChatBtn = document.getElementById("newChatBtn");
  const sessionList = document.getElementById("sessionList");
  const chatTitle = document.getElementById("chatTitle");
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const chatForm = document.getElementById("chatForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const clearBtn = document.getElementById("clearBtn");

  // ── State ─────────────────────────────────────────────
  let currentSessionId = null;
  let isStreaming = false;
  let sessions = {}; // { id: { title, messages: [{role, content, citations}] } }

  // ── Sidebar toggle (mobile) ───────────────────────────
  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
  });

  // Close sidebar when clicking outside on mobile
  document.addEventListener("click", (e) => {
    if (window.innerWidth <= 768 &&
      sidebar.classList.contains("open") &&
      !sidebar.contains(e.target) &&
      !sidebarToggle.contains(e.target)) {
      sidebar.classList.remove("open");
    }
  });

  // ── Input handling ────────────────────────────────────
  messageInput.addEventListener("input", () => {
    sendBtn.disabled = !messageInput.value.trim() || isStreaming;
    autoResize(messageInput);
  });

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (messageInput.value.trim() && !isStreaming) {
        chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    }
  });

  function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  // ── Suggestion cards ──────────────────────────────────
  document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.addEventListener("click", () => {
      const query = card.dataset.query;
      if (query) {
        messageInput.value = query;
        sendBtn.disabled = false;
        chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });
  });

  // ── New chat ──────────────────────────────────────────
  newChatBtn.addEventListener("click", startNewConversation);
  clearBtn.addEventListener("click", () => {
    if (currentSessionId) {
      resetSession(currentSessionId);
    }
    startNewConversation();
  });

  function startNewConversation() {
    currentSessionId = null;
    chatTitle.textContent = "New Conversation";
    renderMessages([]);
    welcomeScreen.style.display = "";
    messageInput.focus();
    updateSessionList();
  }

  // ── Chat form submission ──────────────────────────────
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = messageInput.value.trim();
    if (!query || isStreaming) return;

    // Hide welcome screen
    welcomeScreen.style.display = "none";

    // If new conversation, create session
    if (!currentSessionId) {
      currentSessionId = crypto.randomUUID();
      sessions[currentSessionId] = {
        title: query.slice(0, 50) + (query.length > 50 ? "…" : ""),
        messages: [],
      };
      chatTitle.textContent = sessions[currentSessionId].title;
      updateSessionList();
    }

    // Add user message
    const session = sessions[currentSessionId];
    session.messages.push({ role: "user", content: query });
    renderMessages(session.messages);

    // Clear input
    messageInput.value = "";
    messageInput.style.height = "auto";
    sendBtn.disabled = true;

    // Stream response
    await streamChat(query, currentSessionId);
  });

  // ── Stream chat SSE ───────────────────────────────────
  async function streamChat(query, sessionId) {
    isStreaming = true;
    sendBtn.disabled = true;

    // Show thinking indicator
    const thinkingEl = appendThinking();
    scrollToBottom();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query, session_id: sessionId }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // Keep incomplete line

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();

          if (payload === "[DONE]") {
            break;
          }

          try {
            const data = JSON.parse(payload);

            if (data.type === "session") {
              // Update session ID if server assigned one
              if (data.session_id && data.session_id !== sessionId) {
                sessions[data.session_id] = sessions[sessionId];
                delete sessions[sessionId];
                currentSessionId = data.session_id;
                sessionId = data.session_id;
                updateSessionList();
              }
            }

            if (data.type === "status") {
              thinkingEl.querySelector(".thinking-text").textContent = data.content;
            }

            if (data.type === "answer") {
              // Remove thinking indicator
              thinkingEl.remove();

              // Add assistant message with citations
              const session = sessions[sessionId];
              if (session) {
                session.messages.push({
                  role: "assistant",
                  content: data.content,
                  citations: data.citations || [],
                });
                renderMessages(session.messages);
              }
            }

            if (data.type === "error") {
              thinkingEl.remove();
              const session = sessions[sessionId];
              if (session) {
                session.messages.push({
                  role: "assistant",
                  content: data.content,
                  isError: true,
                });
                renderMessages(session.messages);
              }
            }
          } catch (parseErr) {
            // Ignore JSON parse errors for incomplete lines
          }
        }
      }
    } catch (err) {
      thinkingEl.remove();
      const session = sessions[sessionId];
      if (session) {
        session.messages.push({
          role: "assistant",
          content: "Connection error: " + err.message,
          isError: true,
        });
        renderMessages(session.messages);
      }
    } finally {
      isStreaming = false;
      sendBtn.disabled = !messageInput.value.trim();
      scrollToBottom();
    }
  }

  // ── Render messages ───────────────────────────────────
  function renderMessages(messages) {
    // Clear everything except welcome screen
    const children = Array.from(messagesContainer.children);
    children.forEach((child) => {
      if (child.id !== "welcomeScreen") child.remove();
    });

    messages.forEach((msg, i) => {
      const row = document.createElement("div");
      row.className = `message-row ${msg.role}`;
      row.style.animationDelay = `${Math.min(i * 0.05, 0.3)}s`;

      const wrapper = document.createElement("div");
      wrapper.className = "message-content-wrapper";

      const group = document.createElement("div");
      group.className = "message-group";

      // Avatar
      const avatar = document.createElement("div");
      avatar.className = `avatar ${msg.role === "user" ? "user-avatar" : "assistant-avatar"}`;
      avatar.textContent = msg.role === "user" ? "U" : "AI";

      // Bubble
      const bubble = document.createElement("div");
      bubble.className = `message-bubble${msg.isError ? " error-bubble" : ""}`;

      // Render content with inline citation highlights
      const citations = msg.citations || [];
      bubble.innerHTML = renderMarkdownWithCitations(msg.content, citations);

      // Render citations footer if we have any
      if (citations.length > 0) {
        const citationFooter = buildCitationFooter(citations);
        bubble.appendChild(citationFooter);
      }

      group.appendChild(avatar);
      group.appendChild(bubble);
      wrapper.appendChild(group);
      row.appendChild(wrapper);
      messagesContainer.appendChild(row);
    });

    scrollToBottom();
  }

  // ── Citation rendering ────────────────────────────────

  /**
   * Build the citation reference footer section.
   * Each citation: { document_index, file_name, page_number }
   */
  function buildCitationFooter(citations) {
    const section = document.createElement("details");
    section.className = "citations-section";

    const title = document.createElement("summary");
    title.className = "citations-title";
    title.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
        <polyline points="10 9 9 9 8 9"/>
      </svg>
      <span>References</span>
      <span class="citations-count">${escapeHtml(String(citations.length))}</span>
    `;
    section.appendChild(title);

    const list = document.createElement("div");
    list.className = "citations-list";

    citations.forEach((cit) => {
      const idx = cit.document_index ?? "?";
      const file = cit.file_name ?? "unknown";
      const page = cit.page_number;
      const sourceUrl = String(cit.source_url || "").trim();
      const sourceHref = normalizeCitationHref(sourceUrl);

      const card = document.createElement(sourceHref ? "a" : "div");
      card.className = sourceHref ? "citation-card citation-card-link" : "citation-card";
      if (sourceHref) {
        card.href = sourceHref;
        card.target = "_blank";
        card.rel = "noopener noreferrer";
      }

      card.innerHTML = `
        <span class="citation-badge">${escapeHtml(String(idx))}</span>
        <div class="citation-details">
          <span class="citation-filename">${escapeHtml(file)}</span>
          ${page != null && page !== 0 ? `<span class="citation-page">Page ${escapeHtml(String(page))}</span>` : ""}
        </div>
      `;

      list.appendChild(card);
    });

    section.appendChild(list);
    return section;
  }

  /**
   * Render markdown text and highlight inline citation markers like [1], [2].
   */
  function renderMarkdownWithCitations(text, citations) {
    if (!text) return "";

    const normalizedText = normalizeMarkdownInput(text);

    // Build a lookup for quick citation existence check
    const citationIndices = new Set(citations.map((c) => c.document_index));

    // Parse the full markdown to HTML using marked.js
    // This provides robust support for tables, lists, bold, etc.
    let html = void 0;
    if (typeof marked !== "undefined") {
      html = marked.parse(normalizedText);
    } else {
      html = escapeHtml(normalizedText).replace(/\n/g, "<br>");
    }

    html = wrapTablesInContainers(html);
    html = highlightInlineCitations(html, citationIndices, citations);

    return `<div class="markdown-content">${html}</div>`;
  }

  function configureMarkdownParser() {
    if (typeof marked === "undefined") return;

    marked.setOptions({
      gfm: true,
      breaks: true,
    });
  }

  function normalizeMarkdownInput(text) {
    let normalized = String(text).replace(/\r\n?/g, "\n");

    if (!normalized.includes("\n") && normalized.includes("\\n")) {
      normalized = normalized.replace(/\\n/g, "\n");
    }

    const fencedMarkdown = normalized.match(/^\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$/i);
    if (fencedMarkdown) {
      normalized = fencedMarkdown[1];
    }

    // Tables are easier for parsers to detect when preceded by a blank line.
    normalized = normalized.replace(
      /([^\n])\n(\|[^\n]+\|)\n(\|[\s:|-]+\|)/g,
      "$1\n\n$2\n$3"
    );

    return normalized;
  }

  function wrapTablesInContainers(html) {
    if (!html || !html.includes("<table")) return html;

    const root = document.createElement("div");
    root.innerHTML = html;

    root.querySelectorAll("table").forEach((tableEl) => {
      if (tableEl.parentElement && tableEl.parentElement.classList.contains("markdown-table-wrap")) {
        return;
      }

      const wrapper = document.createElement("div");
      wrapper.className = "markdown-table-wrap";
      tableEl.parentNode.insertBefore(wrapper, tableEl);
      wrapper.appendChild(tableEl);
    });

    return root.innerHTML;
  }

  function highlightInlineCitations(html, citationIndices, citations) {
    if (!html || citationIndices.size === 0) return html;

    const root = document.createElement("div");
    root.innerHTML = html;

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const candidateNodes = [];

    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (!node.nodeValue || !node.nodeValue.includes("[")) continue;
      if (!/\[(\d+)\]/.test(node.nodeValue)) continue;
      if (node.parentElement && (node.parentElement.closest("code") || node.parentElement.closest("pre"))) {
        continue;
      }
      candidateNodes.push(node);
    }

    candidateNodes.forEach((node) => {
      const content = node.nodeValue;
      const pattern = /\[(\d+)\]/g;
      let lastIndex = 0;
      let hasMatch = false;
      const fragment = document.createDocumentFragment();

      for (let match = pattern.exec(content); match; match = pattern.exec(content)) {
        const [fullMatch, indexStr] = match;
        const startIndex = match.index;
        const indexNum = Number.parseInt(indexStr, 10);

        if (startIndex > lastIndex) {
          fragment.append(document.createTextNode(content.slice(lastIndex, startIndex)));
        }

        if (citationIndices.has(indexNum)) {
          const citation = citations.find((c) => c.document_index === indexNum);
          const sourceUrl = String(citation?.source_url || "").trim();
          const sourceHref = normalizeCitationHref(sourceUrl);

          const badge = document.createElement(sourceHref ? "a" : "span");
          badge.className = sourceHref ? "inline-citation is-link" : "inline-citation";
          badge.dataset.citation = String(indexNum);
          badge.textContent = indexStr;
          if (sourceHref) {
            badge.href = sourceHref;
            badge.target = "_blank";
            badge.rel = "noopener noreferrer";
          }
          fragment.append(badge);
        } else {
          fragment.append(document.createTextNode(fullMatch));
        }

        lastIndex = startIndex + fullMatch.length;
        hasMatch = true;
      }

      if (!hasMatch) return;

      if (lastIndex < content.length) {
        fragment.append(document.createTextNode(content.slice(lastIndex)));
      }

      node.parentNode.replaceChild(fragment, node);
    });

    return root.innerHTML;
  }

  // ── Thinking indicator ────────────────────────────────
  function appendThinking() {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    row.innerHTML = `
      <div class="message-content-wrapper">
        <div class="message-group">
          <div class="avatar assistant-avatar">AI</div>
          <div class="thinking-indicator">
            <div class="thinking-dots">
              <span></span><span></span><span></span>
            </div>
            <span class="thinking-text">Thinking...</span>
          </div>
        </div>
      </div>
    `;

    messagesContainer.appendChild(row);
    scrollToBottom();
    return row;
  }

  // ── Session list sidebar ──────────────────────────────
  function updateSessionList() {
    // Remove old items
    sessionList.querySelectorAll(".session-item").forEach((el) => el.remove());

    Object.entries(sessions).forEach(([id, sess]) => {
      const item = document.createElement("div");
      item.className = `session-item${id === currentSessionId ? " active" : ""}`;
      item.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
        </svg>
        <span>${escapeHtml(sess.title)}</span>
      `;
      item.addEventListener("click", () => switchSession(id));
      sessionList.appendChild(item);
    });
  }

  function switchSession(id) {
    currentSessionId = id;
    const session = sessions[id];
    if (!session) return;

    chatTitle.textContent = session.title;
    welcomeScreen.style.display = "none";
    renderMessages(session.messages);
    updateSessionList();

    // Close mobile sidebar
    sidebar.classList.remove("open");
  }

  // ── Reset session on server ───────────────────────────
  async function resetSession(sessionId) {
    try {
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (e) { /* ignore */ }

    delete sessions[sessionId];
    updateSessionList();
  }

  // ── Helpers ───────────────────────────────────────────
  function scrollToBottom() {
    requestAnimationFrame(() => {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
  }

  function normalizeCitationHref(sourceUrl) {
    if (!sourceUrl) return "";

    try {
      const parsed = new URL(sourceUrl);
      if (["http:", "https:", "file:"].includes(parsed.protocol)) {
        return parsed.href;
      }
      return "";
    } catch (_err) {
      // Fall back to a local file URL for absolute filesystem paths.
    }

    if (sourceUrl.startsWith("/")) {
      return `file://${encodeURI(sourceUrl)}`;
    }

    if (/^[a-zA-Z]:[\\/]/.test(sourceUrl)) {
      return `file:///${encodeURI(sourceUrl.replace(/\\/g, "/"))}`;
    }

    return "";
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Init ──────────────────────────────────────────────
  configureMarkdownParser();
  startNewConversation();
})();
