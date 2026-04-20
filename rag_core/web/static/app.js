/* ──────────────────────────────────────────────────────────
   Advanced RAG Chat — Application Logic
   ────────────────────────────────────────────────────────── */

(function () {
  "use strict";

  // ── DOM References ────────────────────────────────────
  const sidebar        = document.getElementById("sidebar");
  const sidebarToggle  = document.getElementById("sidebarToggle");
  const newChatBtn     = document.getElementById("newChatBtn");
  const sessionList    = document.getElementById("sessionList");
  const chatTitle      = document.getElementById("chatTitle");
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeScreen  = document.getElementById("welcomeScreen");
  const chatForm       = document.getElementById("chatForm");
  const messageInput   = document.getElementById("messageInput");
  const sendBtn        = document.getElementById("sendBtn");
  const clearBtn       = document.getElementById("clearBtn");

  // ── State ─────────────────────────────────────────────
  let currentSessionId = null;
  let isStreaming       = false;
  let sessions         = {}; // { id: { title, messages: [{role, content, sources}] } }

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

              // Add assistant message
              const session = sessions[sessionId];
              if (session) {
                session.messages.push({
                  role: "assistant",
                  content: data.content,
                  sources: data.sources || [],
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
      bubble.innerHTML = renderMarkdown(msg.content);

      // Sources
      if (msg.sources && msg.sources.length > 0) {
        const sourcesEl = document.createElement("div");
        sourcesEl.className = "sources-section";
        sourcesEl.innerHTML = `
          <div class="sources-title">📎 Sources</div>
          <div>${msg.sources.map((s) => `<span class="source-tag">${escapeHtml(typeof s === "string" ? s : s.source || s.file || JSON.stringify(s))}</span>`).join("")}</div>
        `;
        bubble.appendChild(sourcesEl);
      }

      group.appendChild(avatar);
      group.appendChild(bubble);
      wrapper.appendChild(group);
      row.appendChild(wrapper);
      messagesContainer.appendChild(row);
    });

    scrollToBottom();
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

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  /** Very lightweight markdown → HTML (bold, italic, code, lists, paragraphs). */
  function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Unordered lists
    html = html.replace(/^[-•] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
    // Blockquote
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
    // Paragraphs (double newlines)
    html = html.replace(/\n\n/g, '</p><p>');
    // Single newlines → <br>
    html = html.replace(/\n/g, '<br>');

    return `<p>${html}</p>`;
  }

  // ── Init ──────────────────────────────────────────────
  startNewConversation();
})();
