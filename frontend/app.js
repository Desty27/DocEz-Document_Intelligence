(() => {
  const apiBaseInput = document.getElementById("api-base");
  const sessionEl = document.getElementById("session-id");
  const messageInput = document.getElementById("message");
  const chatLog = document.getElementById("chat-log");

  const randomSession = () => `demo-${Math.random().toString(36).slice(2, 10)}`;
  let sessionId = randomSession();
  sessionEl.textContent = sessionId;

  function addMessage(role, content) {
    const wrapper = document.createElement("div");
    wrapper.className = `msg ${role}`;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = role.toUpperCase();
    const body = document.createElement("div");
    body.textContent = content;
    wrapper.appendChild(meta);
    wrapper.appendChild(body);
    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  async function pingHealth() {
    const base = apiBaseInput.value.replace(/\/$/, "");
    try {
      const res = await fetch(`${base}/health`);
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      addMessage("assistant", `Health: ${JSON.stringify(data)}`);
    } catch (err) {
      addMessage("assistant", `Health check failed: ${err}`);
    }
  }

  async function sendMessage() {
    const base = apiBaseInput.value.replace(/\/$/, "");
    const message = messageInput.value.trim();
    if (!message) return;
    addMessage("user", message);
    messageInput.value = "";
    try {
      const res = await fetch(`${base}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message })
      });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      addMessage("assistant", data.reply || JSON.stringify(data));
    } catch (err) {
      addMessage("assistant", `Error: ${err}`);
    }
  }

  document.getElementById("regen-session").addEventListener("click", () => {
    sessionId = randomSession();
    sessionEl.textContent = sessionId;
    chatLog.innerHTML = "";
  });

  document.getElementById("ping-health").addEventListener("click", pingHealth);
  document.getElementById("send").addEventListener("click", sendMessage);
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      sendMessage();
    }
  });
})();
