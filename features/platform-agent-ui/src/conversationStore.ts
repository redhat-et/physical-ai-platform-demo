export interface Message {
  role: "user" | "assistant";
  content: string;
  toolsCalled?: string[];
  media?: { kind: "image" | "video"; url: string };
}

export const API_BASE = "https://platform-agent-api-physical-ai.apps.emerg.pcbk.p1.openshiftapps.com";

const MESSAGES_STORAGE_KEY = "platform-agent-messages";
const THREAD_ID_STORAGE_KEY = "platform-agent-thread-id";

interface ConversationState {
  messages: Message[];
  loading: boolean;
  statusText: string;
}

const loadStoredMessages = (): Message[] => {
  try {
    const raw = sessionStorage.getItem(MESSAGES_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

const loadOrCreateThreadId = (): string => {
  try {
    const existing = sessionStorage.getItem(THREAD_ID_STORAGE_KEY);
    if (existing) return existing;
  } catch {
    // storage unavailable -- fall through to an ungenerated, unpersisted id
  }
  const id = crypto.randomUUID();
  try {
    sessionStorage.setItem(THREAD_ID_STORAGE_KEY, id);
  } catch {
    // storage unavailable (e.g. private browsing quota) -- id won't persist
    // across reloads, but the current session still works
  }
  return id;
};

// Module-level so it survives the component unmounting/remounting as the
// dashboard's router navigates away from and back to this tab — an in-flight
// request keeps running and updating this state either way.
let state: ConversationState = {
  messages: loadStoredMessages(),
  loading: false,
  statusText: "",
};

let threadId: string = loadOrCreateThreadId();

// Bumped on every clearMessages() so an in-flight sendMessage() from before
// the reset can tell it's stale -- without this, a response that arrives
// after the user clears the chat gets appended onto the now-empty
// conversation, and its `finally` block would clobber the loading/statusText
// state of whatever new message the user sent immediately after clearing.
let conversationGeneration = 0;

const listeners = new Set<() => void>();

const setState = (patch: Partial<ConversationState>) => {
  state = { ...state, ...patch };
  if (patch.messages) {
    try {
      sessionStorage.setItem(MESSAGES_STORAGE_KEY, JSON.stringify(state.messages));
    } catch {
      // storage unavailable (e.g. private browsing quota) — history just won't persist
    }
  }
  listeners.forEach((listener) => listener());
};

export const subscribe = (listener: () => void): (() => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

export const getSnapshot = (): ConversationState => state;

export const clearMessages = (): void => {
  // A reset means a genuinely new conversation server-side too -- otherwise
  // the checkpointer would keep replaying the old thread's history (and
  // whichever skill was last active in it) into a chat that looks empty.
  conversationGeneration += 1;
  threadId = crypto.randomUUID();
  try {
    sessionStorage.setItem(THREAD_ID_STORAGE_KEY, threadId);
  } catch {
    // storage unavailable -- the new id still applies for the rest of this session
  }
  setState({ messages: [] });
};

export const sendMessage = async (text: string): Promise<void> => {
  if (!text || state.loading) return;

  // Captured once, up front -- if clearMessages() runs while this request is
  // still in flight, conversationGeneration moves on and every check below
  // sees a mismatch, so a late response never lands in the reset (or a
  // subsequent, different) conversation.
  const generation = conversationGeneration;
  const isStale = () => generation !== conversationGeneration;

  setState({ messages: [...state.messages, { role: "user", content: text }], loading: true, statusText: "" });

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, thread_id: threadId }),
      signal: AbortSignal.timeout(300000),
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response stream");

    const decoder = new TextDecoder();
    let buffer = "";
    let pendingMedia: Message["media"] | undefined;

    while (true) {
      if (isStale()) {
        await reader.cancel();
        return;
      }

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") continue;

        try {
          const data = JSON.parse(payload);
          if (isStale()) continue;
          if (data.status) {
            setState({ statusText: data.status });
          } else if (data.media) {
            pendingMedia = data.media;
          } else if (data.response) {
            setState({
              messages: [
                ...state.messages,
                {
                  role: "assistant",
                  content: data.response,
                  toolsCalled: data.tools_called || [],
                  media: pendingMedia,
                },
              ],
            });
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  } catch (err) {
    if (isStale()) return;
    const msg = err instanceof Error ? err.message : "Unknown error";
    setState({
      messages: [
        ...state.messages,
        { role: "assistant", content: `Error: could not reach the agent (${msg}).` },
      ],
    });
  } finally {
    if (!isStale()) {
      setState({ loading: false, statusText: "" });
    }
  }
};
