import React, { useState, useRef, useEffect } from "react";
import Markdown from "react-markdown";
import {
  PageSection,
  Title,
  TextInput,
  Button,
  Flex,
  FlexItem,
  Panel,
  PanelMain,
  PanelMainBody,
  Spinner,
  Label,
} from "@patternfly/react-core";
import ModelStartupGate, { ModelGateState } from "./ModelStartupGate";

interface Message {
  role: "user" | "assistant";
  content: string;
  toolsCalled?: string[];
  media?: { kind: "image" | "video"; url: string };
}

const API_BASE = "https://platform-agent-api-physical-ai.apps.emerg.pcbk.p1.openshiftapps.com";

const POLL_INTERVAL_MS = 10000;
const SLOW_WARNING_MS = 8 * 60 * 1000;

const PlatformAgent: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const [modelState, setModelState] = useState<ModelGateState | "ready">("checking");
  const [modelDetail, setModelDetail] = useState("");
  const [slowWarning, setSlowWarning] = useState<string | undefined>(undefined);
  const startedAtRef = useRef<number | null>(null);

  const checkModelStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/model/status`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const state: string = data.state;
      setModelDetail(data.detail || "");
      setModelState(state === "not_started" ? "not_ready" : (state as ModelGateState));
    } catch {
      setModelDetail("Could not reach the agent to check model status.");
      setModelState("error");
    }
  };

  useEffect(() => {
    checkModelStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (modelState !== "starting") {
      setSlowWarning(undefined);
      return;
    }
    if (startedAtRef.current === null) {
      startedAtRef.current = Date.now();
    }
    const id = setInterval(() => {
      if (startedAtRef.current && Date.now() - startedAtRef.current > SLOW_WARNING_MS) {
        setSlowWarning(
          "Still starting — this is taking longer than the ~5 minutes it usually takes."
        );
      }
      checkModelStatus();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelState]);

  const startModel = async () => {
    setModelState("starting");
    setModelDetail("Sending start request...");
    startedAtRef.current = Date.now();
    try {
      await fetch(`${API_BASE}/api/model/start`, { method: "POST" });
    } catch {
      // ignore — readiness polling will surface the real state
    }
    checkModelStatus();
  };

  const [statusText, setStatusText] = useState("");

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    const history = [...messages];
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    setStatusText("");

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history }),
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
            if (data.status) {
              setStatusText(data.status);
            } else if (data.media) {
              pendingMedia = data.media;
            } else if (data.response) {
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: data.response,
                  toolsCalled: data.tools_called || [],
                  media: pendingMedia,
                },
              ]);
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: could not reach the agent (${msg}).` },
      ]);
    } finally {
      setLoading(false);
      setStatusText("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <PageSection>
        <Flex
          direction={{ default: "row" }}
          alignItems={{ default: "alignItemsCenter" }}
          gap={{ default: "gapSm" }}
        >
          <FlexItem>
            <Title headingLevel="h1" size="xl">
              Platform Agent
            </Title>
          </FlexItem>
          <FlexItem>
            <Label color="orange">Experimental</Label>
          </FlexItem>
        </Flex>
      </PageSection>

      {modelState !== "ready" && (
        <PageSection isFilled>
          <ModelStartupGate
            state={modelState}
            detail={modelDetail}
            slowWarning={slowWarning}
            onStart={startModel}
            onRetry={startModel}
          />
        </PageSection>
      )}

      {modelState === "ready" && (
      <PageSection padding={{ default: "noPadding" }} isFilled>
        <div style={{ display: "flex", flexDirection: "column", height: "100%", padding: "0 1rem 1rem" }}>
          <Panel
            variant="bordered"
            style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}
          >
            <PanelMain style={{ flex: 1, overflow: "auto" }}>
              <PanelMainBody>
                {messages.length === 0 && (
                  <div
                    style={{
                      textAlign: "center",
                      padding: "3rem",
                      color: "var(--pf-t--global--text--color--subtle)",
                    }}
                  >
                    Ask me about your models — health, status, logs, or anything
                    about the platform.
                  </div>
                )}
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      justifyContent:
                        msg.role === "user" ? "flex-end" : "flex-start",
                      marginBottom: "0.75rem",
                    }}
                  >
                    <div
                      style={{
                        maxWidth: "70%",
                        padding: "0.75rem 1rem",
                        borderRadius: "0.5rem",
                        backgroundColor:
                          msg.role === "user"
                            ? "var(--pf-t--global--color--brand--default)"
                            : "var(--pf-t--global--background--color--secondary--default)",
                        color:
                          msg.role === "user"
                            ? "#fff"
                            : "var(--pf-t--global--text--color--regular)",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                      }}
                    >
                      {msg.role === "assistant" && msg.toolsCalled && msg.toolsCalled.length > 0 && (
                        <div style={{ marginBottom: "0.5rem", fontSize: "0.8rem", color: "var(--pf-t--global--text--color--subtle)" }}>
                          Used: {msg.toolsCalled.join(", ")}
                        </div>
                      )}
                      {msg.role === "assistant" ? (
                        <Markdown>{msg.content}</Markdown>
                      ) : (
                        msg.content
                      )}
                      {msg.media?.kind === "image" && (
                        <img
                          src={API_BASE + msg.media.url}
                          alt="Generated by model"
                          style={{ maxWidth: "100%", borderRadius: "0.5rem", marginTop: "0.5rem" }}
                        />
                      )}
                      {msg.media?.kind === "video" && (
                        <video
                          controls
                          src={API_BASE + msg.media.url}
                          style={{ maxWidth: "100%", borderRadius: "0.5rem", marginTop: "0.5rem" }}
                        />
                      )}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div style={{ marginBottom: "0.75rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <Spinner size="md" />
                    {statusText && (
                      <span style={{ color: "var(--pf-t--global--text--color--subtle)", fontSize: "0.85rem" }}>
                        {statusText}
                      </span>
                    )}
                  </div>
                )}
                <div ref={bottomRef} />
              </PanelMainBody>
            </PanelMain>
          </Panel>

          <Flex
            gap={{ default: "gapSm" }}
            style={{ marginTop: "1rem" }}
          >
            <FlexItem grow={{ default: "grow" }}>
              <TextInput
                value={input}
                onChange={(_e, val) => setInput(val)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your models..."
                aria-label="Chat input"
                isDisabled={loading}
              />
            </FlexItem>
            <FlexItem>
              <Button
                onClick={sendMessage}
                isDisabled={!input.trim() || loading}
              >
                Send
              </Button>
            </FlexItem>
          </Flex>
        </div>
      </PageSection>
      )}
    </>
  );
};

export default PlatformAgent;
