import React, { useState, useRef, useEffect } from "react";
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

interface Message {
  role: "user" | "assistant";
  content: string;
}

const PlatformAgent: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const res = await fetch("https://platform-agent-api-physical-ai.apps.emerg.pcbk.p1.openshiftapps.com/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response || "No response from agent." },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: could not reach the agent (${msg}). Tool-calling requests may take up to 60s — please try again.` },
      ]);
    } finally {
      setLoading(false);
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
                      color: "var(--pf-t--global--color--nonstatus--gray--default)",
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
                      {msg.content}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div style={{ marginBottom: "0.75rem" }}>
                    <Spinner size="md" />
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
    </>
  );
};

export default PlatformAgent;
