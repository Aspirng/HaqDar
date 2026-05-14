"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

type Source = {
  law: string;
  section: string;
};

type Message = {
  id: string;
  role: "user" | "bot" | "error";
  content: string;
  sources?: Source[];
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "bot",
      content: "Hello! I am the HaqDar Legal Assistant. How can I help you understand Pakistani Law today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput("");
    
    // Add user message
    const newMsgId = Date.now().toString();
    setMessages((prev) => [
      ...prev,
      { id: newMsgId, role: "user", content: userQuery },
    ]);
    
    setIsLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userQuery }),
      });

      const data = await res.json();

      if (!res.ok) {
        if (res.status === 404 && data.error && data.error.includes("Vector store not found")) {
          throw new Error("The legal database has not been built yet. Please ask the administrator to run the ingestion script.");
        }
        throw new Error(data.error || "An error occurred connecting to the backend.");
      }

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "bot",
          content: data.answer,
          sources: data.sources,
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "error",
          content: err.message,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <header className="header glass-panel">
        <h1>HaqDar</h1>
        <p>AI Legal Assistant</p>
      </header>

      <div className="chat-container glass-panel" style={{ borderRadius: 0, borderTop: 'none', borderBottom: 'none' }}>
        {messages.map((msg) => (
          <div key={msg.id} className={`message-wrapper ${msg.role}`}>
            <div className="message-bubble">
              {msg.role === "bot" ? (
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              ) : (
                <p>{msg.content}</p>
              )}
              
              {/* Sources Rendering */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources-container">
                  <div className="sources-title">Sources Retrieved</div>
                  {msg.sources.map((src, idx) => (
                    <span key={idx} className="source-badge">
                      {src.law} - Sec {src.section}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message-wrapper bot">
            <div className="message-bubble typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="input-container glass-panel" onSubmit={handleSubmit}>
        <input
          type="text"
          className="chat-input"
          placeholder="Ask a legal question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="send-button" disabled={isLoading || !input.trim()}>
          {isLoading ? "..." : "Send"}
        </button>
      </form>
    </>
  );
}
