import { useState, useEffect } from "react";

interface SearchResult {
  id: number;
  content: string;
  distance: number;
  metadata: {
    author?: string;
    category?: string;
    date?: string;
    tags?: string[];
    [key: string]: any;
  };
}

interface Stats {
  total_documents: number;
  total_milvus_documents: number;
  categories: string[];
  has_faiss_index: boolean;
  has_milvus_collection: boolean;
}

interface Health {
  status: string;
  database: string;
  faiss_loaded: boolean;
  milvus_loaded: boolean;
  model_loaded: boolean;
}

interface Message {
  sender: "user" | "assistant";
  text: string;
  error?: boolean;
  sources?: SearchResult[];
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"search" | "rag">("search");

  // Database / Connection States
  const [stats, setStats] = useState<Stats | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  // Search Form States
  const [searchQuery, setSearchQuery] = useState("");
  const [searchEngine, setSearchEngine] = useState<"pgvector" | "faiss" | "milvus">("pgvector");
  const [searchLimit, setSearchLimit] = useState(5);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState("");

  // RAG QA States
  const [ragQueryText, setRagQueryText] = useState("");
  const [ragEngine, setRagEngine] = useState<"pgvector" | "faiss" | "milvus">("pgvector");
  const [ragLimit, setRagLimit] = useState(3);
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  // Fetch Health and Stats on mount
  useEffect(() => {
    fetchHealthAndStats();
  }, []);

  const fetchHealthAndStats = async () => {
    try {
      const healthResp = await fetch("/api/health");
      if (healthResp.ok) {
        const healthData = await healthResp.json();
        setHealth(healthData);
      }

      const statsResp = await fetch("/api/stats");
      if (statsResp.ok) {
        const statsData = await statsResp.json();
        setStats(statsData);
      }
    } catch (err) {
      console.error("Failed to fetch status:", err);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setSearchError("");
    setSearchResults([]);

    const endpoint =
      searchEngine === "faiss"
        ? "/api/search/faiss"
        : searchEngine === "milvus"
          ? "/api/search/milvus"
          : "/api/search";

    try {
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: searchQuery,
          limit: searchLimit,
          filters: null
        })
      });

      if (!resp.ok) {
        const errData = await resp.json();
        throw new Error(errData.detail || "Search request failed");
      }

      const data = await resp.json();
      setSearchResults(data);
    } catch (err: any) {
      setSearchError(err.message || "An error occurred during search.");
    } finally {
      setIsSearching(false);
    }
  };

  const handleRAGSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ragQueryText.trim() || isGenerating) return;

    const query = ragQueryText;
    setRagQueryText("");

    // Append User Message to Chat
    const userMsg: Message = { sender: "user", text: query };
    setChatHistory(prev => [...prev, userMsg]);
    setIsGenerating(true);

    try {
      const resp = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          limit: ragLimit,
          use_hybrid: true,
          use_faiss: ragEngine === "faiss",
          use_milvus: ragEngine === "milvus",
          filters: null
        })
      });

      if (!resp.ok) {
        const errData = await resp.json();
        throw new Error(errData.detail || "RAG query failed");
      }

      const data = await resp.json();

      const assistantMsg: Message = {
        sender: "assistant",
        text: data.answer,
        sources: data.documents
      };

      setChatHistory(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: Message = {
        sender: "assistant",
        text: `Error: ${err.message || "Could not retrieve answer from backend RAG pipeline."}`,
        error: true
      };
      setChatHistory(prev => [...prev, errorMsg]);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">R</div>
          <div className="logo-text">RAG Benchmarking</div>
        </div>

        <nav className="nav-menu">
          <div
            className={`nav-item ${activeTab === "search" ? "active" : ""}`}
            onClick={() => setActiveTab("search")}
          >
            🔍 Semantic Search
          </div>
          <div
            className={`nav-item ${activeTab === "rag" ? "active" : ""}`}
            onClick={() => setActiveTab("rag")}
          >
            🤖 RAG QA Assistant
          </div>
        </nav>

        {/* System Health Summary */}
        <div style={{ marginTop: "auto", borderTop: "1px solid var(--border-card)", paddingTop: "1.5rem" }}>
          <div className="stat-label" style={{ marginBottom: "0.75rem" }}>System Health</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <div className="stat-status">
              <span className={`status-dot ${health?.database === "connected" ? "active" : "inactive"}`}></span>
              PostgreSQL: {health?.database === "connected" ? "OK" : "Error"}
            </div>
            <div className="stat-status">
              <span className={`status-dot ${health?.faiss_loaded ? "active" : "inactive"}`}></span>
              FAISS Index: {health?.faiss_loaded ? "Loaded" : "Not Loaded"}
            </div>
            <div className="stat-status">
              <span className={`status-dot ${health?.model_loaded ? "active" : "inactive"}`}></span>
              LLM Pipeline: {health?.model_loaded ? "Ready" : "Offline"}
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main className="main-content">
        <header className="header-section">
          <h1>
            {activeTab === "search" && "Vector Similarity Search"}
            {activeTab === "rag" && "RAG QA chatbot"}
          </h1>
          <p>
            {activeTab === "search" && "Search 100k arXiv paper abstracts semantically across Postgres and FAISS."}
            {activeTab === "rag" && "Ask questions about ML concepts using contextual papers retrieved from your vector storage."}
          </p>
        </header>

        {/* Global Statistics Display */}
        <div className="stats-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
          <div className="stat-card">
            <span className="stat-label">PostgreSQL Papers</span>
            <span className="stat-value">{stats?.total_documents?.toLocaleString() || "0"}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">FAISS Quantization</span>
            <span className="stat-value">{stats?.has_faiss_index ? "HNSW Index" : "None"}</span>
          </div>
        </div>

        {/* Active Tab Panels */}

        {activeTab === "search" && (
          <div className="dashboard-grid">
            {/* Search Settings Panel */}
            <form className="panel-card" onSubmit={handleSearch}>
              <h3 className="panel-title">Query Parameters</h3>

              <div className="form-group">
                <label className="form-label">Search Query</label>
                <div className="search-input-wrapper">
                  <input
                    type="text"
                    className="search-input"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="e.g. deep learning transformers..."
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Retrieval Engine</label>
                <div className="engine-options">
                  <div
                    className={`engine-option ${searchEngine === "pgvector" ? "active" : ""}`}
                    onClick={() => setSearchEngine("pgvector")}
                  >
                    <input
                      type="radio"
                      checked={searchEngine === "pgvector"}
                      onChange={() => setSearchEngine("pgvector")}
                    />
                    <div>
                      <div className="engine-name">Postgres (pgvector)</div>
                      <div className="engine-desc">Relational DB matching</div>
                    </div>
                  </div>

                  <div
                    className={`engine-option ${searchEngine === "faiss" ? "active" : ""}`}
                    onClick={() => setSearchEngine("faiss")}
                  >
                    <input
                      type="radio"
                      checked={searchEngine === "faiss"}
                      onChange={() => setSearchEngine("faiss")}
                    />
                    <div>
                      <div className="engine-name">pgvector + FAISS</div>
                      <div className="engine-desc">High-speed in-memory indexing</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Results Limit (K)</label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="1"
                    max="20"
                    value={searchLimit}
                    onChange={(e) => setSearchLimit(parseInt(e.target.value))}
                  />
                  <span className="slider-val">{searchLimit}</span>
                </div>
              </div>



              <button type="submit" className="btn-primary" disabled={isSearching}>
                {isSearching ? <span className="loading-spinner"></span> : "Execute Search"}
              </button>
            </form>

            {/* Results Output List */}
            <div className="results-section">
              <div className="results-header">
                <span className="results-count">
                  {isSearching ? "Searching..." : `Found ${searchResults.length} matches`}
                </span>
              </div>

              {searchError && (
                <div className="result-card" style={{ borderColor: "#ef4444" }}>
                  <div className="result-title" style={{ color: "#ef4444" }}>Failed to Execute Search</div>
                  <div className="result-content">{searchError}</div>
                </div>
              )}

              {searchResults.map((result, idx) => (
                <div className="result-card" key={result.id}>
                  <div className="result-header-row">
                    <h4 className="result-title">{`Document #${idx + 1}`}</h4>
                    <div className="result-meta">
                      <span className="badge badge-id">ID: {result.id}</span>
                      <span className="badge badge-score">Score: {result.distance.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="result-content">{result.content}</div>
                  <div className="result-tags">
                    <span className="result-tag">Author: {result.metadata.author || "arXiv"}</span>
                    <span className="result-tag">Category: {result.metadata.category || "cs.AI"}</span>
                    {result.metadata.tags?.map(tag => (
                      <span className="result-tag" key={tag}>{tag}</span>
                    ))}
                  </div>
                </div>
              ))}

              {searchResults.length === 0 && !isSearching && !searchError && (
                <div style={{ textAlign: "center", padding: "4rem 0", color: "var(--text-muted)" }}>
                  <span style={{ fontSize: "3rem" }}>🔍</span>
                  <p>Enter a query and run a search to view vector similarity results</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "rag" && (
          <div className="dashboard-grid">
            {/* RAG Settings Panel */}
            <form className="panel-card" onSubmit={(e) => e.preventDefault()}>
              <h3 className="panel-title">RAG Context Settings</h3>

              <div className="form-group">
                <label className="form-label">Retrieval Database</label>
                <div className="engine-options">
                  <div
                    className={`engine-option ${ragEngine === "pgvector" ? "active" : ""}`}
                    onClick={() => setRagEngine("pgvector")}
                  >
                    <input
                      type="radio"
                      checked={ragEngine === "pgvector"}
                      onChange={() => setRagEngine("pgvector")}
                    />
                    <div>
                      <div className="engine-name">Postgres (pgvector)</div>
                      <div className="engine-desc">SQL-guided semantic matches</div>
                    </div>
                  </div>

                  <div
                    className={`engine-option ${ragEngine === "faiss" ? "active" : ""}`}
                    onClick={() => setRagEngine("faiss")}
                  >
                    <input
                      type="radio"
                      checked={ragEngine === "faiss"}
                      onChange={() => setRagEngine("faiss")}
                    />
                    <div>
                      <div className="engine-name">Postgres + FAISS</div>
                      <div className="engine-desc">Fastest index retrieval</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Retrieval Limit (Context Docs)</label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={ragLimit}
                    onChange={(e) => setRagLimit(parseInt(e.target.value))}
                  />
                  <span className="slider-val">{ragLimit}</span>
                </div>
              </div>
            </form>

            {/* Chatbot Interface */}
            <div className="chat-window">
              <div className="chat-history">
                {chatHistory.length === 0 && (
                  <div style={{ margin: "auto", textAlign: "center", color: "var(--text-muted)" }}>
                    <span style={{ fontSize: "4rem" }}>🤖</span>
                    <h3>RAG Assistant</h3>
                    <p style={{ maxWidth: "400px" }}>Ask a question regarding Machine Learning. The assistant will search the 100k arXiv paper dataset to generate factual answers.</p>
                  </div>
                )}

                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`chat-bubble ${msg.sender} ${msg.error ? "error" : ""}`}>
                    <div>{msg.text}</div>

                    {msg.sources && msg.sources.length > 0 && (
                      <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", marginTop: "1rem", paddingTop: "0.75rem" }}>
                        <div className="chat-sources-title">Retrieved Papers Context:</div>
                        <div className="chat-sources">
                          {msg.sources.map((src, sIdx) => (
                            <details className="chat-source-item" key={sIdx}>
                              <summary style={{ fontWeight: 600, fontSize: "0.85rem", cursor: "pointer" }}>
                                {`Paper ID: ${src.id} (Distance: ${src.distance.toFixed(4)})`}
                              </summary>
                              <div className="chat-source-snippet">{src.content}</div>
                            </details>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {isGenerating && (
                  <div className="chat-bubble assistant">
                    <span className="loading-spinner" style={{ marginRight: "0.5rem" }}></span>
                    Generating answer from context docs...
                  </div>
                )}
              </div>

              <form className="chat-input-bar" onSubmit={handleRAGSubmit}>
                <textarea
                  className="chat-textarea"
                  value={ragQueryText}
                  onChange={(e) => setRagQueryText(e.target.value)}
                  placeholder="Ask a question..."
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleRAGSubmit(e);
                    }
                  }}
                />
                <button type="submit" className="btn-primary" disabled={isGenerating || !ragQueryText.trim()}>
                  Ask
                </button>
              </form>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
