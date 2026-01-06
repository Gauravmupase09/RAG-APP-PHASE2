# 📘 RAG-APP — Phase 2: Agentic RAG with LangGraph

RAG-APP Phase 2 is an advanced upgrade of the Phase 1 RAG system, introducing an Agentic Architecture powered by LangGraph.

---

**In this phase, the LLM autonomously decides:**

 - When to retrieve document context (tool-calling)

 - When a query is general (no retrieval needed)

 - How to combine session memory + document chunks

 - How to construct final answers via a multi-node workflow

This results in faster, smarter, and more context-aware interactions.

---

# 🚀 What’s New in Phase 2?

| Feature |	Phase 1	| Phase 2 (New!) |
|------|--------------|-------------|
| **RAG Pipeline** | Static pipeline | Agentic graph with autonomous routing |
| **Tool Use** | None	| LangGraph ToolNode triggers rag_tool |
| **LLM Routing** | Always retrieval | LLM decides retrieval vs general answer |
| **Conversation Memory** |	Basic sliding window | Fully integrated in agent graph |
| **Architecture** | Linear	| Multi-node agent workflow |
| **Performance** | Redundant retrieval	| Retrieval only when needed |

---

# 🧠 Agentic Workflow Overview
```bash
START
  ↓
assistant_node  →  decides → general OR rag_tool
  ├── tool_call → tool_node → finalize_node → END(Final Response Without The Citations)
  └── NO_TOOL_REQUIRED → finalize_node
                              ↓
                             END(Final Response With The Citations)
```
---

**assistant_node**
 - LLM analyzes the query
 - If document-based → produces a tool_call
 - If general → routes to finalize_node

**tool_node (rag_tool)**
 - Retrieves top-k document chunks
 - Returns chunks + citations to graph

**finalize_node**
- Combines:
  - session memory
  - user question
  - retrieved chunks (if any)
  - Produces final answer

---

# 📁 Project Structure (Phase 2)
```bash
RAG-APP/
│
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── upload.py
│   │   │   ├── process.py
│   │   │   ├── query.py
│   │   │   ├── test_tool.py
│   │   │   ├── reset_session.py
│   │   │   └── list_docs.py
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── rag/
│   │   │   ├── rag_pipeline.py
│   │   │   ├── citation_handler.py
│   │   │   ├── retriever.py
│   │   │   ├── llm_engine.py
│   │   │   ├── session_memory.py
│   │   │   └── resource_store.py
│   │   │
│   │   ├── doc_processing_unit/
│   │   ├── text_extractor.py
│   │   ├── text_cleaner.py
│   │   ├── chunking.py
│   │   ├── embedding_engine.py
│   │   ├── model_manager.py
│   │   └── qdrant_manager.py
│   │   │
│   │   └──agent/
│   │       ├── graph_state.py
│   │       ├── rag_tool.py
│   │       ├── nodes/
│   │       │   ├── assistant_node.py
│   │       │   ├── finalize_node.py
│   │       │   └── tool_node.py
│   │       └── graph_builder.py
│   │
│   ├── data/
│   │   ├── uploads/
│   │   └── processed/
│   │
│   ├── model/
│   │   └── schemas.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── file_manager.py
│   │   └── logger.py
│   │
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── components/
│   │   ├── upload_section.py
│   │   ├── chat_section.py
│   │   └── citation_box.py
│   │
│   ├── utils/
│   │   ├── api_client.py
│   │   └── config.py
│   │
│   ├── app.py
│   └── requirements.txt
│
├── test/
│   ├── test_extract.py
│   ├── test_cleaner.py
│   ├── test_chunking.py
│   ├── test_model.py
│   ├── test_qdrant.py
│   ├── test_llm.py
│   ├── test_rag_pipeline.py
│   ├── test_embeddings.py
│   └── etc.
│
├── .env
├── .gitignore
└── README.md
```
---

# ⚙️ Tech Stack (Phase 2)

| Layer | Technology |
|--------------|-------------|
| **Agent** | Framework |	LangGraph |
| **LLM**	| Google Gemini 2.5 Flash |
| **Vector DB**	| Qdrant |
| **Embeddings**	| BAAI/bge-small-en-v1.5 |
| **Backend**	| FastAPI |
| **Frontend**	| Streamlit |
| **Memory**	| Sliding window via session_memory |

---

# 🛠️ Installation & Setup
**1️⃣ Clone Repository**
```bash
git clone https://github.com/Gauravmupase09/RAG-APP-PHASE2.git
cd RAG-APP-PHASE2
```
---

# 🔧 Backend Setup (FastAPI)
**2️⃣ Create Virtual Environment**
```bash
cd backend
python -m venv venv
venv/Scripts/activate
```
**3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

**4️⃣ Start Qdrant (Docker)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**5️⃣ Launch FastAPI Server**
```bash
uvicorn main:app --reload
```

API available at:
 - http://localhost:8000
 - http://localhost:8000/docs

---

# 🎨 Frontend Setup (Streamlit)
```bash
cd ../frontend
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

Frontend:
👉 http://localhost:8501

---

# 🔄 Agentic Workflow (Detailed)
**1️⃣ User sends a query**
- The system forwards it to assistant_node.

**2️⃣ assistant_node decides:**
- If retrieval is needed → calls rag_tool
- If it's general → skips retrieval

**3️⃣ tool_node retrieves:**
- top-k chunks
- citations
- returns structured payload

**4️⃣ finalize_node creates final answer using:**
- session memory
- retrieved chunks (if any)
- formatted citations

Final output is written into `state.final_output`.

---

# 📡 API Endpoints

| Method | Route	| Purpose |
|--------------|-------------|-------------|
| **POST** |	/api/upload	| Upload documents |
| **POST** | /api/process/{session_id}	| Process + embed documents |
| **POST** | /api/query	| Run Agentic RAG |
| **GET** | /api/list_docs	| List documents |
| **POST** | /api/reset_session	| Clear session + memory |

---

# 📚 Example Agentic Behavior

**User:**
`Who are you?`
LLM decision: general mode → no tool call

User:
`What does the document say about student expectations?`
LLM decision: retrieval required → rag_tool → RAG answer

---

# 🧪 Tests Included

Covers:
  - extraction
  - cleaning
  - chunking
  - embeddings
  - Qdrant
  - LLM engine
  - RAG pipeline
  - LangGraph agent behavior

---

# 🤝 Contributing

Contributions welcome!
You can propose:
- Multi-tool agent workflows
- More evaluators
- Streaming support
- Multi-document reasoning

---

# 📜 License

MIT License




