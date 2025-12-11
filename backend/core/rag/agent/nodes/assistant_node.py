# backend/core/rag/agent/nodes/assistant_node.py

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from backend.core.rag.agent.graph_state import AgentState
from backend.core.rag.agent.rag_tool import rag_tool
from backend.core.rag.llm_engine import get_llm  # Your Gemini Flash 2.5 wrapper


# ============================================================
# 🧠 SYSTEM PROMPT FOR THE ASSISTANT NODE
# ============================================================

ASSISTANT_SYSTEM_PROMPT = """
You are an Intent Classification Controller for a Document Question Answering System.

Your ONLY responsibility is to decide whether the user’s question REQUIRES calling the `rag_tool`.

VERY IMPORTANT:
- You must NOT generate the final answer for the user.
- You must NOT summarize or explain anything.
- You must NOT fabricate document content.
- Another component will produce the final answer. Your job is ONLY to route correctly.

------------------------------------------------------------
WHEN YOU MUST CALL `rag_tool`:
------------------------------------------------------------
Call the tool if the user’s question requires ANY information that may exist inside the uploaded documents.

    - The query references uploaded files, PDFs, reports, sections, rules, tables, or content found inside documents.
    - The user says things like:
        "According to the document..."
        "What does the PDF say about..."
        "Explain section 3..."
        "Summarize findings in the uploaded file..."
    - If the answer needs grounding in the document, ALWAYS call `rag_tool`.

------------------------------------------------------------
WHEN YOU MUST NOT CALL `rag_tool`:
------------------------------------------------------------
Do NOT call the tool for:
    - General knowledge questions.
    - Small-talk or chit-chat ("hi", "hello", "who are you").
    - Reasoning questions that do NOT depend on document content
    - Personal questions
    - Questions that do NOT depend on the uploaded documents.

------------------------------------------------------------
AVAILABLE INFO:
------------------------------------------------------------
    - session_id: identifies the user's session
    - docs: list of uploaded document names (may be empty)

------------------------------------------------------------
YOUR OUTPUT MUST BE ONE OF:
------------------------------------------------------------

1️⃣ If document info IS required:
    → Produce a tool call to `rag_tool`.

2️⃣ If document info is NOT required:
    → Output an assistant message: "NO_TOOL_REQUIRED"

Another system component will produce the final answer.
"""


# ============================================================
# 🤖 ASSISTANT NODE LOGIC
# ============================================================

async def assistant_node(state: AgentState, config: RunnableConfig):
    """
    Decides:
        - Should the query trigger rag_tool?
        - Or is it a general query?

    Output is ALWAYS an AIMessage:
        - AIMessage(tool_calls=[...])   → tool execution path
        - AIMessage("NO_TOOL_REQUIRED") → normal answering path
    """

    # 1️⃣ Load base LLM
    llm = get_llm()  # Your Gemini Flash 2.5 or any model you use

    # 2️⃣ Bind the tool — now LLM can output tool call messages
    llm_with_tools = llm.bind_tools([rag_tool])

    # 3️⃣ Get latest user message
    user_msg = state["messages"][-1]

    # 4️⃣ Prepare docs metadata (MUST be merged into system prompt — no extra SystemMessage allowed)
    docs = state.get("docs") or []
    docs_text = (
        f"Uploaded documents in this session: {', '.join(docs)}"
        if docs else
        "No uploaded documents found in this session."
    )

    session_text = f"session_id for this conversation: {state['session_id']}"

    # 5️⃣ Build FINAL system prompt given to Gemini
    SYSTEM_PROMPT = ASSISTANT_SYSTEM_PROMPT + f"\n\nSESSION METADATA:\n" + docs_text + "\n" + session_text

    # 6️⃣ Build classification prompt
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg.content),
    ]

    # 7️⃣ Run LLM to classify
    response = await llm_with_tools.ainvoke(messages, config=config)

    # 8️⃣ If model did NOT call tool → classify as general query
    if not getattr(response, "tool_calls", None):
        response = AIMessage(content="NO_TOOL_REQUIRED")

    # 9️⃣ Add decision result to state
    updated_messages = state["messages"] + [response]

    # Return updated state to LangGraph
    return {**state, "messages": updated_messages}