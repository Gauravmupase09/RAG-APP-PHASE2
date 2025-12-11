# backend/core/rag/agent/nodes/finalize_node.py

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage

from backend.core.rag.agent.graph_state import AgentState
from backend.core.rag.llm_engine import generate_general_answer
from backend.core.rag.rag_pipeline import run_rag_generation
from backend.core.rag.session_memory import add_to_session_memory, get_session_memory


async def finalize_node(state: AgentState) -> AgentState:
    """
    FINAL NODE in the agent graph.

    It produces the final answer for BOTH paths:

    1) NO_TOOL_REQUIRED (general LLM):
       - last message is AIMessage("NO_TOOL_REQUIRED")
       - we:
           * save user query to session memory
           * load memory context
           * call generate_general_answer(...)
           * save assistant reply to memory
           * store final_output in state

    2) RAG TOOL USED:
       - assistant_node emitted a tool_call
       - tool_node executed rag_tool and appended a ToolMessage
       - last message is that ToolMessage
       - we:
           * extract {query, chunks, citations} from ToolMessage.content
           * call run_rag_generation(...) which internally:
               - combines memory + chunks
               - calls generate_rag_answer(...)
               - updates memory for assistant
           * store final_output in state
    """

    last_msg = state["messages"][-1]
    session_id = state["session_id"]

    # ============================================================
    # 1️⃣ CASE: NO TOOL → GENERAL ANSWER
    #    last_msg is AIMessage("NO_TOOL_REQUIRED")
    # ============================================================
    if isinstance(last_msg, AIMessage) and last_msg.content == "NO_TOOL_REQUIRED":
        # The previous message should be the user's query (HumanMessage)
        user_msg = state["messages"][-2]
        user_query = user_msg.content

        # (A) Save user message into session memory
        add_to_session_memory(session_id, "user", user_query)

        # (B) Build memory text for context, EXCLUDING current user query
        memory = get_session_memory(session_id)
        if memory and memory[-1]["role"] == "user":
            memory_for_context = memory[:-1]
        else:
            memory_for_context = memory
        

        memory_text = (
            "\n".join([f"{m['role']}: {m['content']}" for m in memory_for_context])
            if memory_for_context else None
        )

        # (C) Generate general answer using LLM
        llm_result = generate_general_answer(
            query=user_query,
            memory_text=memory_text,
        )

        # (D) Save assistant response to memory
        add_to_session_memory(session_id, "assistant", llm_result["response"])

        # (E) Build final_output (no citations for general answers)
        final_output: Dict[str, Any] = {
            "query": user_query,
            "response": llm_result["response"],
            "model": llm_result["model"],
            "used_chunks": 0,
            "citations": [],
            "formatted_citations": "No citations available.",
        }

        # Optionally also append the final assistant reply to messages
        state["messages"].append(AIMessage(content=llm_result["response"]))

        return {**state, "final_output": final_output}

    # ============================================================
    # 2️⃣ CASE: TOOL WAS USED → RAG ANSWER
    #    last_msg is a ToolMessage from rag_tool
    # ============================================================
    # At this point, we expect the LangGraph ToolNode to have run
    # rag_tool and appended a ToolMessage with its result.
    if isinstance(last_msg, ToolMessage):
        tool_payload = last_msg.content

        # ToolMessage.content might be dict or JSON string depending on wiring
        if isinstance(tool_payload, str):
            try:
                tool_payload = json.loads(tool_payload)
            except Exception as e:
                raise ValueError(
                    f"❌ finalize_node: Could not parse tool result JSON: {e}"
                )

        if not isinstance(tool_payload, dict):
            raise ValueError(
                f"❌ finalize_node: Unexpected tool result type: {type(tool_payload)}"
            )

        # Extract fields returned by rag_tool / run_rag_retrieval
        # {
        #   "query": "<user query>",
        #   "chunks": [...],
        #   "citations": [...]
        # }
        query_from_tool = tool_payload.get("query")
        chunks = tool_payload.get("chunks", [])
        citations = tool_payload.get("citations", [])

        # Fallback: if for some reason query missing in tool result,
        # use the last human message before tool + assistant.
        if not query_from_tool:
            # last messages: [..., HumanMessage, AI(tool_call), ToolMessage]
            # so human is at index -3
            query_from_tool = state["messages"][-3].content

        # Now run the second stage of RAG:
        # combine memory + chunks, call generate_rag_answer, update memory, format citations
        rag_output = await run_rag_generation(
            session_id=session_id,
            query=query_from_tool,
            chunks=chunks,
            citations=citations,
        )

        # Optionally append final RAG answer as AIMessage
        state["messages"].append(AIMessage(content=rag_output["response"]))

        # Store final_output for FastAPI /query route to return
        return {**state, "final_output": rag_output}

    # ============================================================
    # 3️⃣ SAFETY NET: Unexpected last message type
    # ============================================================
    raise ValueError(
        f"❌ finalize_node: Unexpected last message type: {type(last_msg)}. "
        "Expected AIMessage('NO_TOOL_REQUIRED') or ToolMessage from rag_tool."
    )