import os, sys
from pathlib import Path

# ------------------------------------------------------------------
# FIX: Add project root path
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import asyncio
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.core.rag.agent.nodes.assistant_node import assistant_node
from backend.core.rag.agent.graph_state import AgentState


async def test_case(description, user_query, docs=None):
    print("\n" + "="*80)
    print(f"TEST → {description}")
    print("="*80)

    # Fake State
    state = AgentState(
        messages=[HumanMessage(content=user_query)],
        session_id="test_session_123",
        docs=docs or [],
        final_output=None
    )

    config = RunnableConfig()

    result_state = await assistant_node(state, config)
    last_msg = result_state["messages"][-1]

    print("\nAssistant Node Output:")
    print("----------------------")

    # Tool decision?
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print("Decision: 🟧 TOOL CALL REQUIRED")
        print("Tool Calls:", last_msg.tool_calls)
    else:
        print("Decision: 🟦 NO_TOOL_REQUIRED")
        print("Message:", last_msg.content)

    print("\n")


async def main():
    # TEST 1: Should call RAG
    await test_case(
        "Query referencing document → EXPECT TOOL CALL",
        "According to the uploaded PDF, what does section 4 explain?",
        docs=["research_paper.pdf"]
    )

    # TEST 2: Should NOT call RAG
    await test_case(
        "General knowledge question → NO_TOOL_REQUIRED",
        "What is machine learning?"
    )

    # TEST 3: Another RAG query
    await test_case(
        "Document-specific query → EXPECT TOOL CALL",
        "Summarize the findings of the uploaded report.",
        docs=["financial_report.pdf"]
    )

    # TEST 4: Greeting (no tool)
    await test_case(
        "Small-talk → NO_TOOL_REQUIRED",
        "hello!"
    )


if __name__ == "__main__":
    asyncio.run(main())