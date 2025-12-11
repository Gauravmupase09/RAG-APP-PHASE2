from fastapi import APIRouter, HTTPException
from backend.models.schemas import QueryRequest, QueryResponse

from backend.utils.logger import logger
from backend.utils.file_manager import list_files
from backend.core.rag.agent.graph_builder import agentic_rag_graph

from langchain_core.messages import HumanMessage

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def handle_user_query(query_data: QueryRequest):
    """
    Agentic RAG Query Endpoint

    1. Automatically load uploaded documents for the session.
    2. Build initial AgentState:
          - session_id
          - docs (auto-discovered)
          - messages = [HumanMessage(query)]
    3. Run the agentic workflow graph (assistant → tool → finalize).
    4. Return structured QueryResponse.
    """

    try:
        session_id = query_data.session_id
        query_text = query_data.query

        logger.info(f"💬 New agentic RAG query for session={session_id}: '{query_text}'")

        # -----------------------------------------------------------------
        # 1️⃣ Load uploaded document names (REAL source of truth)
        # -----------------------------------------------------------------
        try:
            docs = list_files(session_id)  # returns list of filenames
        except Exception:
            docs = []

        logger.info(f"📄 Session {session_id} has documents: {docs}")

        # -----------------------------------------------------------------
        # 2️⃣ Build initial agent state for the graph
        # -----------------------------------------------------------------
        initial_state = {
            "session_id": session_id,
            "docs": docs,                          # NOW CORRECT
            "messages": [HumanMessage(content=query_text)],
        }

        # -----------------------------------------------------------------
        # 3️⃣ Invoke the agentic graph (async)
        # -----------------------------------------------------------------
        final_state = await agentic_rag_graph.ainvoke(initial_state)

        # -----------------------------------------------------------------
        # 4️⃣ Extract finalized output (constructed by finalize_node)
        # -----------------------------------------------------------------
        final_output = final_state.get("final_output")

        if not final_output:
            raise RuntimeError("❌ finalize_node did not produce final_output")

        # -----------------------------------------------------------------
        # 5️⃣ Build the API response (Pydantic model)
        # -----------------------------------------------------------------
        response = QueryResponse(
            query=final_output["query"],
            response=final_output["response"],
            model=final_output["model"],
            used_chunks=final_output["used_chunks"],
            citations=final_output["citations"],
            formatted_citations=final_output["formatted_citations"],
        )

        logger.info(f"✅ Agentic RAG query resolved successfully for session {session_id}")

        return response

    except Exception as e:
        logger.exception(f"❌ Error processing agentic RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))