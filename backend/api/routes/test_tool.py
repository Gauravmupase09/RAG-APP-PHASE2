# backend/api/routes/test_tool.py

from fastapi import APIRouter, HTTPException
from backend.models.schemas import QueryRequest
from backend.core.rag.agent.rag_tool import rag_tool
from backend.utils.logger import logger

router = APIRouter()


@router.post("/test_rag_tool")
async def test_rag_tool(query_data: QueryRequest):
    """
    Temporary debug endpoint — calls rag_tool directly and returns chunks + citations.
    Use this to verify retrieval works for a processed session.
    """
    try:
        logger.info(f"🔧 Debug: testing rag_tool for session {query_data.session_id}")
        # rag_tool is an async tool; call it directly
        result = await rag_tool.ainvoke({
            "session_id": query_data.session_id,
            "query": query_data.query,
            "top_k": query_data.top_k or 5
        })

        return {
            "status": "ok",
            "query": result.get("query"),
            "chunks_count": len(result.get("chunks", [])),
            "citations_count": len(result.get("citations", [])),
            "chunks": result.get("chunks", []),
            "citations": result.get("citations", [])
        }

    except Exception as e:
        logger.exception("❌ test_rag_tool failed")
        raise HTTPException(status_code=500, detail=str(e))