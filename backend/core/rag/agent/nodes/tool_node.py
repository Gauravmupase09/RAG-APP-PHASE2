# backend/core/rag/agent/nodes/tool_node.py

from langgraph.prebuilt import ToolNode
from backend.core.rag.agent.rag_tool import rag_tool

# Expose a ready-to-use ToolNode for the graph builder.
tool_node = ToolNode([rag_tool])