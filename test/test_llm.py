# test/test_llm_stream.py(this needed few changes as we have change the llm_engine.py for our aggentic approach)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.rag.llm_engine import generate_llm_response, stream_llm_response

query = "Explain how COVID-19 affected education systems worldwide."
context = [
    "The pandemic forced schools and universities to shift to online learning.",
    "Many students faced challenges with internet access and motivation."
]

# Standard full response
res = generate_llm_response(query, context)
print("\n🧠 Full Response:\n", res["response"])

# Streaming response (typing effect)
print("\n⚡ Streaming Response:\n")
for token in stream_llm_response(query, context):
    print(token, end="", flush=True)