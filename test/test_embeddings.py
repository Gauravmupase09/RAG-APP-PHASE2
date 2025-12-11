import os
import sys
import json
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.doc_processing_unit.embedding_engine import embed_chunks
from backend.utils.config import PROCESSED_DIR
from backend.core.doc_processing_unit.qdrant_manager import client, get_collection_name

# ⚠️ Update session ID before running
session_id = "9d343441-b56b-4e3b-a823-6e66bb775f0a"

print(f"🧠 Running embedding generation test for session: {session_id}\n")

# Run embedding
embeddings = embed_chunks(session_id)

# ✅ Local verification
print(f"\n✅ Total embeddings generated locally: {len(embeddings)}")

if embeddings:
    print("\n📌 Sample embedding metadata:")
    print(json.dumps(embeddings[0]["metadata"], indent=2))

# ✅ Folder structure check
session_dir = PROCESSED_DIR / session_id
print("\n📁 Checking 'embeddings' folder structure:")
for root, dirs, files in os.walk(session_dir):
    if "embeddings" in root:
        print(f"📂 {root}")
        for f in files[:3]:  # Show first few embedding files
            print("  -", f)

# ✅ Verify upserts in Qdrant
try:
    collection_name = get_collection_name(session_id)
    stats = client.get_collection(collection_name)
    print(f"\n📊 Qdrant Collection: {collection_name}")
    print(f"Points count: {stats.points_count}")
except Exception as e:
    print(f"⚠️ Could not verify Qdrant collection: {e}")

print("\n🎯 Embedding test completed.\n")