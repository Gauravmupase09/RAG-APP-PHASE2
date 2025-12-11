import os, sys, json
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.doc_processing_unit.chunking import chunk_session_documents
from backend.utils.config import PROCESSED_DIR

# ⚠️ Use your active session ID
session_id = "9d343441-b56b-4e3b-a823-6e66bb775f0a"

print(f"🧩 Running document chunking test for session: {session_id}\n")

chunks = chunk_session_documents(session_id)

print(f"\n✅ Total chunks generated: {len(chunks)}")

# Print first few metadata entries
print("\n📌 Sample chunk metadata:")
print(json.dumps(chunks[:2], indent=2))

session_dir = PROCESSED_DIR / session_id

# ✅ Find chunk folders
chunk_folders = []
for doc_folder in session_dir.iterdir():
    if doc_folder.is_dir():
        for cf in doc_folder.glob("chunks_*"):
            if cf.is_dir():
                chunk_folders.append(cf)

print(f"\n📁 Total chunk folders: {len(chunk_folders)}")
for cf in chunk_folders:
    print(f" - {cf.name} (in {cf.parent.name})")

# ✅ Validate chunk subfolder contents
sample_chunk_folder = None
for cf in chunk_folders:
    first_chunk = cf / "chunk_1"
    if first_chunk.exists():
        sample_chunk_folder = first_chunk
        break

if sample_chunk_folder:
    print(f"\n🔍 Checking sample chunk folder: {sample_chunk_folder}")
    expected_files = {"text.txt", "meta.json"}
    found_files = {f.name for f in sample_chunk_folder.iterdir() if f.is_file()}
    missing = expected_files - found_files
    if missing:
        print(f"❌ Missing: {missing}")
    else:
        print("✅ Both text.txt and meta.json found!")

print("\n🎯 Chunking test completed.\n")