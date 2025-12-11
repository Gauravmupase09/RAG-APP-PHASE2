import os, sys, json
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.doc_processing_unit.text_cleaner import clean_all_raw_files
from backend.utils.config import PROCESSED_DIR

# ⚠️ Use the new session ID
session_id = "9d343441-b56b-4e3b-a823-6e66bb775f0a"

print(f"🔥 Running text cleaner test for session: {session_id}\n")

cleaned_files = clean_all_raw_files(session_id)

print("\n✅ Cleaned Files:")
for f in cleaned_files:
    print(" -", f)

# ✅ Verify file_index.json got updated
meta_file = PROCESSED_DIR / session_id / "file_index.json"

if meta_file.exists():
    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
    print("\n📑 Updated file_index.json entries:")
    for entry in meta_data:
        print(entry)
else:
    print("❌ ERROR: file_index.json not found!")

print("\n🎯 Text cleaning test completed.\n")