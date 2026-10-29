"""Build a pre-computed retrieval index (kb_index.json) from knowledge_base/.

Run once whenever the documents change. The deployed app loads this cache
instantly instead of re-parsing hundreds of MB of PDFs at startup.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rag import KnowledgeBase

kb = KnowledgeBase(ROOT / "knowledge_base", rebuild=True)
kb.save_cache()

print("Indexed documents:", kb.document_count)
print("Chunks:", kb.chunk_count)
print("Wrote:", ROOT / "kb_index.json")
