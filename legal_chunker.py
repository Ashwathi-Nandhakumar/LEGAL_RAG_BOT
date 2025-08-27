# legal_chunker.py
import re
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def hybrid_legal_chunker(text: str, max_chunk_size: int = 800) -> List[str]:
    # Step 1: Regex split on common legal headers
    pattern = re.compile(r"(?=(Section\s+\d+|Article\s+\d+|^\d+\.\s+[A-Z]))",
                         re.IGNORECASE | re.MULTILINE)
    splits = pattern.split(text)

    # Merge headers with their content
    chunks = []
    for i in range(1, len(splits), 2):
        header = splits[i].strip()
        body = splits[i+1].strip() if i+1 < len(splits) else ""
        chunks.append(f"{header}\n{body}")

    if not chunks:
        chunks = [text]  # fallback if regex finds nothing

    # Step 2: If a chunk is too long, break it semantically
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    semantic_splitter = SemanticSplitterNodeParser(embed_model=embed_model)

    final_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > max_chunk_size:
            nodes = semantic_splitter.get_nodes_from_documents([Document(text=chunk)])
            final_chunks.extend([n.text for n in nodes])
        else:
            final_chunks.append(chunk)

    return final_chunks
