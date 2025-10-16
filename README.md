# LEGAL_RAG_BOT

## Project Overview

A legal Q&A chatbot that uses **Retrieval-Augmented Generation (RAG)**, advanced local embeddings, and a **custom contract-aware chunker** to deliver context-rich, clause-specific answers for uploaded contracts.  
Built with LlamaIndex, Gradio, and a hybrid chunking engine, this bot maximizes answer reliability, semantic precision, and transparency—ideal for contracts, agreements, and legal docs.

***

## Key Features & Highlights

- **Hybrid Chunking Strategy:**  
  Leverages both LlamaIndex’s robust chunking and a custom regex/manual chunker (`legal_chunker.py`) designed specifically for contracts. Sections are detected by both standard algorithms *and* legal key phrases, maximizing clause preservation.

- **Document-Aware RAG Pipeline:**  
  Chunks, embeds, and indexes documents with BAAI/bge-small-en-v1.5 (HuggingFace) for powerful vector search. Always retrieves the most relevant contract context before answering.

- **Purpose-Built Guardrails:**  
  All responses explain obligations or risks only when backed by context, never hallucinate advice, and clearly cite their source sections.

- **Conversational Memory:**  
  Multi-turn context for smarter, more coherent follow-up questions.

- **Audit-Level Logging:**  
  Every upload, retrieval, and answer is logged for transparency and debugging.

***

## Workflow

1. **Contract Upload:**  
   User uploads any legal contract through the Gradio app interface.

2. **Chunking & Preprocessing:**  
   File is split using the hybrid mechanism:
   - LlamaIndex section split for structure
   - Regex/manual chunking for key contract phrases (e.g., “Termination,” “Obligations,” numbered clauses)

3. **Embedding & Indexing:**  
   - Chunks converted into embeddings via HuggingFace
   - Indexed and stored for efficient semantic retrieval

4. **Contextual Q&A:**  
   - On each question, relevant chunks are retrieved and passed to the Llama-3 model to generate precise, clause-cited answers

5. **Logging:**  
   - Every user interaction and backend action is traceable


## Results

- **Superior contract section awareness** using advanced legal chunking
- **Accurate, cite-backed answers** for legal Q&A—never vague, always verifiable
- **Plug-and-play pipeline**: Upload any contract, instantly ready for secure, explainable Q&A

***

*LEGAL_RAG_BOT demonstrates state-of-the-art RAG for legal tech, blending general-purpose LLMs with legal document expertise and transparency-by-design. Designed for demos, research, and practical contract analytics.*

