# Mini RAG Application

A simple PDF Question-Answering system using Retrieval-Augmented Generation (RAG).

## Project Overview

This application allows a user to ask questions about a PDF document.

The application follows this pipeline:

PDF
→ PyMuPDF Text Extraction
→ Chunking + Overlap
→ Embeddings
→ Qdrant
→ Top-10 Retrieval
→ Reranking
→ Top-3 Chunks
→ Prompt Construction
→ Nugen LLM
→ Answer

## Technologies Used

- Python
- FastAPI
- PyMuPDF
- Sentence Transformers
- Qdrant
- Nugen API
- Uvicorn

## Project Structure

```text
mini_rag_app/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
├── documents/
│   └── TOURISM IN INDIA.pdf
│
├── src/
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── qdrant_db.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   └── ingest.py
│
└── venv/