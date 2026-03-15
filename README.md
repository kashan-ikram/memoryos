# MemoryOS — Personal AI Second Brain

I built this to explore whether AI can do more than just answer questions — can it actually remember your life and help you think better?

Add a note, voice memo, or image and the system stores it in a semantic memory. When you ask a question, it searches your personal history, retrieves the most relevant context, and responds with grounded answers — not hallucinations.

👉 [Try the live demo](https://huggingface.co/spaces/kashanikram/memoryos)

## What it does

Add anything and get:
- Semantic search across all your personal notes and memories
- AI responses grounded in YOUR data — not generic answers
- Voice memo transcription (Urdu, English, Arabic supported)
- Image text extraction via OCR
- Personal timeline of everything you've saved
- Multi-user support — each user's memory is completely private

## How it works

Input (text/voice/image) → Embedding model converts to vectors → ChromaDB stores semantically → On query, RAG retrieves top matches → LLM responds using only your context

- RAG layer — sentence-transformers (all-MiniLM-L6-v2) for embeddings, ChromaDB for vector storage
- Agentic system — LangGraph with 4 agents: Ingestion, Retrieval, Pattern Analysis, Proactive Suggestions
- Voice — OpenAI Whisper for multilingual audio transcription
- Image — pytesseract OCR for text extraction from screenshots and documents
- Auth — Supabase Auth with email verification and password reset
- Permanent storage — Supabase PostgreSQL so memories survive restarts

## Stack

- Groq (LLaMA 3.3 70B) — LLM inference, free tier
- sentence-transformers — semantic embeddings
- ChromaDB — vector database for similarity search
- LangGraph — agentic workflow orchestration
- Supabase — PostgreSQL database + authentication
- OpenAI Whisper — multilingual speech-to-text
- Streamlit — UI, deployed on Hugging Face Spaces
- Docker — containerized deployment
- Google Colab (T4 GPU) — development and testing

**Total infrastructure cost: $0**

## Memory Architecture

```
Input Layer        → Text, Voice (Whisper), Image (OCR)
Processing Layer   → Chunk, Embed, Tag, Store
Agentic Layer      → 4 LangGraph agents
Memory Layer       → Short-term + Long-term + Episodic timeline
Storage Layer      → ChromaDB (vectors) + Supabase (permanent)
Output Layer       → Grounded RAG responses in user's language
```

## Agentic System

4 agents running on LangGraph:
- **Ingestion Agent** — decides how to chunk and store new information
- **Retrieval Agent** — semantic search + context assembly for queries
- **Pattern Agent** — analyzes memory history to find recurring themes
- **Proactive Agent** — surfaces reminders and suggestions without being asked

## Project Structure

```
├── app.py                  Streamlit app + RAG pipeline + auth + agents
├── requirements.txt        All dependencies
├── Dockerfile              Container configuration
├── .streamlit/
│   └── config.toml         Server configuration
└── README.md
```

## Built by

Kashan Ikram — BS Computer Science (AI specialization) @ BIMS, Pakistan
