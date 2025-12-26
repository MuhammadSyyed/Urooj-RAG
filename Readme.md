# Closed-World RAG LLM Evaluation for Financial QA

A complete end-to-end project that benchmarks multiple LLMs inside the same Retrieval Augmented Generation (RAG) pipeline for **financial question answering**.  
The system enforces a **closed-world** setting (fixed corpus, no external knowledge) and evaluates models using an **LLM-as-a-judge** scoring pipeline with hallucination analysis.

## Key Features

- Closed-world RAG setup using a curated financial document corpus
- Document preprocessing into JSONL chunks
- Embedding + Vector DB (Chroma) using `all-MiniLM-L6-v2`
- Synthetic QA dataset generation (`144` QA pairs with references)
- Parallel inference across multiple LLMs (example: qwen, deepseek-r1, gemma3, phi3, smollm)
- Automated evaluation using LLM-as-a-judge
- Metrics: average score (1–5), hallucination rate (score ≤ 2)
- Web app: Flask backend + React frontend chat interface
- Model switching endpoint for live comparison

## Tech Stack

- Backend: Python + Flask
- Frontend: React.js
- Vector Database: ChromaDB
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Storage/Artifacts: JSONL outputs for dataset, model answers, and evaluations

## Project Structure 

```text
.
├─ data/                          # Raw financial documents (closed-world corpus)
├─ artifacts/
│  ├─ processed_docs.jsonl        # Preprocessed chunks
│  ├─ qa_dataset.jsonl            # Generated QA dataset (144 items)
│  ├─ outputs/                    # Model outputs (jsonl)
│  └─ eval/                       # Evaluation files (json/jsonl)
├─ chroma_store/                  # Persistent Chroma vector DB
├─ chatbackend/
│  ├─ app.py                      # Flask app
│  ├─ rag_pipeline.py             # Retrieval + inference logic
│  ├─ data_preprocessing.py       # Chunking + cleaning
│  ├─ qa_dataset_generator.py     # QA generation
│  ├─ generate_model_outputs.py   # Multi-model inference
│  └─ evaluation_llm_judge.py     # LLM judge scoring
├─ chatfront/
│  ├─ src/                        # React app
│  └─ package.json
└─ README.md



