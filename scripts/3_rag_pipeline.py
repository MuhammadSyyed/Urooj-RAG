"""
RAG pipeline: ingest processed docs into ChromaDB and answer questions.

Two modes:
1) Ingest: build / update the Chroma collection from processed docs
2) Query: retrieve top chunks and generate an answer with chosen LLM

Run examples:
    # Ingest
    python rag_pipeline.py ingest --processed artifacts/processed_docs.jsonl
    # Query
    python rag_pipeline.py query --question "What is Bancassurance?" --model gpt-4o-mini

Dependencies:
    pip install chromadb openai requests sentence-transformers
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from llm_clients import client_from_env, safe_generate


def load_processed_docs(path: Path) -> List[Dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def get_embedding_function(provider: str, model: str):
    provider = provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY required for OpenAI embeddings.")
        base_url = os.getenv("OPENAI_BASE_URL")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model,
            api_base=base_url,
        )
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return embedding_functions.OllamaEmbeddingFunction(model_name=model, base_url=base_url)
    # Default: sentence-transformers (no API needed)
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)


def ingest_into_chroma(
    processed_path: Path,
    persist_dir: Path,
    collection_name: str,
    embed_provider: str,
    embed_model: str,
) -> None:
    client = chromadb.PersistentClient(path=str(persist_dir))
    emb_fn = get_embedding_function(embed_provider, embed_model)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=emb_fn)

    docs = load_processed_docs(processed_path)
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict] = []
    for doc in docs:
        for chunk in doc.get("chunks", []):
            ids.append(chunk["id"])
            texts.append(chunk["text"])
            metas.append(chunk.get("metadata", {}))

    # Chroma can batch internally; keep simple here.
    collection.upsert(ids=ids, documents=texts, metadatas=metas)
    print(f"Ingested {len(ids)} chunks into collection '{collection_name}'.")


def retrieve(collection, query: str, top_k: int) -> List[Dict]:
    results = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for doc, meta, doc_id in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
        hits.append({"id": doc_id, "text": doc, "metadata": meta})
    return hits


def generate_answer(query: str, contexts: List[Dict], llm_model: str, provider: str, temperature: float) -> str:
    llm = client_from_env(model=llm_model, provider=provider)
    context_text = "\n\n".join([f"[{c['id']}]\n{c['text']}" for c in contexts])
    prompt = f"""You are a helpful assistant using provided context.
Answer concisely using only the context. If unsure, say you do not know.

Context:
{context_text}

Question: {query}
Answer:"""
    return safe_generate(llm, prompt, temperature=temperature, max_tokens=400)


def handle_ingest(args: argparse.Namespace) -> None:
    ingest_into_chroma(
        processed_path=args.processed,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embed_provider=args.embed_provider,
        embed_model=args.embed_model,
    )


def handle_query(args: argparse.Namespace) -> None:
    client = chromadb.PersistentClient(path=str(args.persist_dir))
    emb_fn = get_embedding_function(args.embed_provider, args.embed_model)
    collection = client.get_or_create_collection(
        name=args.collection, embedding_function=emb_fn)
    hits = retrieve(collection, args.question, args.top_k)
    answer = generate_answer(
        query=args.question,
        contexts=hits,
        llm_model=args.model,
        provider=args.provider,
        temperature=args.temperature,
    )
    print("Answer:\n", answer)
    print("\nContexts used:")
    for h in hits:
        print(f"- {h['id']}: {h['text'][:120]}...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG pipeline with ChromaDB.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser(
        "ingest", help="Ingest processed docs into ChromaDB.")
    ingest.add_argument("--processed", type=Path,
                        default=Path("../artifacts/processed_docs.jsonl"))
    ingest.add_argument("--persist_dir", type=Path,
                        default=Path("chroma_store"))
    ingest.add_argument("--collection", type=str, default="data")
    ingest.add_argument("--embed_provider", type=str, default="sentence-transformers",
                        choices=["openai", "ollama", "sentence-transformers"])
    ingest.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    ingest.set_defaults(func=handle_ingest)

    query = sub.add_parser("query", help="Query the RAG system.")
    query.add_argument("--question", type=str, required=True)
    query.add_argument("--persist_dir", type=Path,
                       default=Path("chroma_store"))
    query.add_argument("--collection", type=str, default="data")
    query.add_argument("--top_k", type=int, default=3)
    query.add_argument("--provider", type=str,
                       default="ollama", choices=["openai", "ollama"])
    query.add_argument("--model", type=str, default="gemma3:1b")
    query.add_argument("--temperature", type=float, default=0.1)
    query.add_argument("--embed_provider", type=str, default="sentence-transformers",
                       choices=["openai", "ollama", "sentence-transformers"])
    query.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    query.set_defaults(func=handle_query)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
