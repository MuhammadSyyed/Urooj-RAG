from pathlib import Path
import json
from rag_pipeline import retrieve, generate_answer, get_embedding_function
import chromadb
from dotenv import load_dotenv
load_dotenv(override=True)
import os

qa_path = Path("../artifacts/qa_dataset.jsonl")
out_path = Path("../artifacts/qwen_model_outputs.jsonl")

client = chromadb.PersistentClient(path="chroma_store")
emb_fn = get_embedding_function("sentence-transformers", "all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="data", embedding_function=emb_fn)

def load_jsonl(p): return [json.loads(l) for l in p.open()]

qa_items = load_jsonl(qa_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for item in qa_items:
        hits = retrieve(collection, item["question"], top_k=3)
        ans = generate_answer(item["question"], hits, llm_model=os.getenv("model"), provider="ollama", temperature=0.1)
        # Write each record as a single-line JSON object (proper JSONL)
        f.write(json.dumps({"qa_id": item["qa_id"], "question": item["question"], "answer": ans}, ensure_ascii=False) + "\n")