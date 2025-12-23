from pathlib import Path
import json
from 3_rag_pipeline import retrieve, generate_answer, get_embedding_function
import chromadb

qa_path = Path("artifacts/qa_dataset.jsonl")
out_path = Path("artifacts/model_outputs.jsonl")

client = chromadb.PersistentClient(path="chroma_store")
emb_fn = get_embedding_function("sentence-transformers", "all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="urooj_docs", embedding_function=emb_fn)

def load_jsonl(p): return [json.loads(l) for l in p.open()]

qa_items = load_jsonl(qa_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for item in qa_items:
        hits = retrieve(collection, item["question"], top_k=3)
        ans = generate_answer(item["question"], hits, llm_model="gpt-4o-mini", provider="openai", temperature=0.1)
        f.write(json.dumps({"qa_id": item["qa_id"], "question": item["question"], "answer": ans}, ensure_ascii=False) + "\n")