from __future__ import annotations
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv(override=True)
import os
from llm_clients import client_from_env, safe_generate
from text_utils import extract_json

PROMPT_TEMPLATE = """You are given document content delimited by <doc>.
Generate at least {num_q} diverse question-answer pairs that a user might ask.
Keep answers grounded only in the provided content.
Return ONLY valid JSON array, no code fences, shaped as:
[{{"question": "...", "answer": "..."}}]

<doc>
{context}
</doc>
"""


def load_processed_docs(path: Path) -> List[Dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def generate_qa_for_doc(doc: Dict, llm_model: str, provider: str, num_q: int) -> List[Dict]:
    llm = client_from_env(model=llm_model, provider=provider)
    context = doc.get("full_text") or " ".join([c["text"] for c in doc.get("chunks", [])])
    prompt = PROMPT_TEMPLATE.format(num_q=num_q, context=context[:6000])
    raw = safe_generate(llm, prompt, max_tokens=2000, temperature=0.2)
    try:
        clean = extract_json(raw)
        if not clean:
            raise ValueError(f"JSON extraction failed. Raw output:\n{raw}")
        data = json.loads(extract_json(raw))
        if not all(
            isinstance(x, dict) and "question" in x and "answer" in x
            for x in data):
            raise ValueError("Malformed QA objects returned by LLM")
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return JSON for doc {doc.get('doc_id')}: {raw}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected list of QAs, got {type(data)}")
    cleaned = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            cleaned.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "qa_id": f"{doc.get('doc_id')}::qa::{idx}",
                    "question": q,
                    "answer": a,
                    "source_pdf": doc.get("source_pdf"),
                    "metadata": doc.get("metadata", {}),
                }
            )
    if len(cleaned) < num_q:
        raise ValueError(f"Only got {len(cleaned)} QAs for {doc.get('doc_id')}, expected {num_q}+")
    return cleaned


def build_dataset(
    processed_path: Path,
    out_path: Path,
    llm_model: str,
    provider: str,
    num_q: int,
) -> None:
    docs = load_processed_docs(processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in tqdm(docs):
            try:
                qas = generate_qa_for_doc(doc, llm_model=llm_model, provider=provider, num_q=num_q)
            except Exception as e:
                print(f"Skipping {doc.get('doc_id')}: {e}")
                continue
            for qa in qas:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            print(f"Wrote {len(qas)} QAs for {doc.get('doc_id')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA dataset from processed docs.")
    parser.add_argument("--processed", type=Path, default=Path("../artifacts/processed_docs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("../artifacts/qa_dataset.jsonl"))
    parser.add_argument("--model", type=str, default=os.getenv("model"))
    parser.add_argument("--provider", type=str, default="ollama", choices=["openai", "ollama"])
    parser.add_argument("--num_questions", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        processed_path=args.processed,
        out_path=args.out,
        llm_model=args.model,
        provider=args.provider,
        num_q=args.num_questions,
    )


