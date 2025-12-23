from __future__ import annotations
import argparse
import json
from dotenv import load_dotenv
load_dotenv(override=True)
import os
from pathlib import Path
from typing import Dict, List
from pypdf import PdfReader
from text_utils import chunk_text, clean_text

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)

def load_metadata(json_path: Path) -> Dict:
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def process_document(pdf_path: Path, json_path: Path, chunk_size: int, overlap: int) -> Dict:
    doc_id = pdf_path.stem
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    metadata = load_metadata(json_path)

    chunk_objs: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        chunk_objs.append(
            {
                "id": f"{doc_id}::{idx}",
                "text": chunk,
                "metadata": {"doc_id": doc_id, **metadata},
            }
        )

    return {
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "metadata": metadata,
        "full_text": cleaned,
        "chunks": chunk_objs,
    }

def find_pairs(input_dir: Path) -> List[Dict[str, Path]]:
    pairs = []
    for pdf_file in input_dir.glob("*.pdf"):
        json_file = input_dir / f"{pdf_file.stem}.json"
        pairs.append({"pdf": pdf_file, "json": json_file})
    return pairs

def preprocess_corpus(input_dir: Path, out_path: Path, chunk_size: int, overlap: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(input_dir)
    if not pairs:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    with out_path.open("w", encoding="utf-8") as outfile:
        for pair in pairs:
            doc = process_document(pair["pdf"], pair["json"], chunk_size, overlap)
            outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(f"Processed {pair['pdf'].name} -> {out_path.name}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess PDF + metadata to JSONL.")
    parser.add_argument("--input_dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/processed_docs.jsonl"))
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    preprocess_corpus(args.input_dir, args.out, args.chunk_size, args.overlap)


