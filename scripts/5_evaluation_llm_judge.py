"""
Evaluate model responses using LLM-as-a-judge.

Inputs:
- Ground truth Q&A: artifacts/qa_dataset.jsonl
- Model outputs: artifacts/model_outputs.jsonl
    each line: {"qa_id": "...", "question": "...", "answer": "..."}  # answer = model prediction

Output:
- artifacts/eval_results.jsonl with judge score and rationale
- Prints aggregate metrics

Run:
    python evaluation_llm_judge.py --qa artifacts/qa_dataset.jsonl --pred artifacts/model_outputs.jsonl --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from llm_clients import client_from_env, safe_generate


JUDGE_PROMPT = """You are grading a model's answer.
Question: {question}
Reference answer: {reference}
Model answer: {candidate}

Score from 1 to 5 (5 = fully correct, 1 = incorrect). Be strict but fair.
Return only JSON: {{"score": <int>, "reason": "<short explanation>"}}"""


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def to_map(items: List[Dict], key: str) -> Dict[str, Dict]:
    return {item[key]: item for item in items if key in item}


def judge_one(llm_model: str, provider: str, question: str, reference: str, candidate: str) -> Dict:
    llm = client_from_env(model=llm_model, provider=provider)
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, candidate=candidate)
    raw = safe_generate(llm, prompt, max_tokens=150, temperature=0.0)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"score": 1, "reason": f"Could not parse judge output: {raw}"}
    score = int(parsed.get("score", 1))
    return {"score": max(1, min(score, 5)), "reason": parsed.get("reason", "")}


def evaluate(qa_path: Path, pred_path: Path, out_path: Path, llm_model: str, provider: str) -> None:
    gold = to_map(load_jsonl(qa_path), "qa_id")
    preds = load_jsonl(pred_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores: List[int] = []

    with out_path.open("w", encoding="utf-8") as f:
        for pred in preds:
            qa_id = pred.get("qa_id")
            gt = gold.get(qa_id)
            if not gt:
                continue
            result = judge_one(
                llm_model=llm_model,
                provider=provider,
                question=gt["question"],
                reference=gt["answer"],
                candidate=pred.get("answer", ""),
            )
            scores.append(result["score"])
            record = {
                "qa_id": qa_id,
                "question": gt["question"],
                "reference": gt["answer"],
                "candidate": pred.get("answer", ""),
                "score": result["score"],
                "reason": result["reason"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if scores:
            avg = sum(scores) / len(scores)
            print(f"Evaluated {len(scores)} items. Average score: {avg:.2f}/5")
        else:
            print("No overlapping qa_id between predictions and ground truth.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge evaluation.")
    parser.add_argument("--qa", type=Path, default=Path("../artifacts/qa_dataset.jsonl"))
    parser.add_argument("--pred", type=Path, default=Path("../artifacts/model_outputs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("../artifacts/eval_results.jsonl"))
    parser.add_argument("--model", type=str, default="gemma3:1b")
    parser.add_argument("--provider", type=str, default="ollama", choices=["openai", "ollama"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.qa, args.pred, args.out, args.model, args.provider)


