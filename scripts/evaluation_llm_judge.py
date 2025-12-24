from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
from llm_clients import client_from_env, safe_generate
from dotenv import load_dotenv
from text_utils import extract_json
load_dotenv(override=True)
import os
import re


JUDGE_PROMPT = """You are evaluating a model's answer.

Question:
{question}

Reference answer:
{reference}

Model answer:
{candidate}

Instructions:
- Score from 1 to 5.
- Use the full scale.
- Partial correctness is allowed.
- You may include brief analysis, but ensure a JSON object is present.

Rubric:
5 = Fully correct and complete
4 = Mostly correct, minor omissions or imprecision
3 = Partially correct, key ideas present but incomplete
2 = Mostly incorrect, minimal relevant content
1 = Incorrect or irrelevant

Return your decision in JSON.
If unsure, still assign the closest score.

Preferred format:
{{
  "score": <integer 1-5>,
  "reason": "<one or two short sentences>"
}}
"""

def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def to_map(items: List[Dict], key: str) -> Dict[str, Dict]:
    return {item[key]: item for item in items if key in item}


def judge_one(
    llm_model: str,
    provider: str,
    question: str,
    reference: str,
    candidate: str,
) -> Dict:
    llm = client_from_env(model=llm_model, provider=provider)
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        candidate=candidate,
    )

    raw = safe_generate(
        llm,
        prompt,
        max_tokens=200,
        temperature=0.2,  # allow slight flexibility
    )

    # Try strict JSON first
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract score manually
        score_match = re.search(r'([1-5])', raw)
        parsed = {
            "score": int(score_match.group(1)) if score_match else 1,
            "reason": raw.strip()[:200],
        }

    score = int(parsed.get("score", 1))
    return {
        "score": max(1, min(score, 5)),
        "reason": parsed.get("reason", ""),
    }


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
                "reason": extract_json(result["reason"]),
            }
            f.write(json.dumps(record,indent=4,ensure_ascii=False) + "\n")
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
    parser.add_argument("--model", type=str, default=os.getenv("model"))
    parser.add_argument("--provider", type=str, default="ollama", choices=["openai", "ollama"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.qa, args.pred, args.out, args.model, args.provider)


