python scripts/3_rag_pipeline.py ingest --processed artifacts/processed_docs.jsonl --persist_dir chroma_store --collection urooj_docs

python scripts/4_evaluation_llm_judge.py --qa artifacts/qa_dataset.jsonl --pred artifacts/model_outputs.jsonl --model gpt-4o-mini --provider openai