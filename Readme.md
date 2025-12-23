python data_preprocessing.py --input_dir data --out artifacts/processed_docs.jsonl

python rag_pipeline.py ingest --processed artifacts/processed_docs.jsonl

python rag_pipeline.py query --question "What is Bancassurance?" --model gpt-4o-mini

python scripts/3_rag_pipeline.py ingest --processed artifacts/processed_docs.jsonl --persist_dir chroma_store --collection data

python evaluation_llm_judge.py --qa artifacts/qa_dataset.jsonl --pred artifacts/model_outputs.jsonl --model gpt-4o-mini

python scripts/4_evaluation_llm_judge.py --qa artifacts/qa_dataset.jsonl --pred artifacts/model_outputs.jsonl --model gpt-4o-mini --provider openai

python qa_dataset_generator.py --model gpt-4o-mini

python evaluation_llm_judge.py --qa artifacts/qa_dataset.jsonl --pred artifacts/model_outputs.jsonl --model gpt-4o-mini