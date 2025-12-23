1. python data_preprocessing.py --input_dir data --out artifacts/processed_docs.jsonl
2. python qa_dataset_generator.py --model gemma3:1b
3. python rag_pipeline.py ingest --processed artifacts/processed_docs.jsonl --persist_dir chroma_store --collection data
4. python rag_pipeline.py query --question "What is Bancassurance?" --model gemma3:1b
5. python generate_model_outputs.py
6. python evaluation_llm_judge.py --qa artifacts/qa_dataset.jsonl --pred artifacts/model_outputs.jsonl --model gemma3:1b
v2
6. python evaluation_llm_judge.py --qa ../artifacts/qa_dataset.jsonl --out ../artifacts/gemma3_eval.jsonl  --pred ../artifacts/gemma3_model_outputs.jsonl --model gemma3:1b 





