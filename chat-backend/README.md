# Urooj RAG Chat Backend

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Flask server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### GET /api/models

Returns list of available models

**Response:**

```json
{
  "models": ["gemma3", "qwen", "phi3", "deepseek", "smollm"]
}
```

### POST /api/chat

Process a user question and return an answer with sources

**Request:**

```json
{
  "question": "What is HBL?",
  "model": "gemma3",
  "temperature": 0.7,
  "top_k": 3
}
```

**Response:**

```json
{
  "answer": "HBL is Habib Bank Limited...",
  "sources": ["HBL CarLoan - Training.json", "PB Training.json"],
  "score": 4,
  "model": "gemma3"
}
```

### GET /api/settings

Get default settings

**Response:**

```json
{
  "defaultModel": "gemma3",
  "temperature": 0.7,
  "topK": 3,
  "models": ["gemma3", "qwen", "phi3", "deepseek", "smollm"]
}
```

### GET /api/health

Health check endpoint

**Response:**

```json
{
  "status": "ok"
}
```

## Environment Variables

Create a `.env` file if needed:

```
FLASK_ENV=development
FLASK_DEBUG=1
```

## Dependencies

- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing support
- **chromadb**: Vector database for document retrieval
- **sentence-transformers**: Embedding models
- **python-dotenv**: Environment variable management

### Evaluation

# Install optional dependencies first

pip install rouge-score bert-score

# Evaluate with all metrics (including LLM judge)

python evaluation_multi_metric.py \
 --qa ../artifacts/qa_dataset.jsonl \
 --pred ../artifacts/gemma3_model_outputs.jsonl \
 --out ../artifacts/gemma3_multi_eval.jsonl \
 --model qwen:latest

# Skip LLM judge for faster evaluation (F1, ROUGE, BERTScore only)

python evaluation_multi_metric.py \
 --pred ../artifacts/deepseek_model_outputs.jsonl \
 --out ../artifacts/deepseek_multi_eval.jsonl \
 --skip-llm-judge
