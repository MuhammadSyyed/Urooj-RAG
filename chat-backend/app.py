from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import time
import chromadb
from dotenv import load_dotenv

load_dotenv(override=True)

# Add parent directory to path to import scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rag_pipeline import retrieve, generate_answer, get_embedding_function

app = Flask(__name__)
CORS(app)

# Available models
MODELS = ['gemma3:1b', 'qwen:latest', 'Phi3:latest', 'deepseek-r1:1.5b', 'smollm:latest']

# ChromaDB configuration
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / 'scripts' / 'chroma_store')
COLLECTION_NAME = 'data'

def get_chroma_collection():
    """Load or create ChromaDB collection"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        emb_fn = get_embedding_function('sentence-transformers', 'all-MiniLM-L6-v2')
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn
        )
        return collection
    except Exception as e:
        print(f"Error loading ChromaDB collection: {str(e)}")
        raise

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    return jsonify({
        'models': MODELS
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Process a chat message and return an answer with sources
    
    Expected JSON:
    {
        "question": "What is HBL?",
        "model": "gemma3:1b",
        "temperature": 0.7,
        "top_k": 3
    }
    """
    try:
        data = request.json
        question = data.get('question', '').strip()
        model = data.get('model', 'gemma3:1b')
        temperature = data.get('temperature', 0.7)
        top_k = data.get('top_k', 3)

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        if model not in MODELS:
            return jsonify({'error': f'Model must be one of: {", ".join(MODELS)}'}), 400

        start_time = time.time()
        
        # Load ChromaDB collection
        collection = get_chroma_collection()
        print(f"✓ ChromaDB collection loaded")
        
        # Retrieve relevant documents
        results = retrieve(collection, question, top_k=top_k)
        print(f"✓ Retrieved {len(results)} documents for query: {question}")
        
        # Extract sources
        sources = []
        for result in results:
            sources.append({
                'id': result.get('id', 'Unknown'),
                'source': result.get('metadata', {}).get('source', 'Unknown'),
                'text': result.get('text', '')[:500]  # First 500 chars for preview
            })

        # Generate answer using the selected model
        print(f"✓ Generating answer with model: {model}")
        answer = generate_answer(
            query=question,
            contexts=results,
            llm_model=model,
            provider='ollama',
            temperature=temperature
        )
        
        print(f"✓ Generated answer: {answer[:100]}...")

        latency = time.time() - start_time
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'model': model,
            'latency': round(latency, 2),  
            'status': 'success'
        }), 200

    except Exception as e:
        print(f"✗ Error in /api/chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get default settings"""
    return jsonify({
        'defaultModel': 'gemma3:1b',
        'temperature': 0.7,
        'topK': 3,
        'models': MODELS
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

@app.route('/api/debug/chroma', methods=['POST'])
def debug_chroma():
    """Debug endpoint to test ChromaDB"""
    try:
        data = request.json
        question = data.get('question', 'What is HBL?')
        top_k = data.get('top_k', 3)
        
        collection = get_chroma_collection()
        results = retrieve(collection, question, top_k=top_k)
        
        return jsonify({
            'question': question,
            'num_results': len(results),
            'results': results,
            'status': 'success'
        }), 200
    except Exception as e:
        print(f"✗ ChromaDB Debug Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
