from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
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

# Import utilities
from utils.file_handler import FileHandler
from utils.processor import DocumentProcessor
from utils.retrieval_evaluator import RetrievalEvaluator

app = Flask(__name__)
CORS(app)

# Swagger UI configuration
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.yaml'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "RAG Chatbot API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Available models
MODELS = ['gemma3:1b', 'qwen:latest', 'Phi3:latest', 'deepseek-r1:1.5b', 'smollm:latest']

# ChromaDB configuration
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / 'scripts' / 'chroma_store')
COLLECTION_NAME = 'data'
UPLOAD_DIR = Path(__file__).parent / 'uploads'

# Initialize utilities
file_handler = FileHandler(UPLOAD_DIR)
doc_processor = DocumentProcessor(Path(CHROMA_PERSIST_DIR))
retrieval_evaluator = RetrievalEvaluator()

def get_chroma_collection(collection_name=None):
    """Load or create ChromaDB collection"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        emb_fn = get_embedding_function('sentence-transformers', 'all-MiniLM-L6-v2')
        collection = client.get_or_create_collection(
            name=collection_name or COLLECTION_NAME,
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
        "top_k": 3,
        "collection_name": "data" (optional)
    }
    """
    try:
        data = request.json
        question = data.get('question', '').strip()
        model = data.get('model', 'gemma3:1b')
        temperature = data.get('temperature', 0.7)
        top_k = data.get('top_k', 3)
        collection_name = data.get('collection_name', COLLECTION_NAME)

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        if model not in MODELS:
            return jsonify({'error': f'Model must be one of: {", ".join(MODELS)}'}), 400

        start_time = time.time()
        
        # Load ChromaDB collection
        collection = get_chroma_collection(collection_name)
        print(f"✓ ChromaDB collection loaded: {collection_name}")
        
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
            'collection': collection_name,
            'latency': round(latency, 2),  
            'status': 'success'
        }), 200

    except Exception as e:
        print(f"✗ Error in /api/chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

# ==================== DOCUMENT UPLOAD ENDPOINTS ====================

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """
    Upload PDF/JSON files for processing
    
    Expects multipart/form-data with files
    """
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        result = file_handler.save_uploaded_files(files)
        
        return jsonify({
            'status': 'success',
            'session_id': result['session_id'],
            'files': result['files'],
            'errors': result['errors']
        }), 200
        
    except Exception as e:
        print(f"✗ Upload error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/process/<session_id>', methods=['POST'])
def process_uploaded_documents(session_id):
    """
    Process uploaded documents and ingest into ChromaDB
    
    Expected JSON (optional):
    {
        "collection_name": "my_docs",
        "chunk_size": 1000,
        "overlap": 200
    }
    """
    try:
        data = request.json or {}
        collection_name = data.get('collection_name', f'uploaded_{session_id}')
        chunk_size = data.get('chunk_size', 1000)
        overlap = data.get('overlap', 200)
        
        # Get session directory
        session_dir = UPLOAD_DIR / session_id
        if not session_dir.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        # Process and ingest
        result = doc_processor.process_session_files(
            session_dir=session_dir,
            collection_name=collection_name,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        if result['status'] == 'error':
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"✗ Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/collections', methods=['GET'])
def list_collections():
    """List all available ChromaDB collections"""
    try:
        collections = doc_processor.list_collections()
        return jsonify({
            'status': 'success',
            'collections': collections
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/collections/<collection_name>', methods=['DELETE'])
def delete_collection(collection_name):
    """Delete a ChromaDB collection"""
    try:
        success = doc_processor.delete_collection(collection_name)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Collection {collection_name} deleted'
            }), 200
        else:
            return jsonify({'error': 'Failed to delete collection'}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/sessions/<session_id>/files', methods=['GET'])
def get_session_files(session_id):
    """Get list of files in an upload session"""
    try:
        files = file_handler.get_session_files(session_id)
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'files': files
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete an upload session and its files"""
    try:
        success = file_handler.delete_session(session_id)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Session {session_id} deleted'
            }), 200
        else:
            return jsonify({'error': 'Failed to delete session'}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


# ==================== RETRIEVAL EVALUATION ENDPOINTS ====================

@app.route('/api/evaluate/retrieval', methods=['POST'])
def evaluate_retrieval():
    """
    Evaluate retrieval quality using IR metrics
    
    Expected JSON:
    {
        "collection_name": "data",
        "test_queries": [
            {
                "query": "What is HBL?",
                "relevant_doc_ids": ["doc1::0", "doc1::1"]
            }
        ],
        "top_k": 5
    }
    
    Returns metrics: MRR, Recall@K, Precision@K, NDCG@K
    """
    try:
        data = request.json
        collection_name = data.get('collection_name', COLLECTION_NAME)
        test_queries = data.get('test_queries', [])
        top_k = data.get('top_k', 5)
        
        if not test_queries:
            return jsonify({'error': 'test_queries required'}), 400
        
        # Get collection
        collection = get_chroma_collection(collection_name)
        
        # Evaluate
        results = retrieval_evaluator.evaluate_retrieval(
            collection=collection,
            test_queries=test_queries,
            top_k=top_k
        )
        
        return jsonify({
            'status': 'success',
            'collection': collection_name,
            **results
        }), 200
        
    except Exception as e:
        print(f"✗ Evaluation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500


# ==================== EXISTING ENDPOINTS ====================

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
