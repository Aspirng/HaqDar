from flask import Blueprint, request, jsonify
from rag.llm import query_gemini
from rag.retriever import retrieve

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
        
    user_query = data['query']
    
    try:
        # Retrieve real context chunks from the vector store
        chunks = retrieve(user_query, top_k=5)
        
        # Call the LLM logic with the real retrieved chunks
        response = query_gemini(user_query, chunks)
        return jsonify(response)
    except FileNotFoundError as fnfe:
        return jsonify({"error": str(fnfe)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
