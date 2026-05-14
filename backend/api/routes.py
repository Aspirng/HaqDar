from flask import Blueprint, request, jsonify
from rag.llm import query_gemini

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
        
    user_query = data['query']
    
    # TODO: Wait for Salman to finish the FAISS retriever.
    # For now, we will use mock chunks to test the API layer in isolation.
    mock_chunks = [
        {
            "text": "Section 302 PPC — whoever commits murder shall be punished with death.",
            "law": "Pakistan Penal Code 1860", 
            "section_ref": "302", 
            "doc_type": "criminal"
        }
    ]
    
    try:
        # Call the LLM logic
        response = query_gemini(user_query, mock_chunks)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
