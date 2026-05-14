from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

from api.routes import api_bp

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS so the frontend can hit these endpoints
CORS(app)

# Register the API routes
app.register_blueprint(api_bp)

@app.route('/')
def index():
    return "HaqDar Backend is running!"

if __name__ == '__main__':
    # Start the server
    app.run(host='0.0.0.0', port=5000, debug=True)
