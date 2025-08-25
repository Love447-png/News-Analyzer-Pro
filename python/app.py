### python/app.py
"""
Main Flask application for News Analyzer Pro
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

from bias_detector import BiasDetector
from summarizer import NewsSummarizer
from config import Config

load_dotenv()
app = Flask(__name__, 
           static_folder='../css', 
           static_url_path='/css',
           template_folder='../')
app.config.from_object(Config)

CORS(app, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))

bias_detector = BiasDetector()
summarizer = NewsSummarizer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_article():
    """Main endpoint for article analysis"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or not data.get('content'):
            return jsonify({'error': 'No content provided'}), 400
        
        content = data['content']
        options = data.get('options', {})
        
        # Perform analysis
        summary = summarizer.summarize(content, options.get('style', 'neutral'))
        bias_analysis = bias_detector.analyze_bias(content)
        
        # Prepare response
        result = {
            'summary': summary,
            'bias_analysis': bias_analysis,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Analysis completed for content length: {len(content)}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting News Analyzer Pro on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
