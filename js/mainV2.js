/**
 * Main JavaScript functionality for News Analyzer Pro
 */

class NewsAnalyzer {
    constructor() {
        this.currentTab = 'url';
        this.isAnalyzing = false;
        this.apiEndpoint = '/api/analyze';
        
        this.initializeEventListeners();
        this.initializeDemoData();
    }
    
    initializeEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => this.switchTab(e));
        });
        
        // Analysis button
        const analyzeButton = document.querySelector('.analyze-button');
        if (analyzeButton) {
            analyzeButton.addEventListener('click', () => this.analyzeArticle());
        }
        
        // Form submission handling
        document.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.analyzeArticle();
            }
        });
        
        // Input validation
        document.querySelectorAll('.input-field').forEach(input => {
            input.addEventListener('input', () => this.validateInput());
        });
    }
    
    initializeDemoData() {
        // Pre-populate with example URL for demo
        const urlInput = document.getElementById('article-url');
        if (urlInput && !urlInput.value) {
            urlInput.value = 'https://example-news.com/renewable-energy-policy-update';
        }
    }
    
    switchTab(event) {
        const tabName = event.target.textContent.includes('URL') ? 'url' :
                       event.target.textContent.includes('Text') ? 'text' : 'multi';
        
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Remove active class from all buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Show selected tab and mark button as active
        document.getElementById(`${tabName}-tab`).classList.add('active');
        event.target.classList.add('active');
        
        this.currentTab = tabName;
        this.validateInput();
    }
    
    validateInput() {
        const content = this.getInputContent();
        const analyzeButton = document.querySelector('.analyze-button');
        
        if (analyzeButton) {
            analyzeButton.disabled = !content.trim() || this.isAnalyzing;
        }
    }
    
    getInputContent() {
        switch(this.currentTab) {
            case 'url':
                return document.getElementById('article-url')?.value || '';
            case 'text':
                return document.getElementById('article-text')?.value || '';
            case 'multi':
                return document.getElementById('multi-urls')?.value || '';
            default:
                return '';
        }
    }
    
    getSelectedOptions() {
        const summaryStyle = document.querySelector('input[name="summary-style"]:checked')?.value || 'neutral';
        const analysisDepth = document.querySelector('input[name="analysis-depth"]:checked')?.value || 'standard';
        const focusAreas = Array.from(document.querySelectorAll('input[name="focus-areas"]:checked'))
            .map(cb => cb.value);
        
        return { summaryStyle, analysisDepth, focusAreas };
    }
    
    async analyzeArticle() {
        const content = this.getInputContent();
        
        if (!content.trim()) {
            this.showAlert('Please enter a URL, text, or multiple URLs to analyze.', 'warning');
            return;
        }
        
        if (this.isAnalyzing) {
            return;
        }
        
        this.isAnalyzing = true;
        const options = this.getSelectedOptions();
        
        try {
            // Show loading state
            this.showLoading(true);
            this.hideResults();
            
            // In a real implementation, this would call your backend API
            // For demo purposes, we'll simulate the API call
            const analysisResult = await this.simulateAPICall(content, options);
            
            // Update UI with results
            this.displayResults(analysisResult);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showAlert('Analysis failed. Please try again.', 'error');
        } finally {
            this.isAnalyzing = false;
            this.showLoading(false);
            this.validateInput();
        }
    }
    
    async simulateAPICall(content, options) {
        // Simulate realistic API response time
        await this.delay(2500 + Math.random() * 1500);
        
        // Generate realistic demo data
        return {
            summary: this.generateDemoSummary(options.summaryStyle, content),
            bias_analysis: this.generateDemoBiasAnalysis(),
            sentiment: this.generateDemoSentiment(),
            key_points: this.generateDemoKeyPoints(),
            bias_indicators: this.generateDemoBiasIndicators(),
            timestamp: new Date().toISOString(),
            status: 'success'
        };
    }
    
    generateDemoSummary(style, content) {
        const summaries = {
            neutral: "The article discusses recent developments in renewable energy policy, presenting multiple perspectives from industry experts and government officials. Key stakeholders have varying opinions on the proposed timeline for implementation, with some expressing optimism about economic benefits while others raise concerns about feasibility and cost implications for traditional energy sectors.",
            
            facts: "• New renewable energy bill proposed in Congress with $50 billion budget allocation\n• Implementation timeline spans 2025-2030 period\n• Projected job creation: 500,000 new positions across clean energy sector\n• Opposition from 3 major fossil fuel companies citing economic concerns\n• Support from 15 environmental organizations and labor unions\n• Previous similar legislation failed in 2019 due to insufficient bipartisan support",
            
            simple: "The government wants to use more clean energy like wind and solar power instead of coal and oil. This plan could create lots of new jobs for people, but some companies that sell oil and coal don't like this idea. The plan would take about 5 years to finish and cost a lot of money. Some people think it's a good idea for the environment, while others worry about the cost.",
            
            detailed: "This comprehensive policy analysis reveals a complex legislative landscape surrounding renewable energy transition. The proposed legislation represents a significant paradigm shift in national energy strategy, with substantial economic implications across multiple industrial sectors. The article presents generally balanced reporting, though subtle linguistic choices suggest editorial sympathy toward environmental concerns. Key stakeholders include traditional labor unions (historically skeptical but now supportive), environmental advocacy groups (unanimously supportive), and fossil fuel corporations (predictably opposed). The policy framework includes specific provisions for workforce retraining, regional economic transition support, and graduated implementation timelines designed to minimize economic disruption while maximizing environmental benefits."
        };
        
        return summaries[style] || summaries.neutral;
    }
    
    generateDemoBiasAnalysis() {
        return {
            overall_bias_score: (Math.random() * 3 + 3).toFixed(1), // 3-6 range
            political_lean: ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right'][Math.floor(Math.random() * 5)],
            credibility_score: (Math.random() * 2 + 7).toFixed(1), // 7-9 range
            political_confidence: Math.random() * 0.4 + 0.6 // 0.6-1.0 range
        };
    }
    
    generateDemoSentiment() {
        const positive = Math.random() * 40 + 25; // 25-65%
        const negative = Math.random() * 25 + 10;  // 10-35%
        const neutral = 100 - positive - negative;
        
        return {
            positive: positive.toFixed(1),
            neutral: Math.max(neutral, 0).toFixed(1),
            negative: negative.toFixed(1)
        };
    }
    
    generateDemoKeyPoints() {
        const pointSets = [
            [
                "Article cites multiple credible sources from government agencies and academic institutions",
                "Balanced presentation of economic benefits and potential challenges",
                "Specific timeline and budget figures provided with contextual analysis",
                "Multiple stakeholder perspectives represented in the reporting"
            ],
            [
                "Comprehensive coverage of policy implications across different sectors",
                "Historical context provided for similar legislative attempts",
                "Expert commentary from both supporters and critics included",
                "Regional impact analysis demonstrates thorough research"
            ]
        ];
        
        return pointSets[Math.floor(Math.random() * pointSets.length)];
    }
    
    generateDemoBiasIndicators() {
        const indicatorSets = [
            [
                "Slight preference for renewable energy benefits in headline construction",
                "More detailed coverage of supporter arguments versus opposition concerns (60/40 ratio)",
                "Emotive language detected in environmental impact descriptions",
                "Source selection shows environmental expert preference over industry representatives"
            ],
            [
                "Balanced quote selection with equal representation from all stakeholders",
                "Minimal use of loaded or emotionally charged language",
                "Comprehensive fact-checking with multiple source verification",
                "Clear distinction maintained between reported facts and editorial analysis"
            ]
        ];
        
        return indicatorSets[Math.floor(Math.random() * indicatorSets.length)];
    }
    
    displayResults(data) {
        try {
            // Update summary
            const summaryContent = document.getElementById('summary-content');
            if (summaryContent) {
                summaryContent.textContent = data.summary;
            }
            
            // Update bias scores
            this.updateBiasScores(data.bias_analysis);
            
            // Update sentiment analysis
            this.updateSentimentBars(data.sentiment);
            
            // Update key points and bias indicators
            this.populateKeyPoints(data.key_points, 'key-points');
            this.populateKeyPoints(data.bias_indicators, 'bias-indicators');
            
            // Show results with smooth animation
            this.showResults();
            
        } catch (error) {
            console.error('Error displaying results:', error);
            this.showAlert('Error displaying results. Please try again.', 'error');
        }
    }
    
    updateBiasScores(biasData) {
        const overallEl = document.getElementById('overall-bias');
        const politicalEl = document.getElementById('political-lean');
        const credibilityEl = document.getElementById('credibility');
        
        if (overallEl) {
            const score = parseFloat(biasData.overall_bias_score);
            overallEl.textContent = score.toFixed(1);
            
            // Update color based on score
            overallEl.className = 'bias-score ' + 
                (score < 4 ? 'low' : score < 6 ? 'medium' : 'high');
        }
        
        if (politicalEl) {
            politicalEl.textContent = biasData.political_lean;
        }
        
        if (credibilityEl) {
            credibilityEl.textContent = biasData.credibility_score;
        }
    }
    
    updateSentimentBars(sentiment) {
        const elements = {
            'positive-percent': sentiment.positive + '%',
            'neutral-percent': sentiment.neutral + '%',
            'negative-percent': sentiment.negative + '%'
        };
        
        // Update percentages
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
        
        // Animate bars with delay for better UX
        setTimeout(() => {
            this.animateSentimentBar('positive-bar', sentiment.positive);
            this.animateSentimentBar('neutral-bar', sentiment.neutral);
            this.animateSentimentBar('negative-bar', sentiment.negative);
        }, 300);
    }
    
    animateSentimentBar(barId, percentage) {
        const bar = document.getElementById(barId);
        if (bar) {
            bar.style.width = percentage + '%';
        }
    }
    
    populateKeyPoints(points, containerId) {
        const container = document.getElementById(containerId);
        if (!container || !points) return;
        
        container.innerHTML = '';
        
        points.forEach((point, index) => {
            const li = document.createElement('li');
            li.textContent = point;
            li.style.opacity = '0';
            li.style.transform = 'translateY(10px)';
            container.appendChild(li);
            
            // Staggered animation
            setTimeout(() => {
                li.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                li.style.opacity = '1';
                li.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }
    
    showLoading(show) {
        const loadingElement = document.getElementById('loading');
        const analyzeButton = document.querySelector('.analyze-button');
        
        if (loadingElement) {
            loadingElement.style.display = show ? 'block' : 'none';
        }
        
        if (analyzeButton) {
            analyzeButton.disabled = show;
            analyzeButton.textContent = show ? '🔄 Analyzing...' : '🚀 Analyze Article';
        }
    }
    
    showResults() {
        const resultsElement = document.getElementById('results');
        if (resultsElement) {
            resultsElement.style.display = 'block';
            resultsElement.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
    
    hideResults() {
        const resultsElement = document.getElementById('results');
        if (resultsElement) {
            resultsElement.style.display = 'none';
        }
    }
    
    showAlert(message, type = 'info') {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            background: ${type === 'error' ? '#dc3545' : type === 'warning' ? '#ffc107' : '#007bff'};
            color: white;
            border-radius: 8px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 400px;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
        `;
        alert.textContent = message;
        
        document.body.appendChild(alert);
        
        // Show alert
        setTimeout(() => {
            alert.style.opacity = '1';
            alert.style.transform = 'translateX(0)';
        }, 100);
        
        // Auto-remove alert
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.parentNode.removeChild(alert);
                }
            }, 300);
        }, 4000);
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.newsAnalyzer = new NewsAnalyzer();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NewsAnalyzer;
}


# js/api.js
/**
 * API integration for News Analyzer Pro
 */

class NewsAnalyzerAPI {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
        this.timeout = 30000; // 30 second timeout
    }
    
    async analyzeArticle(content, options = {}) {
        try {
            const response = await this.makeRequest('/analyze', {
                method: 'POST',
                body: JSON.stringify({
                    content,
                    options,
                    input_type: this.detectInputType(content)
                })
            });
            
            return response;
        } catch (error) {
            console.error('API Analysis Error:', error);
            throw new Error(`Analysis failed: ${error.message}`);
        }
    }
    
    async fetchArticleFromURL(url) {
        try {
            const response = await this.makeRequest('/fetch', {
                method: 'POST',
                body: JSON.stringify({ url })
            });
            
            return response.content;
        } catch (error) {
            console.error('URL Fetch Error:', error);
            throw new Error(`Failed to fetch article: ${error.message}`);
        }
    }
    
    async batchAnalyze(urls) {
        try {
            const response = await this.makeRequest('/batch-analyze', {
                method: 'POST',
                body: JSON.stringify({ urls })
            });
            
            return response;
        } catch (error) {
            console.error('Batch Analysis Error:', error);
            throw new Error(`Batch analysis failed: ${error.message}`);
        }
    }
    
    async getHealthStatus() {
        try {
            const response = await this.makeRequest('/health');
            return response;
        } catch (error) {
            console.error('Health Check Error:', error);
            return { status: 'error', message: error.message };
        }
    }
    
    detectInputType(content) {
        const urlPattern = /^https?:\/\/.+/;
        const multiUrlPattern = /^https?:\/\/.+(\n|\r\n)https?:\/\/.+/;
        
        if (multiUrlPattern.test(content.trim())) {
            return 'multi_url';
        } else if (urlPattern.test(content.trim())) {
            return 'url';
        } else {
            return 'text';
        }
    }
    
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        
        const config = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                ...options.headers
            },
            ...options
        };
        
        // Add timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        config.signal = controller.signal;
        
        try {
            const response = await fetch(url, config);
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - please try again');
            }
            
            throw error;
        }
    }
    
    // Utility method for retry logic
    async withRetry(fn, maxRetries = 3, delay = 1000) {
        let lastError;
        
        for (let i = 0; i < maxRetries; i++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;
                
                if (i < maxRetries - 1) {
                    await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
                }
            }
        }
        
        throw lastError;
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.NewsAnalyzerAPI = NewsAnalyzerAPI;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = NewsAnalyzerAPI;
}


# tests/test_bias_detector.py
"""
Unit tests for bias detector functionality
"""

import unittest
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from bias_detector import BiasDetector

class TestBiasDetector(unittest.TestCase):
    def setUp(self):
        self.detector = BiasDetector()
    
    def test_analyze_bias_basic(self):
        """Test basic bias analysis functionality"""
        text = "This is a neutral news article about recent events."
        result = self.detector.analyze_bias(text)
        
        self.assertIn('overall_bias_score', result)
        self.assertIn('political_lean', result)
        self.assertIn('credibility_score', result)
        self.assertIn('sentiment', result)
    
    def test_detect_political_lean(self):
        """Test political lean detection"""
        left_text = "Progressive policies promote social justice and equality for marginalized communities."
        right_text = "Conservative values emphasize traditional free market individual responsibility."
        neutral_text = "The government announced new policies yesterday."
        
        left_result = self.detector._detect_political_lean(left_text)
        right_result = self.detector._detect_political_lean(right_text)
        neutral_result = self.detector._detect_political_lean(neutral_text)
        
        self.assertEqual(left_result['direction'], 'Left')
        self.assertEqual(right_result['direction'], 'Right')
        self.assertEqual(neutral_result['direction'], 'Center')
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        positive_text = "This excellent news brings great benefits to everyone."
        negative_text = "This terrible situation causes awful problems for people."
        
        pos_result = self.detector._analyze_sentiment(positive_text)
        neg_result = self.detector._analyze_sentiment(negative_text)
        
        self.assertGreater(pos_result['positive'], pos_result['negative'])
        self.assertGreater(neg_result['negative'], neg_result['positive'])
    
    def test_emotional_language_detection(self):
        """Test emotional language detection"""
        emotional_text = "This devastating and shocking news is absolutely unbelievable!"
        neutral_text = "The report indicates changes in the current situation."
        
        emotional_score = self.detector._detect_emotional_language(emotional_text)
        neutral_score = self.detector._detect_emotional_language(neutral_text)
        
        self.assertGreater(emotional_score, neutral_score)
    
    def test_bias_indicators(self):
        """Test bias indicator identification"""
        biased_text = "Sources claim this outrageous policy will completely destroy our economy."
        result = self.detector._identify_bias_indicators(biased_text)
        
        self.assertIsInstance(result, list)
        self.assertTrue(any("Emotive language" in indicator for indicator in result))

if __name__ == '__main__':
    unittest.main()


# tests/test_summarizer.py
"""
Unit tests for news summarizer functionality
"""

import unittest
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from summarizer import NewsSummarizer

class TestNewsSummarizer(unittest.TestCase):
    def setUp(self):
        self.summarizer = NewsSummarizer()
        self.sample_text = """
        The renewable energy sector has seen unprecedented growth in recent years, 
        with solar and wind power leading the charge. According to industry reports, 
        renewable energy capacity increased by 25% last year alone. Government 
        policies have played a crucial role in this expansion, providing tax 
        incentives and regulatory support. However, challenges remain, including 
        grid integration issues and storage limitations. Environmental groups 
        praise the progress, while traditional energy companies express concerns 
        about market disruption. The transition represents a significant shift 
        in how we generate and consume energy.
        """
    
    def test_neutral_summary(self):
        """Test neutral summary generation"""
        result = self.summarizer.summarize(self.sample_text, 'neutral')
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertTrue(len(result) < len(self.sample_text))
    
    def test_facts_summary(self):
        """Test facts-only summary generation"""
        result = self.summarizer.summarize(self.sample_text, 'facts')
        
        self.assertIsInstance(result, str)
        # Facts summary should contain bullet points
        self.assertTrue('•' in result or '25%' in result)
    
    def test_simple_summary(self):
        """Test simple (child-friendly) summary generation"""
        result = self.summarizer.summarize(self.sample_text, 'simple')
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_detailed_summary(self):
        """Test detailed summary generation"""
        result = self.summarizer.summarize(self.sample_text, 'detailed')
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_short_text_handling(self):
        """Test handling of very short text"""
        short_text = "Short text."
        result = self.summarizer.summarize(short_text)
        
        self.assertIn("too short", result.lower())
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        dirty_text = "This   is    a   test\n\n\nwith    weird    spacing."
        cleaned = self.summarizer._clean_text(dirty_text)
        
        self.assertNotIn('   ', cleaned)  # No triple spaces
        self.assertNotIn('\n\n', cleaned)  # No double newlines
    
    def test_extractive_summary(self):
        """Test extractive summarization"""
        result = self.summarizer._extractive_summary(self.sample_text, max_sentences=2)
        
        self.assertIsInstance(result, str)
        sentences = result.split('.')
        self.assertTrue(len(sentences) <= 3)  # 2 sentences + empty string from final period

if __name__ == '__main__':
    unittest.main()


# docs/API.md
# News Analyzer Pro API Documentation

## Base URL
```
Production: https://your-domain.com/api
Development: http://localhost:5000/api
```

## Authentication
Currently, the API does not require authentication for demo purposes. In production, you would implement API key authentication.

## Endpoints

### POST /analyze
Analyze a news article for bias, sentiment, and generate summary.

**Request Body:**
```json
{
  "content": "Article text or URL",
  "options": {
    "style": "neutral|facts|simple|detailed",
    "depth": "basic|standard|advanced",
    "focus_areas": ["political", "emotional", "factual", "sources"]
  },
  "input_type": "text|url|multi_url"
}
```

**Response:**
```json
{
  "summary": "Generated summary based on selected style",
  "bias_analysis": {
    "overall_bias_score": 6.2,
    "political_lean": "Center-Left",
    "political_confidence": 0.75,
    "credibility_score": 8.1,
    "emotional_language_score": 4.3,
    "bias_indicators": [
      "Emotive language detected: devastating, shocking",
      "Source selection shows 60% environmental experts"
    ],
    "recommendations": [
      "Consider reading multiple sources on this topic"
    ]
  },
  "sentiment": {
    "positive": 45.2,
    "neutral": 34.8,
    "negative": 20.0
  },
  "key_points": [
    "Article cites multiple credible sources",
    "Balanced presentation of viewpoints"
  ],
  "timestamp": "2025-01-20T10:30:00Z",
  "status": "success"
}
```

### POST /fetch
Fetch article content from URL.

**Request Body:**
```json
{
  "url": "https://example.com/article"
}
```

**Response:**
```json
{
  "content": "Extracted article text",
  "title": "Article title",
  "author": "Author name",
  "publish_date": "2025-01-20",
  "source": "example.com"
}
```

### POST /batch-analyze
Analyze multiple articles for cross-source bias comparison.

**Request Body:**
```json
{
  "urls": [
    "https://source1.com/article",
    "https://source2.com/article"
  ],
  "options": {
    "style": "neutral",
    "comparison_mode": true
  }
}
```

### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-20T10:30:00Z",
  "version": "1.0.0"
}
```

## Error Responses

All error responses follow this format:
```json
{
  "error": "Error description",
  "details": "Additional error details",
  "status": "error",
  "timestamp": "2025-01-20T10:30:00Z"
}
```

### Common Error Codes
- **400 Bad Request**: Invalid input data
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: External API unavailable

## Rate Limits
- **60 requests per minute** per IP address
- **1000 requests per hour** per IP address

## Examples

### Analyze URL
```bash
curl -X POST https://your-domain.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "content": "https://example.com/news-article",
    "options": {
      "style": "neutral",
      "depth": "standard",
      "focus_areas": ["political", "emotional"]
    }
  }'
```

### Analyze Text
```bash
curl -X POST https://your-domain.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your article text here...",
    "options": {
      "style": "simple",
      "depth": "basic"
    }
  }'
```


# docs/CONTRIBUTING.md
# Contributing to News Analyzer Pro

We welcome contributions from the community! This guide will help you get started.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

### Setup Development Environment
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/news-analyzer-pro.git
cd news-analyzer-pro

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up Node.js dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run tests to ensure everything works
python -m pytest tests/
npm test
```

## 📋 How to Contribute

### Reporting Issues
- Use the [GitHub Issues](https://github.com/yourusername/news-analyzer-pro/issues) page
- Search existing issues before creating new ones
- Use the provided issue templates
- Include detailed reproduction steps
- Add relevant labels

### Submitting Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Your Changes**
   - Follow the coding standards below
   - Write tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run Python tests
   python -m pytest tests/ -v
   
   # Run JavaScript tests
   npm test
   
   # Run linting
   flake8 python/ --max-line-length=88
   npm run lint
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new bias detection algorithm"
   # Follow conventional commit format
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   # Create PR on GitHub
   ```

## 🎯 Development Guidelines

### Code Style

**Python:**
- Follow PEP 8 with 88 character line limit
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Use meaningful variable and function names

**JavaScript:**
- Use ES6+ features
- Follow consistent indentation (2 spaces)
- Use meaningful variable names
- Add JSDoc comments for complex functions

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Testing Requirements
- All new features must include tests
- Maintain minimum 80% code coverage
- Tests should be clear and descriptive
- Include both positive and negative test cases

### Documentation
- Update README.md for major changes
- Add/update API documentation
- Include inline code comments
- Update changelog

## 🔄 Pull Request Process

1. **Before Submitting:**
   - Ensure all tests pass
   - Update documentation
   - Rebase on latest main branch
   - Ensure commit messages follow convention

2. **PR Description Should Include:**
   - Clear description of changes
   - Link to related issues
   - Screenshots for UI changes
   - Breaking changes noted

3. **Review Process:**
   - At least one maintainer review required
   - Address all review comments
   - Ensure CI/CD passes
   - Squash commits if requested

## 🌟 Areas for Contribution

### High Priority
- [ ] Improve bias detection algorithms
- [ ] Add support for more news sources
- [ ] Enhance mobile responsiveness
- [ ] Add caching layer
- [ ] Implement user authentication

### Medium Priority
- [ ] Add more summary styles
- [ ] Improve error handling
- [ ] Add export functionality
- [ ] Implement rate limiting
- [ ] Add more test coverage

### Good First Issues
- [ ] Fix typos in documentation
- [ ] Add more unit tests
- [ ] Improve CSS styling
- [ ] Add loading animations
- [ ] Update dependencies

## 📞 Getting Help

- **Discord**: [Join our community](https://discord.gg/your-server)
- **Email**: developer@your-domain.com
- **GitHub Discussions**: Use for questions and ideas

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given appropriate GitHub repository permissions

Thank you for contributing to News Analyzer Pro!


# .github/ISSUE_TEMPLATE/bug_report.md
---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Screenshots
If applicable, add screenshots to help explain your problem.

## Environment
- OS: [e.g., Windows 10, macOS 11.0, Ubuntu 20.04]
- Browser: [e.g., Chrome 96, Firefox 95, Safari 15]
- Device: [e.g., Desktop, iPhone 12, Samsung Galaxy]
- Version: [e.g., 1.0.0]

## Additional Context
Add any other context about the problem here.

## Error Logs
If applicable, paste any relevant error logs here.
```
Paste error logs here
```

## Possible Solution
If you have an idea of how to fix this bug, please describe it here.
