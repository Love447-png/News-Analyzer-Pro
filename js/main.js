# js/main.js
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
            throw new Error(`Failed to fetch article: ${error.message
