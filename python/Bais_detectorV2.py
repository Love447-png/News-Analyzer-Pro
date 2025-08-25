"""
Advanced bias detection using NLP and ML techniques
"""

import os
import openai
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import Dict, List, Tuple

class BiasDetector:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Hugging Face models
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.emotion_analyzer = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base"
            )
        except Exception as e:
            print(f"Warning: Could not load HF models: {e}")
            self.sentiment_analyzer = None
            self.emotion_analyzer = None
        
        # Bias keywords dictionary
        self.bias_keywords = {
            'political_left': ['progressive', 'liberal', 'social justice', 'inequality', 'marginalized'],
            'political_right': ['conservative', 'traditional', 'free market', 'individual responsibility'],
            'emotional': ['devastating', 'shocking', 'outrageous', 'incredible', 'unbelievable'],
            'loaded': ['claims', 'alleges', 'supposedly', 'reportedly', 'apparently']
        }
    
    def analyze_bias(self, text: str) -> Dict:
        """
        Comprehensive bias analysis of text
        """
        try:
            # Basic metrics
            word_count = len(text.split())
            
            # Sentiment analysis
            sentiment_scores = self._analyze_sentiment(text)
            
            # Political lean detection
            political_lean = self._detect_political_lean(text)
            
            # Emotional language detection
            emotional_score = self._detect_emotional_language(text)
            
            # Source credibility (simulated)
            credibility_score = self._assess_credibility(text)
            
            # Overall bias score calculation
            overall_bias = self._calculate_overall_bias(
                sentiment_scores, emotional_score, political_lean
            )
            
            return {
                'overall_bias_score': round(overall_bias, 1),
                'political_lean': political_lean['direction'],
                'political_confidence': political_lean['confidence'],
                'credibility_score': round(credibility_score, 1),
                'sentiment': sentiment_scores,
                'emotional_language_score': round(emotional_score, 1),
                'word_count': word_count,
                'bias_indicators': self._identify_bias_indicators(text),
                'recommendations': self._generate_recommendations(overall_bias)
            }
            
        except Exception as e:
            return {
                'error': f'Bias analysis failed: {str(e)}',
                'overall_bias_score': 5.0,
                'political_lean': 'Center',
                'credibility_score': 7.0
            }
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using multiple approaches"""
        if self.sentiment_analyzer:
            try:
                # Split text into chunks for better analysis
                chunks = self._split_text(text, max_length=500)
                sentiments = []
                
                for chunk in chunks:
                    result = self.sentiment_analyzer(chunk)[0]
                    sentiments.append(result)
                
                # Aggregate results
                positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
                negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
                neutral_count = len(sentiments) - positive_count - negative_count
                
                total = len(sentiments)
                return {
                    'positive': round((positive_count / total) * 100, 1),
                    'neutral': round((neutral_count / total) * 100, 1),
                    'negative': round((negative_count / total) * 100, 1)
                }
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
        
        # Fallback to simple keyword-based analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'beneficial']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'harmful']
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            return {'positive': 33.3, 'neutral': 33.4, 'negative': 33.3}
        
        pos_percent = (pos_count / total_sentiment_words) * 100
        neg_percent = (neg_count / total_sentiment_words) * 100
        neu_percent = 100 - pos_percent - neg_percent
        
        return {
            'positive': round(pos_percent, 1),
            'neutral': round(neu_percent, 1),
            'negative': round(neg_percent, 1)
        }
    
    def _detect_political_lean(self, text: str) -> Dict:
        """Detect political lean using keyword analysis"""
        text_lower = text.lower()
        
        left_score = sum(1 for keyword in self.bias_keywords['political_left'] 
                        if keyword in text_lower)
        right_score = sum(1 for keyword in self.bias_keywords['political_right'] 
                         if keyword in text_lower)
        
        total_political = left_score + right_score
        
        if total_political == 0:
            return {'direction': 'Center', 'confidence': 0.5}
        
        left_ratio = left_score / total_political
        
        if left_ratio > 0.6:
            return {'direction': 'Left', 'confidence': left_ratio}
        elif left_ratio < 0.4:
            return {'direction': 'Right', 'confidence': 1 - left_ratio}
        else:
            return {'direction': 'Center', 'confidence': 0.5}
    
    def _detect_emotional_language(self, text: str) -> float:
        """Detect emotional/loaded language"""
        text_lower = text.lower()
        emotional_words = self.bias_keywords['emotional'] + self.bias_keywords['loaded']
        
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        # Return emotional language percentage scaled to 0-10
        emotional_ratio = (emotional_count / word_count) * 100
        return min(emotional_ratio * 2, 10.0)  # Scale and cap at 10
    
    def _assess_credibility(self, text: str) -> float:
        """Assess source credibility based on text characteristics"""
        factors = []
        
        # Length factor (longer articles tend to be more credible)
        word_count = len(text.split())
        length_factor = min(word_count / 500, 1.0)  # Normalize to 1.0 max
        factors.append(length_factor)
        
        # Citation/quote factor
        citation_indicators = ['according to', 'said', 'reported', 'source:', 'study shows']
        citation_count = sum(1 for indicator in citation_indicators if indicator in text.lower())
        citation_factor = min(citation_count / 3, 1.0)  # Normalize
        factors.append(citation_factor)
        
        # Fact-based language
        fact_indicators = ['data', 'research', 'study', 'analysis', 'evidence']
        fact_count = sum(1 for indicator in fact_indicators if indicator in text.lower())
        fact_factor = min(fact_count / 5, 1.0)
        factors.append(fact_factor)
        
        # Average all factors and scale to 0-10
        avg_factor = np.mean(factors)
        return 5.0 + (avg_factor * 5.0)  # Base score of 5, add up to 5 more
    
    def _calculate_overall_bias(self, sentiment: Dict, emotional: float, political: Dict) -> float:
        """Calculate overall bias score (0-10, where 5 is neutral)"""
        # Sentiment imbalance
        sentiment_balance = abs(sentiment['positive'] - sentiment['negative']) / 100
        
        # Political lean strength
        political_strength = political['confidence']
        
        # Emotional language weight
        emotional_weight = emotional / 10
        
        # Combine factors (higher = more biased)
        bias_score = 3.0 + (sentiment_balance * 2) + (political_strength * 2) + (emotional_weight * 3)
        
        return min(bias_score, 10.0)  # Cap at 10
    
    def _identify_bias_indicators(self, text: str) -> List[str]:
        """Identify specific bias indicators in the text"""
        indicators = []
        text_lower = text.lower()
        
        # Check for loaded language
        loaded_words = self.bias_keywords['emotional']
        found_loaded = [word for word in loaded_words if word in text_lower]
        if found_loaded:
            indicators.append(f"Emotive language detected: {', '.join(found_loaded[:3])}")
        
        # Check for political keywords
        for direction in ['political_left', 'political_right']:
            found_political = [word for word in self.bias_keywords[direction] if word in text_lower]
            if found_political:
                lean_name = direction.split('_')[1].capitalize()
                indicators.append(f"{lean_name}-leaning language: {', '.join(found_political[:2])}")
        
        # Check for lack of attribution
        attribution_words = ['according to', 'said', 'reported by', 'source']
        has_attribution = any(word in text_lower for word in attribution_words)
        if not has_attribution:
            indicators.append("Limited source attribution detected")
        
        # Check for absolute statements
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'completely']
        found_absolute = [word for word in absolute_words if word in text_lower]
        if found_absolute:
            indicators.append(f"Absolute language: {', '.join(found_absolute[:2])}")
        
        return indicators[:4]  # Return top 4 indicators
    
    def _generate_recommendations(self, bias_score: float) -> List[str]:
        """Generate recommendations based on bias analysis"""
        recommendations = []
        
        if bias_score < 4:
            recommendations.append("This article appears relatively neutral and balanced.")
        elif bias_score < 6:
            recommendations.append("Some bias detected. Consider reading multiple sources on this topic.")
        elif bias_score < 8:
            recommendations.append("Significant bias detected. Strongly recommend cross-referencing with other sources.")
        else:
            recommendations.append("High bias detected. Exercise caution and seek diverse perspectives.")
        
        recommendations.append("Always verify facts through multiple reputable sources.")
        recommendations.append("Consider the publication's editorial stance and target audience.")
        
        return recommendations
    
    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]


# python/summarizer.py
"""
Advanced news summarization using multiple AI techniques
"""

import os
import openai
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
from typing import List, Dict

class NewsSummarizer:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # Initialize Hugging Face summarization model
        try:
            self.hf_summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                max_length=150,
                min_length=50
            )
        except Exception as e:
            print(f"Warning: Could not load HF summarizer: {e}")
            self.hf_summarizer = None
    
    def summarize(self, text: str, style: str = 'neutral') -> str:
        """
        Generate summary based on specified style
        """
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            if len(cleaned_text.split()) < 50:
                return "Text too short for meaningful summarization."
            
            # Choose summarization method based on style
            if style == 'facts':
                return self._extract_facts_summary(cleaned_text)
            elif style == 'simple':
                return self._generate_simple_summary(cleaned_text)
            elif style == 'detailed':
                return self._generate_detailed_summary(cleaned_text)
            else:  # neutral
                return self._generate_neutral_summary(cleaned_text)
                
        except Exception as e:
            return f"Summarization failed: {str(e)}"
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', '', text)
        
        return text.strip()
    
    def _generate_neutral_summary(self, text: str) -> str:
        """Generate a neutral summary"""
        if self.hf_summarizer:
            try:
                # Split text if too long
                max_chunk_length = 1000
                if len(text.split()) > max_chunk_length:
                    chunks = self._split_text_into_chunks(text, max_chunk_length)
                    summaries = []
                    
                    for chunk in chunks:
                        if len(chunk.split()) > 50:  # Only summarize substantial chunks
                            result = self.hf_summarizer(chunk, max_length=100, min_length=30)
                            summaries.append(result[0]['summary_text'])
                    
                    return ' '.join(summaries)
                else:
                    result = self.hf_summarizer(text, max_length=150, min_length=50)
                    return result[0]['summary_text']
            except Exception as e:
                print(f"HF summarization failed: {e}")
        
        # Fallback to extractive summarization
        return self._extractive_summary(text)
    
    def _extract_facts_summary(self, text: str) -> str:
        """Extract key facts and present as bullet points"""
        sentences = sent_tokenize(text)
        
        # Identify fact-bearing sentences
        fact_indicators = [
            r'\d+', r'percent', r'%', r'\$[\d,]+', r'according to',
            r'reported', r'study', r'research', r'data', r'statistics'
        ]
        
        fact_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            fact_score = sum(1 for pattern in fact_indicators 
                           if re.search(pattern, sentence_lower))
            
            if fact_score >= 2:  # Sentence contains multiple fact indicators
                fact_sentences.append(sentence.strip())
        
        # Select top facts
        if len(fact_sentences) > 6:
            fact_sentences = fact_sentences[:6]
        elif len(fact_sentences) == 0:
            # Fallback: select sentences with numbers or specific data
            fact_sentences = [s for s in sentences if re.search(r'\d+', s)][:4]
        
        # Format as bullet points
        if fact_sentences:
            return '\n'.join([f"• {fact}" for fact in fact_sentences])
        else:
            return self._extractive_summary(text, max_sentences=4)
    
    def _generate_simple_summary(self, text: str) -> str:
        """Generate a summary suitable for a 10-year-old"""
        # Use OpenAI for more sophisticated language simplification
        if os.getenv('OPENAI_API_KEY'):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at explaining complex topics to children. Rewrite the following news article in simple language that a 10-year-old would understand. Use short sentences, simple words, and explain any complex concepts."
                        },
                        {
                            "role": "user",
                            "content": f"Please summarize this article for a 10-year-old:\n\n{text[:2000]}"
                        }
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI simplification failed: {e}")
        
        # Fallback: simple extractive summary with word replacement
        summary = self._extractive_summary(text, max_sentences=3)
        
        # Replace complex words with simpler ones
        simple_replacements = {
            'demonstrate': 'show',
            'approximately': 'about',
            'significant': 'big',
            'investigate': 'look into',
            'contribute': 'help',
            'implement': 'put in place',
            'substantial': 'large',
            'additional': 'more'
        }
        
        for complex_word, simple_word in simple_replacements.items():
            summary = re.sub(rf'\b{complex_word}\b', simple_word, summary, flags=re.IGNORECASE)
        
        return summary
    
    def _generate_detailed_summary(self, text: str) -> str:
        """Generate a comprehensive detailed summary"""
        if os.getenv('OPENAI_API_KEY'):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert journalist and analyst. Provide a comprehensive summary that includes the main points, context, key stakeholders, potential implications, and different perspectives mentioned in the article."
                        },
                        {
                            "role": "user",
                            "content": f"Please provide a detailed analysis and summary of this article:\n\n{text[:3000]}"
                        }
                    ],
                    max_tokens=400,
                    temperature=0.2
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI detailed summary failed: {e}")
        
        # Fallback: extended extractive summary
        return self._extractive_summary(text, max_sentences=8)
    
    def _extractive_summary(self, text: str, max_sentences: int = 5) -> str:
        """Create extractive summary using sentence ranking"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Tokenize and remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Calculate word frequencies
        word_freq = {}
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word.isalpha() and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            score = 0
            word_count = 0
            
            for word in words:
                if word.isalpha() and word not in stop_words:
                    score += word_freq.get(word, 0)
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[i] = score / word_count
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, score in top_sentences[:max_sentences]])
        
        return ' '.join([sentences[i] for i in selected_indices])
    
    def _split_text_into_chunks(self, text: str, max_words: int) -> List[str]:
        """Split text into chunks of approximately max_words"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i + max_words]
            chunks.append(' '.join(chunk_words))
        
        return chunks


# python/config.py
"""
Configuration settings for News Analyzer Pro
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///news_analyzer.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Redis (for caching and rate limiting)
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 60))
    RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_HOUR', 1000))
    
    # Web scraping
    USER_AGENT = os.getenv('USER_AGENT', 'NewsAnalyzerPro/1.0')
    REQUEST_TIMEOUT = 30
    
    # CORS
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    
    # Cache settings
    CACHE_EXPIRATION = timedelta(hours=1)
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
