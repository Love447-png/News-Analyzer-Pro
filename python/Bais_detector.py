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
