"""
Earnings sentiment analysis for Break-even Lab
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EarningsSentimentAnalyzer:
    """Earnings call sentiment analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Financial sentiment words (simplified Loughran-McDonald approach)
        self.positive_words = {
            'strong', 'growth', 'increase', 'improve', 'positive', 'better', 'excellent',
            'outstanding', 'robust', 'solid', 'momentum', 'expansion', 'opportunity',
            'success', 'profit', 'gain', 'revenue', 'earnings', 'beat', 'exceed',
            'outperform', 'optimistic', 'confident', 'bullish', 'upside', 'potential'
        }
        
        self.negative_words = {
            'weak', 'decline', 'decrease', 'worse', 'negative', 'poor', 'challenge',
            'difficult', 'struggle', 'concern', 'risk', 'uncertainty', 'volatility',
            'pressure', 'headwind', 'loss', 'miss', 'disappoint', 'underperform',
            'pessimistic', 'bearish', 'downside', 'threat', 'problem', 'issue'
        }
        
        # Financial metrics keywords
        self.metrics_keywords = {
            'revenue', 'sales', 'earnings', 'profit', 'margin', 'growth', 'guidance',
            'outlook', 'forecast', 'expectation', 'target', 'goal', 'objective'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text"""
        sentences = sent_tokenize(text)
        phrases = []
        
        for sentence in sentences:
            # Look for sentences with financial keywords
            if any(keyword in sentence.lower() for keyword in self.metrics_keywords):
                # Clean and add the sentence
                clean_sentence = self.preprocess_text(sentence)
                if len(clean_sentence) > 20:  # Filter out very short sentences
                    phrases.append(clean_sentence)
        
        # Return top phrases
        return phrases[:max_phrases]
    
    def calculate_sentiment_score(self, text: str) -> Dict[str, float]:
        """Calculate sentiment score using multiple methods"""
        # Method 1: TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Method 2: Custom financial sentiment
        words = word_tokenize(self.preprocess_text(text))
        words = [word for word in words if word not in self.stop_words]
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            custom_sentiment = (positive_count - negative_count) / total_sentiment_words
        else:
            custom_sentiment = 0
        
        # Method 3: Weighted average
        weighted_sentiment = (textblob_polarity * 0.6 + custom_sentiment * 0.4)
        
        return {
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'custom_sentiment': custom_sentiment,
            'weighted_sentiment': weighted_sentiment,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def analyze_earnings_transcript(self, transcript: str) -> Dict:
        """Analyze earnings call transcript"""
        if not transcript or len(transcript.strip()) < 100:
            return {
                'error': 'Transcript too short or empty',
                'sentiment_score': 0,
                'key_phrases': [],
                'summary': 'No meaningful content to analyze'
            }
        
        # Calculate overall sentiment
        sentiment_scores = self.calculate_sentiment_score(transcript)
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(transcript)
        
        # Generate summary
        summary = self.generate_summary(transcript, sentiment_scores)
        
        # Determine sentiment label
        sentiment_label = self.get_sentiment_label(sentiment_scores['weighted_sentiment'])
        
        return {
            'sentiment_score': sentiment_scores['weighted_sentiment'],
            'sentiment_label': sentiment_label,
            'textblob_polarity': sentiment_scores['textblob_polarity'],
            'custom_sentiment': sentiment_scores['custom_sentiment'],
            'positive_words': sentiment_scores['positive_words'],
            'negative_words': sentiment_scores['negative_words'],
            'key_phrases': key_phrases,
            'summary': summary,
            'word_count': len(transcript.split()),
            'analysis_confidence': min(abs(sentiment_scores['weighted_sentiment']) * 100, 100)
        }
    
    def generate_summary(self, text: str, sentiment_scores: Dict) -> str:
        """Generate a summary of the earnings call"""
        sentiment = sentiment_scores['weighted_sentiment']
        
        if sentiment > 0.1:
            sentiment_desc = "positive"
        elif sentiment < -0.1:
            sentiment_desc = "negative"
        else:
            sentiment_desc = "neutral"
        
        # Count key metrics mentioned
        metrics_mentioned = sum(1 for keyword in self.metrics_keywords if keyword in text.lower())
        
        summary_parts = [
            f"The earnings call shows a {sentiment_desc} sentiment overall.",
            f"Key financial metrics were mentioned {metrics_mentioned} times.",
            f"Sentiment analysis indicates {sentiment_scores['positive_words']} positive and {sentiment_scores['negative_words']} negative sentiment words."
        ]
        
        return " ".join(summary_parts)
    
    def get_sentiment_label(self, score: float) -> str:
        """Get sentiment label based on score"""
        if score > 0.2:
            return "Very Bullish"
        elif score > 0.1:
            return "Bullish"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.2:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def compare_quarters(self, transcripts: List[Dict]) -> Dict:
        """Compare sentiment across multiple quarters"""
        if len(transcripts) < 2:
            return {'error': 'Need at least 2 transcripts to compare'}
        
        results = []
        for i, transcript_data in enumerate(transcripts):
            if 'transcript' in transcript_data:
                analysis = self.analyze_earnings_transcript(transcript_data['transcript'])
                results.append({
                    'quarter': transcript_data.get('quarter', f'Q{i+1}'),
                    'sentiment_score': analysis['sentiment_score'],
                    'sentiment_label': analysis['sentiment_label'],
                    'key_phrases': analysis['key_phrases'][:3]  # Top 3 phrases
                })
        
        # Calculate trend
        scores = [r['sentiment_score'] for r in results]
        if len(scores) >= 2:
            trend = "improving" if scores[-1] > scores[0] else "declining"
        else:
            trend = "stable"
        
        return {
            'quarterly_results': results,
            'trend': trend,
            'latest_sentiment': results[-1]['sentiment_label'] if results else 'Unknown',
            'sentiment_change': scores[-1] - scores[0] if len(scores) >= 2 else 0
        }

# Global analyzer instance
earnings_analyzer = EarningsSentimentAnalyzer()

def analyze_earnings_transcript(transcript: str) -> Dict:
    """Analyze earnings call transcript"""
    return earnings_analyzer.analyze_earnings_transcript(transcript)

def compare_earnings_quarters(transcripts: List[Dict]) -> Dict:
    """Compare sentiment across multiple quarters"""
    return earnings_analyzer.compare_quarters(transcripts)
