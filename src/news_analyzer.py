#!/usr/bin/env python3
"""News sentiment analysis module for Kalshi trading bot."""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from textblob import TextBlob
import re
from config import NEWS_API_KEY, NEWS_API_BASE_URL

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """Analyzes news sentiment for trading signals."""

    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.base_url = NEWS_API_BASE_URL
        self.session = requests.Session()

        # News API rate limit protection
        self._cached_articles = []
        self._cache_timestamp = None
        self._cache_ttl_seconds = 1800  # Reuse cached results for 30 minutes
        self._consecutive_429s = 0

        # Keywords focused on crypto markets
        self.keywords = [
            'bitcoin', 'BTC', 'ethereum', 'ETH', 'solana', 'SOL',
            'crypto', 'cryptocurrency', 'blockchain',
            'bitcoin price', 'crypto market', 'bitcoin ETF', 'ethereum ETF',
            'crypto regulation', 'SEC crypto', 'bitcoin reserve',
            'stablecoin', 'DeFi', 'crypto exchange',
            'federal reserve', 'interest rate', 'inflation',
        ]

    def _cache_is_fresh(self) -> bool:
        """Check if cached articles are still usable."""
        if not self._cache_timestamp or not self._cached_articles:
            return False
        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl_seconds

    def fetch_news(self, query: str = None, days_back: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch news articles from NewsAPI with rate-limit protection.

        Uses cached results when rate-limited (429) instead of returning empty.
        """
        if not self.api_key or self.api_key == "your_news_api_key":
            logger.warning("NewsAPI key not configured, skipping news fetch")
            return []

        # If we've been rate-limited recently, use cache instead of hammering the API
        if self._consecutive_429s >= 2 and self._cache_is_fresh():
            logger.info(f"Using cached news ({len(self._cached_articles)} articles, "
                       f"avoiding rate limit after {self._consecutive_429s} consecutive 429s)")
            return self._cached_articles

        try:
            if not query:
                query = ' OR '.join(f'"{kw}"' for kw in self.keywords[:5])

            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.api_key
            }

            response = self.session.get(f"{self.base_url}/everything", params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            articles = data.get('articles', [])

            # Cache successful results
            self._cached_articles = articles
            self._cache_timestamp = datetime.now()
            self._consecutive_429s = 0

            logger.info(f"Fetched {len(articles)} news articles")
            return articles

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                self._consecutive_429s += 1
                if self._cached_articles:
                    logger.warning(f"NewsAPI rate limited (429), using cached articles "
                                  f"({len(self._cached_articles)} articles)")
                    return self._cached_articles
                else:
                    logger.warning("NewsAPI rate limited (429) and no cached articles available")
                    return []
            logger.error(f"Error fetching news: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            # Fall back to cache on any error
            if self._cached_articles:
                return self._cached_articles
            return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using TextBlob.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with polarity and subjectivity scores
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1 (negative to positive)
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1 (objective to subjective)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5}

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better sentiment analysis.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    def analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment across multiple news articles.

        Args:
            articles: List of news articles

        Returns:
            Aggregated sentiment analysis
        """
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'positive_articles': 0,
                'negative_articles': 0,
                'neutral_articles': 0
            }

        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')

            # Combine title and description for analysis
            text = f"{title} {description}".strip()
            clean_text = self.preprocess_text(text)

            if clean_text:
                sentiment = self.analyze_sentiment(clean_text)
                sentiments.append(sentiment)

                # Classify sentiment
                polarity = sentiment['polarity']
                if polarity > 0.1:
                    positive_count += 1
                elif polarity < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1

        if not sentiments:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'positive_articles': 0,
                'negative_articles': 0,
                'neutral_articles': 0
            }

        # Calculate aggregate metrics
        avg_polarity = sum(s['polarity'] for s in sentiments) / len(sentiments)
        avg_subjectivity = sum(s['subjectivity'] for s in sentiments) / len(sentiments)

        # Confidence based on agreement and article count
        polarity_variance = sum((s['polarity'] - avg_polarity) ** 2 for s in sentiments) / len(sentiments)
        agreement_factor = 1 / (1 + polarity_variance)  # Higher agreement = higher confidence
        volume_factor = min(len(sentiments) / 10, 1.0)  # More articles = higher confidence
        confidence = agreement_factor * volume_factor

        return {
            'overall_sentiment': round(avg_polarity, 3),
            'avg_subjectivity': round(avg_subjectivity, 3),
            'confidence': round(confidence, 3),
            'article_count': len(sentiments),
            'positive_articles': positive_count,
            'negative_articles': negative_count,
            'neutral_articles': neutral_count,
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
        }

    def get_market_relevant_news(self, market_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Get news specifically relevant to Kalshi markets.

        Args:
            market_keywords: Market-specific keywords

        Returns:
            News sentiment analysis for market relevance
        """
        if market_keywords:
            # Search for market-specific news
            query = ' OR '.join(f'"{kw}"' for kw in market_keywords)
            articles = self.fetch_news(query=query, days_back=2)
        else:
            # General news search
            articles = self.fetch_news(days_back=1)

        sentiment_analysis = self.analyze_news_sentiment(articles)

        # Add timestamp and source info
        sentiment_analysis.update({
            'timestamp': datetime.now().isoformat(),
            'source': 'NewsAPI',
            'query_used': market_keywords or self.keywords[:5]
        })

        logger.info(f"Market news sentiment: {sentiment_analysis['overall_sentiment']:.3f} "
                   f"(confidence: {sentiment_analysis['confidence']:.3f})")

        return sentiment_analysis

    def should_trade_based_on_sentiment(self, sentiment_analysis: Dict[str, Any],
                                       threshold: float = 0.6) -> Dict[str, Any]:
        """
        Determine if sentiment warrants a trading decision.

        Args:
            sentiment_analysis: Result from analyze_news_sentiment
            threshold: Minimum confidence and sentiment threshold

        Returns:
            Trading decision with reasoning
        """
        sentiment = sentiment_analysis.get('overall_sentiment', 0)
        confidence = sentiment_analysis.get('confidence', 0)

        decision = {
            'should_trade': False,
            'direction': None,  # 'long' or 'short'
            'confidence': confidence,
            'sentiment_score': sentiment,
            'reason': ''
        }

        if confidence < 0.15:  # Minimum confidence threshold (lowered to generate data)
            decision['reason'] = f"Low confidence ({confidence:.2f}) - insufficient data"
            return decision

        if sentiment > threshold and confidence > 0.2:
            decision['should_trade'] = True
            decision['direction'] = 'long'
            decision['reason'] = f"Positive sentiment ({sentiment:.2f}) with confidence ({confidence:.2f})"
        elif sentiment < -threshold and confidence > 0.2:
            decision['should_trade'] = True
            decision['direction'] = 'short'
            decision['reason'] = f"Negative sentiment ({sentiment:.2f}) with confidence ({confidence:.2f})"
        else:
            decision['reason'] = f"Sentiment ({sentiment:.2f}) below threshold or insufficient confidence"

        return decision
