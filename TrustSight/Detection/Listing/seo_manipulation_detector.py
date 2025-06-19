import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from typing import List, Dict
from collections import defaultdict, Counter

class SEOManipulationDetector:
    """Detects all forms of SEO manipulation, keyword abuse, and deceptive optimization tactics."""
    
    def __init__(self):
        self.keyword_threshold = 5
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.classifier = None
        
        self.spam_patterns = [
            (r'\b(\w+)\s+\1\s+\1\b', 'triple_repetition'),
            (r'(.{10,})\1{2,}', 'phrase_repetition'),
            (r'(?i)(best|cheap|buy|discount|sale|free shipping){5,}', 'promotional_stuffing'),
            (r'(?i)(nike|adidas|apple|samsung|sony){4,}', 'brand_stuffing'),
            (r'\s{5,}', 'excessive_spaces'),
            (r'[\u200b\u200c\u200d\ufeff]', 'invisible_unicode'),
            (r'[A-Z\s]{30,}', 'excessive_caps'),
            (r'[★☆✓✗✔✘]{5,}', 'special_char_spam'),
            (r'(?i)(www\.|http:|\.com|\.net){3,}', 'url_spam')
        ]
        
        self.manipulation_keywords = {
            'competitor_hijacking': [
                'better than', 'replacement for', 'alternative to',
                'vs', 'versus', 'compared to'
            ],
            'fake_urgency': [
                'limited time', 'hurry', 'last chance', 'ending soon',
                'today only', 'flash sale'
            ],
            'trust_manipulation': [
                'authentic', 'genuine', 'original', 'official',
                'authorized dealer', 'warranty'
            ],
            'review_manipulation': [
                '5 star', 'five star', 'top rated', 'best seller',
                '#1', 'number one'
            ]
        }
    
    def detect(self, listing: Dict) -> Dict:
        results = {
            'score': 0,
            'evidence': [],
            'manipulation_types': [],
            'features': []
        }
        
        text = self._extract_all_text(listing)
        
        manipulations = []
        
        stuffing_results = self._detect_keyword_stuffing(text)
        if stuffing_results:
            manipulations.extend(stuffing_results)
        
        spam_results = self._detect_spam_patterns(text)
        if spam_results:
            manipulations.extend(spam_results)
        
        hidden_results = self._detect_hidden_text(listing)
        if hidden_results:
            manipulations.extend(hidden_results)
        
        category_abuse = self._detect_category_abuse(text, listing)
        if category_abuse:
            manipulations.extend(category_abuse)
        
        keyword_manipulation = self._detect_manipulation_keywords(text)
        if keyword_manipulation:
            manipulations.extend(keyword_manipulation)
        
        title_manipulation = self._detect_title_manipulation(listing.get('title', ''))
        if title_manipulation:
            manipulations.extend(title_manipulation)
        
        results['manipulation_types'] = manipulations
        results['evidence'] = manipulations[:10]
        results['score'] = min(len(manipulations) * 0.15, 1.0)
        
        results['features'] = [
            len(stuffing_results) / 10.0,
            len(spam_results) / 5.0,
            len(hidden_results) / 3.0,
            len(category_abuse) / 5.0,
            len(keyword_manipulation) / 10.0
        ]
        
        return results
    
    def _extract_all_text(self, listing: Dict) -> str:
        parts = [
            listing.get('title', ''),
            listing.get('description', ''),
            ' '.join(listing.get('bullet_points', [])),
            ' '.join(listing.get('search_terms', [])),
            listing.get('backend_keywords', '')
        ]
        return ' '.join(filter(None, parts))
    
    def _detect_keyword_stuffing(self, text: str) -> List[Dict]:
        manipulations = []
        words = text.lower().split()
        word_counts = Counter(words)
        
        for word, count in word_counts.items():
            if len(word) > 3 and count > self.keyword_threshold:
                density = count / len(words) if words else 0
                
                if density > 0.05:
                    manipulations.append({
                        'type': 'keyword_stuffing',
                        'severity': 'high' if density > 0.1 else 'medium',
                        'keyword': word,
                        'count': count,
                        'density': f'{density*100:.1f}%'
                    })
        
        brands = ['nike', 'adidas', 'apple', 'samsung', 'sony', 'amazon']
        brand_mentions = sum(word_counts.get(brand, 0) for brand in brands)
        
        if brand_mentions > 10:
            manipulations.append({
                'type': 'brand_stuffing',
                'severity': 'high',
                'count': brand_mentions,
                'detail': 'Excessive brand name mentions'
            })
        
        return manipulations
    
    def _detect_spam_patterns(self, text: str) -> List[Dict]:
        manipulations = []
        
        for pattern, pattern_type in self.spam_patterns:
            matches = re.findall(pattern, text)
            if matches:
                manipulations.append({
                    'type': f'spam_pattern_{pattern_type}',
                    'severity': 'medium',
                    'matches': len(matches),
                    'example': str(matches[0])[:50] if matches else ''
                })
        
        return manipulations
    
    def _detect_hidden_text(self, listing: Dict) -> List[Dict]:
        manipulations = []
        
        description = listing.get('description', '')
        
        if '     ' in description:
            manipulations.append({
                'type': 'hidden_text_whitespace',
                'severity': 'medium',
                'detail': 'Excessive whitespace detected'
            })
        
        invisible_chars = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2060']
        for char in invisible_chars:
            if char in description:
                manipulations.append({
                    'type': 'hidden_text_unicode',
                    'severity': 'high',
                    'detail': 'Invisible Unicode characters detected'
                })
                break
        
        backend = listing.get('backend_keywords', '')
        if backend and len(backend) > 1000:
            manipulations.append({
                'type': 'backend_keyword_abuse',
                'severity': 'medium',
                'length': len(backend),
                'detail': 'Excessive backend keywords'
            })
        
        return manipulations
    
    def _detect_category_abuse(self, text: str, listing: Dict) -> List[Dict]:
        manipulations = []
        text_lower = text.lower()
        
        categories = [
            'electronics', 'clothing', 'shoes', 'jewelry', 'books',
            'toys', 'games', 'sports', 'outdoors', 'automotive',
            'tools', 'home', 'kitchen', 'garden', 'pet supplies',
            'beauty', 'health', 'grocery', 'baby', 'industrial',
            'handmade', 'music', 'movies', 'software'
        ]
        
        mentioned_categories = [cat for cat in categories if cat in text_lower]
        
        actual_category = listing.get('category', '').lower()
        
        if len(mentioned_categories) > 5:
            manipulations.append({
                'type': 'category_stuffing',
                'severity': 'high',
                'count': len(mentioned_categories),
                'categories': mentioned_categories[:10]
            })
        
        if actual_category:
            unrelated = [cat for cat in mentioned_categories 
                        if cat not in actual_category and actual_category not in cat]
            
            if len(unrelated) > 2:
                manipulations.append({
                    'type': 'unrelated_category_mentions',
                    'severity': 'medium',
                    'actual': actual_category,
                    'unrelated': unrelated
                })
        
        return manipulations
    
    def _detect_manipulation_keywords(self, text: str) -> List[Dict]:
        manipulations = []
        text_lower = text.lower()
        
        for manipulation_type, keywords in self.manipulation_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            
            if len(matches) > 2:
                manipulations.append({
                    'type': f'keyword_manipulation_{manipulation_type}',
                    'severity': 'medium',
                    'keywords_found': matches,
                    'count': len(matches)
                })
        
        return manipulations
    
    def _detect_title_manipulation(self, title: str) -> List[Dict]:
        manipulations = []
        
        if len(title) > 200:
            manipulations.append({
                'type': 'title_too_long',
                'severity': 'medium',
                'length': len(title),
                'detail': 'Title exceeds recommended length'
            })
        
        special_chars = re.findall(r'[|/\-,]', title)
        if len(special_chars) > 5:
            manipulations.append({
                'type': 'title_separator_abuse',
                'severity': 'low',
                'count': len(special_chars),
                'detail': 'Excessive separators in title'
            })
        
        caps_words = re.findall(r'\b[A-Z]{4,}\b', title)
        if len(caps_words) > 3:
            manipulations.append({
                'type': 'title_caps_abuse',
                'severity': 'low',
                'count': len(caps_words),
                'examples': caps_words[:5]
            })
        
        return manipulations
    
    def train(self, training_data: pd.DataFrame):
        training_data['listing_text'] = (
            training_data['title'].fillna('') + ' ' +
            training_data['description'].fillna('') + ' ' +
            training_data['bullet_points'].fillna('') + ' ' +
            training_data['backend_keywords'].fillna('')
        )
        
        texts = training_data['listing_text'].values
        labels = training_data['has_seo_manipulation'].values
        
        X = self.tfidf.fit_transform(texts)
        
        from sklearn.svm import SVC
        self.classifier = SVC(probability=True, random_state=42)
        self.classifier.fit(X, labels)
        
        joblib.dump(self.classifier, 'seo_detector_model.pkl')
        joblib.dump(self.tfidf, 'seo_tfidf.pkl')
    
    def load_model(self):
        try:
            self.classifier = joblib.load('seo_detector_model.pkl')
            self.tfidf = joblib.load('seo_tfidf.pkl')
        except:
            pass
