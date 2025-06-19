import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
from typing import List, Dict, Optional
from collections import defaultdict
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReviewProductMismatchDetector:
    """Detects mismatches between product listings and customer reviews across all Amazon categories."""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        from amazon_category_taxonomy import AmazonCategoryTaxonomy
        self.category_taxonomy = AmazonCategoryTaxonomy()
        self.threshold = 0.3
        self.classifier = None
        self.sentence_model.to(device)

    def detect(self, listing: Dict, reviews: List[Dict]) -> Dict:
        results = {
            'score': 0,
            'evidence': [],
            'mismatch_count': 0,
            'mismatch_types': defaultdict(int),
            'features': []
        }
        
        if not reviews:
            results['features'] = [0, 0, 0, 0, 0]
            return results
        
        listing_text = self._create_listing_text(listing)
        listing_embedding = self.sentence_model.encode(listing_text)
        listing_category, listing_all_cats = self.category_taxonomy.identify_category(listing_text)
        
        mismatches = []
        
        for review in reviews:
            review_text = review.get('text', '')
            if not review_text:
                continue
            
            review_embedding = self.sentence_model.encode(review_text)
            similarity = cosine_similarity([listing_embedding], [review_embedding])[0][0]
            
            review_category, review_all_cats = self.category_taxonomy.identify_category(review_text)
            
            mismatch_info = {
                'review_id': review.get('id'),
                'similarity_score': float(similarity),
                'listing_category': listing_category,
                'review_mentions': review_all_cats,
                'mismatches': []
            }
            
            for rev_cat in review_all_cats:
                if not self.category_taxonomy.are_categories_compatible(listing_category, rev_cat):
                    mismatch_info['mismatches'].append({
                        'type': 'incompatible_category',
                        'severity': 'high',
                        'detail': f'{listing_category} incompatible with {rev_cat}'
                    })
                    results['mismatch_types']['incompatible_category'] += 1
            
            temporal_mismatch = self._check_temporal_mismatch(listing, review_text)
            if temporal_mismatch:
                mismatch_info['mismatches'].append(temporal_mismatch)
                results['mismatch_types']['temporal'] += 1
            
            feature_mismatch = self._check_feature_mismatch(listing, review_text)
            if feature_mismatch:
                mismatch_info['mismatches'].append(feature_mismatch)
                results['mismatch_types']['feature'] += 1
            
            brand_mismatch = self._check_brand_confusion(listing, review_text)
            if brand_mismatch:
                mismatch_info['mismatches'].append(brand_mismatch)
                results['mismatch_types']['brand'] += 1
            
            if mismatch_info['mismatches'] or similarity < self.threshold:
                mismatches.append(mismatch_info)
        
        results['mismatch_count'] = len(mismatches)
        results['score'] = len(mismatches) / len(reviews) if reviews else 0
        results['evidence'] = sorted(mismatches, 
                                   key=lambda x: len(x['mismatches']), 
                                   reverse=True)[:10]
        
        results['features'] = [
            results['score'],
            results['mismatch_types']['incompatible_category'] / len(reviews),
            results['mismatch_types']['temporal'] / len(reviews),
            results['mismatch_types']['feature'] / len(reviews),
            results['mismatch_types']['brand'] / len(reviews)
        ]
        
        return results
    
    def _create_listing_text(self, listing: Dict) -> str:
        parts = [
            listing.get('title', ''),
            listing.get('brand', ''),
            listing.get('category', ''),
            listing.get('subcategory', ''),
            ' '.join(listing.get('bullet_points', [])),
            listing.get('description', '')[:500]
        ]
        return ' '.join(filter(None, parts))
    
    def _check_temporal_mismatch(self, listing: Dict, review_text: str) -> Optional[Dict]:
        review_lower = review_text.lower()
        
        year_pattern = r'\b(19|20)\d{2}\b'
        mentioned_years = [int(y) for y in re.findall(year_pattern, review_text)]
        
        if not mentioned_years:
            return None
        
        launch_date = listing.get('launch_date')
        if launch_date:
            if isinstance(launch_date, str):
                launch_year = int(launch_date[:4])
            else:
                launch_year = launch_date.year
            
            for year in mentioned_years:
                if year < launch_year - 1:
                    return {
                        'type': 'temporal_mismatch',
                        'severity': 'high',
                        'detail': f'Review mentions {year} but product launched {launch_year}'
                    }
        
        return None
    
    def _check_feature_mismatch(self, listing: Dict, review_text: str) -> Optional[Dict]:
        review_lower = review_text.lower()
        
        feature_checks = {
            'wireless': ['wireless', 'bluetooth', 'wifi'],
            'waterproof': ['waterproof', 'water resistant', 'ipx'],
            'rechargeable': ['rechargeable', 'usb charging', 'battery life'],
            'smart': ['smart', 'app', 'alexa', 'google assistant'],
            'organic': ['organic', 'non-gmo', 'natural'],
            'gluten_free': ['gluten free', 'gluten-free', 'celiac']
        }
        
        listing_features = listing.get('features', [])
        listing_text = self._create_listing_text(listing).lower()
        
        for feature_type, keywords in feature_checks.items():
            review_has_feature = any(kw in review_lower for kw in keywords)
            listing_has_feature = any(kw in listing_text for kw in keywords)
            
            if review_has_feature and not listing_has_feature:
                return {
                    'type': 'feature_mismatch',
                    'severity': 'medium',
                    'detail': f'Review mentions {feature_type} but listing does not'
                }
        
        return None
    
    def _check_brand_confusion(self, listing: Dict, review_text: str) -> Optional[Dict]:
        listing_brand = listing.get('brand', '').lower()
        if not listing_brand:
            return None
        
        brand_groups = [
            ['nike', 'adidas', 'puma', 'reebok', 'under armour'],
            ['apple', 'samsung', 'google', 'oneplus', 'xiaomi'],
            ['sony', 'bose', 'jbl', 'beats', 'sennheiser'],
            ['dell', 'hp', 'lenovo', 'asus', 'acer'],
            ['coca cola', 'pepsi', 'sprite', 'fanta'],
            ['pampers', 'huggies', 'luvs', 'honest']
        ]
        
        review_lower = review_text.lower()
        
        for group in brand_groups:
            if listing_brand in group:
                for other_brand in group:
                    if other_brand != listing_brand and other_brand in review_lower:
                        return {
                            'type': 'brand_confusion',
                            'severity': 'high',
                            'detail': f'Listing is {listing_brand} but review mentions {other_brand}'
                        }
        
        return None
    
    def train(self, training_data: pd.DataFrame):
        device = 'cpu'
        self.sentence_model.to(device)
        
        training_data['listing_text'] = (
            training_data['listing_title'].fillna('') + ' ' +
            training_data['listing_category'].fillna('') + ' ' +
            training_data['listing_brand'].fillna('') + ' ' +
            training_data['listing_description'].fillna('')
        )
        
        batch_size = 100
        X = []
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data.iloc[i:i+batch_size]
            
            listing_texts = batch['listing_text'].tolist()
            review_texts = batch['review_text'].tolist()
            
            listing_embeddings = self.sentence_model.encode(
                listing_texts, device=device
            )
            
            review_embeddings = self.sentence_model.encode(
                review_texts, device=device
            )
            
            for j in range(len(batch)):
                similarity = cosine_similarity([listing_embeddings[j]], [review_embeddings[j]])[0][0]
                
                listing_cat, _ = self.category_taxonomy.identify_category(listing_texts[j])
                review_cat, _ = self.category_taxonomy.identify_category(review_texts[j])
                cat_compatible = 1 if self.category_taxonomy.are_categories_compatible(listing_cat, review_cat) else 0
                
                X.append([similarity, cat_compatible])
        
        X = np.array(X)
        y = training_data['is_mismatch'].values
        
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X, y)
        
        joblib.dump(self.classifier, 'mismatch_detector_model.pkl')
    
    def load_model(self):
        try:
            self.classifier = joblib.load('mismatch_detector_model.pkl')
        except:
            pass
