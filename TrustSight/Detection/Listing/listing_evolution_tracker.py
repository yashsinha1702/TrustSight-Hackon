import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

class ListingEvolutionTracker:
    """Tracks listing changes over time to detect suspicious fraud patterns and evolution anomalies."""
    
    def __init__(self):
        self.suspicious_patterns = {
            'brand_injection': {'score': 0.9, 'threshold': 1},
            'category_hopping': {'score': 0.8, 'threshold': 2},
            'product_transformation': {'score': 0.85, 'threshold': 1},
            'price_manipulation': {'score': 0.7, 'threshold': 0.7},
            'seller_change': {'score': 0.95, 'threshold': 1},
            'location_change': {'score': 0.8, 'threshold': 1}
        }
        
        self.premium_brands = [
            'nike', 'adidas', 'apple', 'samsung', 'sony', 'bose', 
            'louis vuitton', 'gucci', 'prada', 'rolex', 'omega',
            'north face', 'patagonia', 'yeti', 'dyson', 'vitamix'
        ]
        
    def analyze(self, current_listing: Dict, historical_data: List[Dict]) -> Dict:
        results = {
            'score': 0,
            'evidence': [],
            'suspicious_changes': [],
            'risk_timeline': [],
            'features': []
        }
        
        if not historical_data:
            results['features'] = [0, 0, 0, 0, 0]
            return results
        
        timeline = sorted(historical_data + [current_listing], 
                         key=lambda x: x.get('timestamp', datetime.now()))
        
        suspicious_changes = []
        change_counts = defaultdict(int)
        
        for i in range(1, len(timeline)):
            prev = timeline[i-1]
            curr = timeline[i]
            
            changes = self._detect_changes(prev, curr)
            
            for change in changes:
                suspicious_changes.append(change)
                change_counts[change['type']] += 1
                
                results['risk_timeline'].append({
                    'timestamp': curr.get('timestamp'),
                    'change_type': change['type'],
                    'risk_score': change['risk_score']
                })
        
        max_risk = 0
        for pattern_type, config in self.suspicious_patterns.items():
            if change_counts[pattern_type] >= config['threshold']:
                max_risk = max(max_risk, config['score'])
        
        results['score'] = max_risk
        results['suspicious_changes'] = suspicious_changes
        results['evidence'] = sorted(suspicious_changes, 
                                   key=lambda x: x['risk_score'], 
                                   reverse=True)[:5]
        
        results['features'] = [
            change_counts['brand_injection'] / len(timeline),
            change_counts['category_hopping'] / len(timeline),
            change_counts['product_transformation'] / len(timeline),
            change_counts['price_manipulation'] / len(timeline),
            change_counts['seller_change'] / len(timeline)
        ]
        
        return results
    
    def _detect_changes(self, prev: Dict, curr: Dict) -> List[Dict]:
        changes = []
        
        brand_change = self._check_brand_injection(prev, curr)
        if brand_change:
            changes.append(brand_change)
        
        category_change = self._check_category_hopping(prev, curr)
        if category_change:
            changes.append(category_change)
        
        transformation = self._check_product_transformation(prev, curr)
        if transformation:
            changes.append(transformation)
        
        price_change = self._check_price_manipulation(prev, curr)
        if price_change:
            changes.append(price_change)
        
        seller_change = self._check_seller_change(prev, curr)
        if seller_change:
            changes.append(seller_change)
        
        location_change = self._check_location_change(prev, curr)
        if location_change:
            changes.append(location_change)
        
        return changes
    
    def _check_brand_injection(self, prev: Dict, curr: Dict) -> Optional[Dict]:
        prev_brand = prev.get('brand', '').lower().strip()
        curr_brand = curr.get('brand', '').lower().strip()
        
        if (not prev_brand or prev_brand in ['generic', 'unbranded', 'no brand']) and curr_brand:
            if curr_brand in self.premium_brands:
                return {
                    'type': 'brand_injection',
                    'timestamp': curr.get('timestamp'),
                    'change': f'No brand → {curr_brand}',
                    'risk_score': 0.95,
                    'severity': 'critical'
                }
        
        if prev_brand and curr_brand and prev_brand != curr_brand:
            if curr_brand in self.premium_brands and prev_brand not in self.premium_brands:
                return {
                    'type': 'brand_injection',
                    'timestamp': curr.get('timestamp'),
                    'change': f'{prev_brand} → {curr_brand}',
                    'risk_score': 0.9,
                    'severity': 'high'
                }
        
        return None
    
    def _check_category_hopping(self, prev: Dict, curr: Dict) -> Optional[Dict]:
        prev_cat = prev.get('category', '').lower()
        curr_cat = curr.get('category', '').lower()
        
        if prev_cat != curr_cat and prev_cat and curr_cat:
            competitive_categories = [
                'electronics', 'cell phones', 'computers', 'tablets',
                'cameras', 'video games', 'toys', 'sports'
            ]
            
            less_competitive = [
                'books', 'office products', 'industrial', 'scientific'
            ]
            
            if any(cat in prev_cat for cat in competitive_categories) and \
               any(cat in curr_cat for cat in less_competitive):
                return {
                    'type': 'category_hopping',
                    'timestamp': curr.get('timestamp'),
                    'change': f'{prev_cat} → {curr_cat}',
                    'risk_score': 0.8,
                    'severity': 'high',
                    'detail': 'Moving to less competitive category'
                }
        
        return None
    
    def _check_product_transformation(self, prev: Dict, curr: Dict) -> Optional[Dict]:
        prev_title = prev.get('title', '').lower()
        curr_title = curr.get('title', '').lower()
        
        if not prev_title or not curr_title:
            return None
        
        prev_words = set(prev_title.split())
        curr_words = set(curr_title.split())
        
        if len(prev_words) > 0:
            overlap = len(prev_words.intersection(curr_words)) / len(prev_words)
            
            if overlap < 0.3:
                transformations = [
                    ('case', 'phone'),
                    ('cover', 'tablet'),
                    ('accessory', 'device'),
                    ('protector', 'screen'),
                    ('cable', 'adapter'),
                    ('generic', 'original')
                ]
                
                for from_word, to_word in transformations:
                    if from_word in prev_title and to_word in curr_title:
                        return {
                            'type': 'product_transformation',
                            'timestamp': curr.get('timestamp'),
                            'change': f'{from_word} → {to_word}',
                            'risk_score': 0.9,
                            'severity': 'critical',
                            'detail': 'Product type changed'
                        }
                
                return {
                    'type': 'product_transformation',
                    'timestamp': curr.get('timestamp'),
                    'change': 'Major title change',
                    'risk_score': 0.7,
                    'severity': 'medium',
                    'overlap': overlap
                }
        
        return None
    
    def _check_price_manipulation(self, prev: Dict, curr: Dict) -> Optional[Dict]:
        prev_price = prev.get('price', 0)
        curr_price = curr.get('price', 0)
        
        if prev_price > 0 and curr_price > 0:
            price_change_pct = abs(curr_price - prev_price) / prev_price
            
            if curr_price < prev_price and price_change_pct > 0.7:
                return {
                    'type': 'price_manipulation',
                    'timestamp': curr.get('timestamp'),
                    'change': f'${prev_price:.2f} → ${curr_price:.2f}',
                    'risk_score': 0.8,
                    'severity': 'high',
                    'detail': f'{price_change_pct*100:.0f}% price drop'
                }
            
            if curr_price > prev_price and price_change_pct > 3.0:
                return {
                    'type': 'price_manipulation',
                    'timestamp': curr.get('timestamp'),
                    'change': f'${prev_price:.2f} → ${curr_price:.2f}',
                    'risk_score': 0.7,
                    'severity': 'medium',
                    'detail': f'{price_change_pct*100:.0f}% price increase'
                }
        
        return None
    
    def _check_seller_change(self, prev: Dict, curr: Dict) -> Optional[Dict]:
        prev_seller = prev.get('seller_id', '')
        curr_seller = curr.get('seller_id', '')
        
        if prev_seller and curr_seller and prev_seller != curr_seller:
            return {
                'type': 'seller_change',
                'timestamp': curr.get('timestamp'),
                'change': f'Seller changed',
                'risk_score': 0.95,
                'severity': 'critical',
                'detail': f'{prev_seller} → {curr_seller}'
            }
        
        return None
    
    def _check_location_change(self, prev: Dict, curr: Dict) -> Optional[Dict]:
        prev_location = prev.get('ship_from', '').lower()
        curr_location = curr.get('ship_from', '').lower()
        
        if prev_location != curr_location and prev_location and curr_location:
            high_risk_changes = [
                ('united states', 'china'),
                ('usa', 'china'),
                ('us', 'cn'),
                ('united kingdom', 'china'),
                ('fulfilled by amazon', 'third party')
            ]
            
            for from_loc, to_loc in high_risk_changes:
                if from_loc in prev_location and to_loc in curr_location:
                    return {
                        'type': 'location_change',
                        'timestamp': curr.get('timestamp'),
                        'change': f'{prev_location} → {curr_location}',
                        'risk_score': 0.85,
                        'severity': 'high',
                        'detail': 'Suspicious location change'
                    }
        
        return None
    
    def train(self, training_data: pd.DataFrame):
        pass
    
    def load_model(self):
        pass
