import pandas as pd
import re
from typing import List, Dict
from collections import defaultdict

class VariationAbuseDetector:
    """Detects all forms of product variation abuse including unrelated variations, price manipulation, and SEO gaming."""
    
    def __init__(self):
        self.max_reasonable_variations = 100
        self.price_variance_threshold = 10
        self.stock_out_threshold = 0.8
        
    def detect(self, listing: Dict) -> Dict:
        results = {
            'score': 0,
            'evidence': [],
            'abuse_types': [],
            'features': []
        }
        
        variations = listing.get('variations', [])
        
        if not variations:
            results['features'] = [0, 0, 0, 0, 0]
            return results
        
        abuses = []
        
        unrelated_results = self._detect_unrelated_variations(variations, listing)
        if unrelated_results:
            abuses.extend(unrelated_results)
        
        price_abuse = self._detect_price_manipulation(variations)
        if price_abuse:
            abuses.extend(price_abuse)
        
        fake_variations = self._detect_fake_variations(variations)
        if fake_variations:
            abuses.extend(fake_variations)
        
        review_pooling = self._detect_review_pooling_abuse(variations)
        if review_pooling:
            abuses.extend(review_pooling)
        
        if len(variations) > self.max_reasonable_variations:
            abuses.append({
                'type': 'excessive_variations',
                'severity': 'medium',
                'count': len(variations),
                'threshold': self.max_reasonable_variations
            })
        
        seo_abuse = self._detect_seo_variation_abuse(variations)
        if seo_abuse:
            abuses.extend(seo_abuse)
        
        results['abuse_types'] = abuses
        results['evidence'] = sorted(abuses, 
                                   key=lambda x: x.get('severity_score', 0), 
                                   reverse=True)[:5]
        results['score'] = min(len(abuses) * 0.2, 1.0)
        
        results['features'] = [
            len(unrelated_results) / len(variations) if variations else 0,
            1 if price_abuse else 0,
            len(fake_variations) / len(variations) if variations else 0,
            1 if review_pooling else 0,
            len(variations) / 100.0
        ]
        
        return results
    
    def _detect_unrelated_variations(self, variations: List[Dict], listing: Dict) -> List[Dict]:
        abuses = []
        
        if not variations:
            return abuses
        
        base_category = listing.get('category', '').lower()
        base_title = listing.get('title', '').lower()
        
        variation_types = defaultdict(list)
        
        for var in variations:
            var_name = var.get('name', '').lower()
            var_type = self._extract_product_type(var_name)
            variation_types[var_type].append(var)
        
        if len(variation_types) > 3:
            base_type = self._extract_product_type(base_title)
            
            for var_type, vars_list in variation_types.items():
                if var_type != base_type and not self._are_types_related(base_type, var_type):
                    abuses.append({
                        'type': 'unrelated_variation',
                        'severity': 'high',
                        'severity_score': 0.9,
                        'base_product': base_type,
                        'variation_type': var_type,
                        'count': len(vars_list),
                        'examples': [v['name'] for v in vars_list[:3]]
                    })
        
        return abuses
    
    def _extract_product_type(self, name: str) -> str:
        product_types = {
            'phone': ['phone', 'smartphone', 'mobile', 'iphone', 'android'],
            'case': ['case', 'cover', 'protector', 'skin'],
            'cable': ['cable', 'cord', 'wire', 'usb', 'hdmi'],
            'charger': ['charger', 'adapter', 'power'],
            'headphone': ['headphone', 'earphone', 'earbud', 'airpod'],
            'screen': ['screen', 'protector', 'glass', 'film'],
            'holder': ['holder', 'mount', 'stand', 'grip'],
            'keyboard': ['keyboard', 'keys'],
            'mouse': ['mouse', 'trackpad'],
            'clothing': ['shirt', 'pants', 'dress', 'jacket', 'coat'],
            'shoe': ['shoe', 'sneaker', 'boot', 'sandal'],
            'bag': ['bag', 'backpack', 'purse', 'wallet'],
            'watch': ['watch', 'band', 'strap'],
            'jewelry': ['ring', 'necklace', 'bracelet', 'earring']
        }
        
        name_lower = name.lower()
        
        for type_name, keywords in product_types.items():
            if any(keyword in name_lower for keyword in keywords):
                return type_name
        
        return 'other'
    
    def _are_types_related(self, type1: str, type2: str) -> bool:
        related_groups = [
            {'phone', 'case', 'screen', 'cable', 'charger'},
            {'laptop', 'case', 'charger', 'mouse', 'keyboard'},
            {'watch', 'band', 'strap'},
            {'shoe', 'lace', 'insole'},
            {'clothing', 'belt', 'scarf'}
        ]
        
        for group in related_groups:
            if type1 in group and type2 in group:
                return True
        
        return False
    
    def _detect_price_manipulation(self, variations: List[Dict]) -> List[Dict]:
        abuses = []
        
        prices = [var.get('price', 0) for var in variations if var.get('price', 0) > 0]
        
        if not prices:
            return abuses
        
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        
        if min_price > 0:
            price_ratio = max_price / min_price
            
            if price_ratio > self.price_variance_threshold:
                abuses.append({
                    'type': 'extreme_price_variance',
                    'severity': 'high',
                    'severity_score': 0.8,
                    'min_price': min_price,
                    'max_price': max_price,
                    'ratio': price_ratio,
                    'detail': f'Prices vary by {price_ratio:.1f}x'
                })
        
        if min_price < avg_price * 0.2:
            low_price_variations = [v for v in variations if v.get('price', 0) == min_price]
            
            abuses.append({
                'type': 'price_anchoring',
                'severity': 'medium',
                'severity_score': 0.6,
                'anchor_price': min_price,
                'average_price': avg_price,
                'low_price_count': len(low_price_variations)
            })
        
        return abuses
    
    def _detect_fake_variations(self, variations: List[Dict]) -> List[Dict]:
        abuses = []
        
        in_stock = 0
        out_of_stock = 0
        
        for var in variations:
            if var.get('in_stock', True) or var.get('quantity', 1) > 0:
                in_stock += 1
            else:
                out_of_stock += 1
        
        total = len(variations)
        
        if total > 0 and out_of_stock / total > self.stock_out_threshold:
            abuses.append({
                'type': 'fake_variations_stock',
                'severity': 'high',
                'severity_score': 0.85,
                'total_variations': total,
                'out_of_stock': out_of_stock,
                'percentage': f'{(out_of_stock/total)*100:.0f}%',
                'detail': 'Most variations perpetually out of stock'
            })
        
        placeholder_patterns = [
            'coming soon', 'placeholder', 'tbd', 'n/a',
            'not available', 'temp', 'test'
        ]
        
        placeholder_count = 0
        for var in variations:
            var_name = var.get('name', '').lower()
            if any(pattern in var_name for pattern in placeholder_patterns):
                placeholder_count += 1
        
        if placeholder_count > 2:
            abuses.append({
                'type': 'placeholder_variations',
                'severity': 'medium',
                'severity_score': 0.6,
                'count': placeholder_count,
                'detail': 'Multiple placeholder variations detected'
            })
        
        return abuses
    
    def _detect_review_pooling_abuse(self, variations: List[Dict]) -> List[Dict]:
        abuses = []
        
        product_types = set()
        price_ranges = []
        
        for var in variations:
            var_type = self._extract_product_type(var.get('name', ''))
            product_types.add(var_type)
            
            price = var.get('price', 0)
            if price > 0:
                price_ranges.append(price)
        
        if len(product_types) > 3:
            abuses.append({
                'type': 'review_pooling_abuse',
                'severity': 'high',
                'severity_score': 0.9,
                'product_types': list(product_types),
                'detail': 'Unrelated products sharing review pool'
            })
        
        if price_ranges:
            min_price = min(price_ranges)
            max_price = max(price_ranges)
            
            if min_price > 0 and max_price / min_price > 5:
                abuses.append({
                    'type': 'review_pooling_price_abuse',
                    'severity': 'medium',
                    'severity_score': 0.7,
                    'price_range': f'${min_price:.2f} - ${max_price:.2f}',
                    'detail': 'Products with vastly different values sharing reviews'
                })
        
        return abuses
    
    def _detect_seo_variation_abuse(self, variations: List[Dict]) -> List[Dict]:
        abuses = []
        
        all_variation_text = ' '.join([var.get('name', '') for var in variations])
        
        brands = ['nike', 'adidas', 'apple', 'samsung', 'sony']
        brand_mentions = sum(all_variation_text.lower().count(brand) for brand in brands)
        
        if brand_mentions > len(variations) * 2:
            abuses.append({
                'type': 'variation_brand_stuffing',
                'severity': 'medium',
                'severity_score': 0.6,
                'brand_mentions': brand_mentions,
                'variation_count': len(variations)
            })
        
        variation_names = [var.get('name', '').lower().strip() for var in variations]
        
        duplicates = 0
        for i, name1 in enumerate(variation_names):
            for name2 in variation_names[i+1:]:
                if self._are_names_similar(name1, name2):
                    duplicates += 1
        
        if duplicates > len(variations) * 0.2:
            abuses.append({
                'type': 'duplicate_variations',
                'severity': 'medium',
                'severity_score': 0.5,
                'duplicate_pairs': duplicates,
                'detail': 'Many variations with similar names'
            })
        
        return abuses
    
    def _are_names_similar(self, name1: str, name2: str) -> bool:
        for pattern in ['size', 'color', 'style', 'model']:
            name1 = re.sub(r'\b\w*' + pattern + r'\w*\b', '', name1)
            name2 = re.sub(r'\b\w*' + pattern + r'\w*\b', '', name2)
        
        name1 = re.sub(r'\d+', '', name1)
        name2 = re.sub(r'\d+', '', name2)
        
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1.intersection(words2))
            similarity = overlap / min(len(words1), len(words2))
            return similarity > 0.8
        
        return False
    
    def train(self, training_data: pd.DataFrame):
        pass
    
    def load_model(self):
        pass
