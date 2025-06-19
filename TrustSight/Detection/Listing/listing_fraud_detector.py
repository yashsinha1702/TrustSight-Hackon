import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Dict, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import torch
torch.set_num_threads(4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ListingFraudDetector:
    """Main orchestrating class that coordinates all fraud detection components and provides comprehensive fraud analysis."""
    
    def __init__(self):
        from review_product_mismatch_detector import ReviewProductMismatchDetector
        from listing_evolution_tracker import ListingEvolutionTracker
        from seo_manipulation_detector import SEOManipulationDetector
        from variation_abuse_detector import VariationAbuseDetector
        from listing_hijacking_detector import ListingHijackingDetector
        from amazon_category_taxonomy import AmazonCategoryTaxonomy
        
        self.mismatch_detector = ReviewProductMismatchDetector()
        self.evolution_tracker = ListingEvolutionTracker()
        self.seo_detector = SEOManipulationDetector()
        self.variation_detector = VariationAbuseDetector()
        self.hijack_detector = ListingHijackingDetector()
        
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
        self.category_taxonomy = AmazonCategoryTaxonomy()
        
        self.thresholds = {
            'mismatch': 0.3,
            'evolution': 0.4,
            'seo': 0.35,
            'variation': 0.4,
            'hijacking': 0.5
        }
        
    def detect_listing_fraud(self, listing: Dict, reviews: List[Dict] = None, 
                           historical_data: List[Dict] = None) -> Dict:
        
        fraud_report = {
            'listing_id': listing.get('id', 'unknown'),
            'asin': listing.get('asin', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'fraud_scores': {},
            'evidence': [],
            'overall_risk': 0,
            'confidence': 0,
            'fraud_types_detected': [],
            'action_recommended': None,
            'estimated_impact': 0,
            'network_indicators': []
        }
        
        features = []
        
        if reviews:
            mismatch_results = self.mismatch_detector.detect(listing, reviews)
            fraud_report['fraud_scores']['mismatch'] = mismatch_results['score']
            fraud_report['evidence'].extend(mismatch_results['evidence'])
            features.extend(mismatch_results['features'])
            
            if mismatch_results['score'] > self.thresholds['mismatch']:
                fraud_report['fraud_types_detected'].append('review_product_mismatch')
        
        if historical_data:
            evolution_results = self.evolution_tracker.analyze(listing, historical_data)
            fraud_report['fraud_scores']['evolution'] = evolution_results['score']
            fraud_report['evidence'].extend(evolution_results['evidence'])
            features.extend(evolution_results['features'])
            
            if evolution_results['score'] > self.thresholds['evolution']:
                fraud_report['fraud_types_detected'].append('suspicious_evolution')
        
        seo_results = self.seo_detector.detect(listing)
        fraud_report['fraud_scores']['seo_manipulation'] = seo_results['score']
        fraud_report['evidence'].extend(seo_results['evidence'])
        features.extend(seo_results['features'])
        
        if seo_results['score'] > self.thresholds['seo']:
            fraud_report['fraud_types_detected'].append('seo_manipulation')
        
        if listing.get('variations'):
            variation_results = self.variation_detector.detect(listing)
            fraud_report['fraud_scores']['variation_abuse'] = variation_results['score']
            fraud_report['evidence'].extend(variation_results['evidence'])
            features.extend(variation_results['features'])
            
            if variation_results['score'] > self.thresholds['variation']:
                fraud_report['fraud_types_detected'].append('variation_abuse')
        
        if historical_data:
            hijack_results = self.hijack_detector.detect(listing, historical_data)
            fraud_report['fraud_scores']['hijacking'] = hijack_results['score']
            fraud_report['evidence'].extend(hijack_results['evidence'])
            features.extend(hijack_results['features'])
            
            if hijack_results['score'] > self.thresholds['hijacking']:
                fraud_report['fraud_types_detected'].append('listing_hijacking')
        
        if self.ensemble_model and features:
            try:
                features_array = np.array(features).reshape(1, -1)
                features_scaled = self.scaler.transform(features_array)
                
                fraud_report['overall_risk'] = float(self.ensemble_model.predict_proba(features_scaled)[0, 1])
                fraud_report['confidence'] = self._calculate_confidence(fraud_report['fraud_scores'])
            except:
                fraud_report['overall_risk'] = self._calculate_overall_risk(fraud_report['fraud_scores'])
                fraud_report['confidence'] = 0.7
        else:
            fraud_report['overall_risk'] = self._calculate_overall_risk(fraud_report['fraud_scores'])
            fraud_report['confidence'] = 0.7
        
        fraud_report['estimated_impact'] = self._estimate_impact(listing, fraud_report)
        
        fraud_report['action_recommended'] = self._recommend_action(
            fraud_report['overall_risk'], 
            fraud_report['fraud_types_detected']
        )
        
        fraud_report['network_indicators'] = self._check_network_indicators(fraud_report)
        
        return fraud_report
    
    def _calculate_overall_risk(self, scores: Dict) -> float:
        weights = {
            'mismatch': 0.25,
            'evolution': 0.20,
            'seo_manipulation': 0.15,
            'variation_abuse': 0.20,
            'hijacking': 0.30
        }
        
        total_score = 0
        total_weight = 0
        
        for key, score in scores.items():
            if key in weights:
                total_score += score * weights[key]
                total_weight += weights[key]
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_confidence(self, scores: Dict) -> float:
        active_signals = sum(1 for score in scores.values() if score > 0.3)
        high_signals = sum(1 for score in scores.values() if score > 0.7)
        
        confidence = 0.5 + (active_signals * 0.1) + (high_signals * 0.15)
        return min(confidence, 0.95)
    
    def _estimate_impact(self, listing: Dict, fraud_report: Dict) -> float:
        base_impact = 0
        
        price = listing.get('price', 0)
        monthly_sales = listing.get('estimated_monthly_sales', 100)
        base_impact = price * monthly_sales
        
        multipliers = {
            'listing_hijacking': 3.0,
            'review_product_mismatch': 2.0,
            'variation_abuse': 1.5,
            'suspicious_evolution': 2.5,
            'seo_manipulation': 1.2
        }
        
        max_multiplier = 1.0
        for fraud_type in fraud_report['fraud_types_detected']:
            max_multiplier = max(max_multiplier, multipliers.get(fraud_type, 1.0))
        
        return base_impact * max_multiplier
    
    def _recommend_action(self, risk_score: float, fraud_types: List[str]) -> Dict:
        action = {
            'severity': '',
            'actions': [],
            'priority': 0,
            'automated_actions': [],
            'manual_review_required': False
        }
        
        critical_types = ['listing_hijacking', 'review_product_mismatch']
        
        if any(ft in critical_types for ft in fraud_types) or risk_score >= 0.8:
            action['severity'] = 'CRITICAL'
            action['priority'] = 1
            action['actions'] = [
                'IMMEDIATE_LISTING_SUSPENSION',
                'FREEZE_SELLER_FUNDS',
                'NOTIFY_BRAND_OWNER',
                'INITIATE_INVESTIGATION'
            ]
            action['automated_actions'] = ['suspend_listing', 'notify_team']
            action['manual_review_required'] = True
            
        elif risk_score >= 0.6:
            action['severity'] = 'HIGH'
            action['priority'] = 2
            action['actions'] = [
                'FLAG_FOR_URGENT_REVIEW',
                'RESTRICT_VISIBILITY',
                'MONITOR_CLOSELY'
            ]
            action['automated_actions'] = ['flag_listing', 'reduce_visibility']
            action['manual_review_required'] = True
            
        elif risk_score >= 0.4:
            action['severity'] = 'MEDIUM'
            action['priority'] = 3
            action['actions'] = [
                'ADD_TO_WATCHLIST',
                'INCREASE_MONITORING',
                'REQUEST_SELLER_VERIFICATION'
            ]
            action['automated_actions'] = ['add_watchlist']
            
        elif risk_score >= 0.2:
            action['severity'] = 'LOW'
            action['priority'] = 4
            action['actions'] = [
                'ROUTINE_MONITORING',
                'COLLECT_MORE_DATA'
            ]
        else:
            action['severity'] = 'MINIMAL'
            action['priority'] = 5
            action['actions'] = ['NO_ACTION_REQUIRED']
        
        return action
    
    def _check_network_indicators(self, fraud_report: Dict) -> List[str]:
        indicators = []
        
        if fraud_report['confidence'] > 0.8 and len(fraud_report['fraud_types_detected']) >= 3:
            indicators.append('MULTI_VECTOR_ATTACK')
        
        fraud_combos = {
            ('listing_hijacking', 'variation_abuse'): 'HIJACK_AND_EXPLOIT',
            ('review_product_mismatch', 'seo_manipulation'): 'DECEPTIVE_MARKETING',
            ('suspicious_evolution', 'listing_hijacking'): 'GRADUAL_TAKEOVER'
        }
        
        detected = set(fraud_report['fraud_types_detected'])
        for combo, indicator in fraud_combos.items():
            if all(ft in detected for ft in combo):
                indicators.append(indicator)
        
        return indicators
    
    def train_ensemble_model(self, training_data: pd.DataFrame):
        feature_columns = [
            'feature_mismatch_score',
            'feature_evolution_score', 
            'feature_seo_score',
            'feature_variation_score',
            'feature_hijack_score',
            'feature_review_count',
            'feature_price_variance',
            'feature_seller_age',
            'feature_category_competitiveness'
        ]
        
        X = training_data[feature_columns].values
        
        label_mapping = {
            'legitimate': 0,
            'suspicious': 1,
            'fraudulent': 1
        }
        y = training_data['overall_fraud_label'].map(label_mapping).values
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.ensemble_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.ensemble_model.fit(X_scaled, y)
        
        joblib.dump(self.ensemble_model, 'listing_fraud_ensemble_model.pkl')
        joblib.dump(self.scaler, 'listing_fraud_scaler.pkl')
    
    def load_models(self):
        try:
            self.ensemble_model = joblib.load('listing_fraud_ensemble_model.pkl')
            self.scaler = joblib.load('listing_fraud_scaler.pkl')
            
            self.mismatch_detector.load_model()
            self.evolution_tracker.load_model()
            self.seo_detector.load_model()
            self.variation_detector.load_model()
            self.hijack_detector.load_model()
        except Exception as e:
            print(f"Error loading models: {e}")
