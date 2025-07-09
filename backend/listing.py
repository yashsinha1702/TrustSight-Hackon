"""
LISTING FRAUD DETECTOR - COMPLETE PRODUCTION VERSION
Amazon HackOn - TrustSight Platform
Detects ALL listing fraud types across ALL Amazon categories
"""
from typing import Dict, Any

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

# Then add after imports but before any model initialization:
import torch
torch.set_num_threads(4)  # Use 4 CPU threads for better performance
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
# --- Patch for _loss module error ---
import sys
import types

_loss = types.ModuleType('_loss')
for name in [
    'CyHalfBinomialLoss', 'CyHalfPoissonLoss', 'CyHalfSquaredError', 'CyHalfMultinomialLoss',
    'CyAbsoluteError', 'CyPinballLoss', 'CyHalfGammaLoss', 'CyHalfTweedieLoss',
    'CyHalfTweedieLossIdentity', 'CyExponentialLoss'
]:
    setattr(_loss, name, type(name, (), {}))  # create dummy class

sys.modules['_loss'] = _loss


class ListingFraudInference:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, listing_features: Dict[str, float]) -> Dict[str, Any]:
        feature_vector = [
            listing_features.get("feature_mismatch_score", 0.0),
            listing_features.get("feature_evolution_score", 0.0),
            listing_features.get("feature_seo_score", 0.0),
            listing_features.get("feature_variation_score", 0.0),
            listing_features.get("feature_hijack_score", 0.0),
            listing_features.get("feature_review_count", 0.0),
            listing_features.get("feature_price_variance", 0.0),
            listing_features.get("feature_seller_age", 0.0),
            listing_features.get("feature_category_competitiveness", 0.0),
        ]
        scaled = self.scaler.transform([feature_vector])
        pred = self.model.predict(scaled)[0]
        prob = self.model.predict_proba(scaled)[0].tolist() if hasattr(self.model, "predict_proba") else [None]
        return {
            "label": str(pred),
            "confidence": max(prob) if prob else None,
            "probabilities": prob
        }


class ListingFraudDetector:
    """
    COMPLETE Production-Ready Listing Fraud Detection System
    """
    
    def __init__(self):
        print("Initializing Listing Fraud Detector...")
        
        # Initialize all sub-detectors
        self.mismatch_detector = ReviewProductMismatchDetector()
        self.evolution_tracker = ListingEvolutionTracker()
        self.seo_detector = SEOManipulationDetector()
        self.variation_detector = VariationAbuseDetector()
        self.hijack_detector = ListingHijackingDetector()
        
        # Initialize models
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
        # Load Amazon's complete category taxonomy
        self.category_taxonomy = AmazonCategoryTaxonomy()
        
        # Fraud thresholds
        self.thresholds = {
            'mismatch': 0.3,
            'evolution': 0.4,
            'seo': 0.35,
            'variation': 0.4,
            'hijacking': 0.5
        }
        
        print("Listing Fraud Detector initialized successfully!")
        
    def detect_listing_fraud(self, listing: Dict, reviews: List[Dict] = None, 
                           historical_data: List[Dict] = None) -> Dict:
        """
        Main detection method - Comprehensive fraud analysis
        
        Args:
            listing: Current listing data
            reviews: List of reviews for the listing
            historical_data: Historical snapshots of the listing
            
        Returns:
            Comprehensive fraud report with evidence
        """
        
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
        
        # Extract features for ML model
        features = []
        
        # 1. Review-Product Mismatch Detection
        if reviews:
            mismatch_results = self.mismatch_detector.detect(listing, reviews)
            fraud_report['fraud_scores']['mismatch'] = mismatch_results['score']
            fraud_report['evidence'].extend(mismatch_results['evidence'])
            features.extend(mismatch_results['features'])
            
            if mismatch_results['score'] > self.thresholds['mismatch']:
                fraud_report['fraud_types_detected'].append('review_product_mismatch')
        
        # 2. Listing Evolution Tracking
        if historical_data:
            evolution_results = self.evolution_tracker.analyze(listing, historical_data)
            fraud_report['fraud_scores']['evolution'] = evolution_results['score']
            fraud_report['evidence'].extend(evolution_results['evidence'])
            features.extend(evolution_results['features'])
            
            if evolution_results['score'] > self.thresholds['evolution']:
                fraud_report['fraud_types_detected'].append('suspicious_evolution')
        
        # 3. SEO Manipulation Detection
        seo_results = self.seo_detector.detect(listing)
        fraud_report['fraud_scores']['seo_manipulation'] = seo_results['score']
        fraud_report['evidence'].extend(seo_results['evidence'])
        features.extend(seo_results['features'])
        
        if seo_results['score'] > self.thresholds['seo']:
            fraud_report['fraud_types_detected'].append('seo_manipulation')
        
        # 4. Variation Abuse Detection
        if listing.get('variations'):
            variation_results = self.variation_detector.detect(listing)
            fraud_report['fraud_scores']['variation_abuse'] = variation_results['score']
            fraud_report['evidence'].extend(variation_results['evidence'])
            features.extend(variation_results['features'])
            
            if variation_results['score'] > self.thresholds['variation']:
                fraud_report['fraud_types_detected'].append('variation_abuse')
        
        # 5. Listing Hijacking Detection
        if historical_data:
            hijack_results = self.hijack_detector.detect(listing, historical_data)
            fraud_report['fraud_scores']['hijacking'] = hijack_results['score']
            fraud_report['evidence'].extend(hijack_results['evidence'])
            features.extend(hijack_results['features'])
            
            if hijack_results['score'] > self.thresholds['hijacking']:
                fraud_report['fraud_types_detected'].append('listing_hijacking')
        
        # 6. Calculate overall risk using ensemble model
        if self.ensemble_model and features:
            try:
                features_array = np.array(features).reshape(1, -1)
                features_scaled = self.scaler.transform(features_array)
                
                fraud_report['overall_risk'] = float(self.ensemble_model.predict_proba(features_scaled)[0, 1])
                fraud_report['confidence'] = self._calculate_confidence(fraud_report['fraud_scores'])
            except:
                # Fallback to weighted average
                fraud_report['overall_risk'] = self._calculate_overall_risk(fraud_report['fraud_scores'])
                fraud_report['confidence'] = 0.7
        else:
            fraud_report['overall_risk'] = self._calculate_overall_risk(fraud_report['fraud_scores'])
            fraud_report['confidence'] = 0.7
        
        # 7. Estimate financial impact
        fraud_report['estimated_impact'] = self._estimate_impact(listing, fraud_report)
        
        # 8. Recommend action
        fraud_report['action_recommended'] = self._recommend_action(
            fraud_report['overall_risk'], 
            fraud_report['fraud_types_detected']
        )
        
        # 9. Check for network indicators
        fraud_report['network_indicators'] = self._check_network_indicators(fraud_report)
        
        return fraud_report
    
    def _calculate_overall_risk(self, scores: Dict) -> float:
        """Calculate weighted overall risk score"""
        weights = {
            'mismatch': 0.25,
            'evolution': 0.20,
            'seo_manipulation': 0.15,
            'variation_abuse': 0.20,
            'hijacking': 0.30  # Highest weight for hijacking
        }
        
        total_score = 0
        total_weight = 0
        
        for key, score in scores.items():
            if key in weights:
                total_score += score * weights[key]
                total_weight += weights[key]
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_confidence(self, scores: Dict) -> float:
        """Calculate confidence in the fraud detection"""
        # Higher confidence when multiple signals agree
        active_signals = sum(1 for score in scores.values() if score > 0.3)
        high_signals = sum(1 for score in scores.values() if score > 0.7)
        
        confidence = 0.5 + (active_signals * 0.1) + (high_signals * 0.15)
        return min(confidence, 0.95)
    
    def _estimate_impact(self, listing: Dict, fraud_report: Dict) -> float:
        """Estimate financial impact of the fraud"""
        base_impact = 0
        
        # Price-based impact
        price = listing.get('price', 0)
        monthly_sales = listing.get('estimated_monthly_sales', 100)
        base_impact = price * monthly_sales
        
        # Multipliers based on fraud type
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
        """Recommend specific actions based on risk and fraud types"""
        action = {
            'severity': '',
            'actions': [],
            'priority': 0,
            'automated_actions': [],
            'manual_review_required': False
        }
        
        # Critical fraud types requiring immediate action
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
        """Check if this fraud is part of a larger network"""
        indicators = []
        
        # High confidence fraud across multiple categories suggests network
        if fraud_report['confidence'] > 0.8 and len(fraud_report['fraud_types_detected']) >= 3:
            indicators.append('MULTI_VECTOR_ATTACK')
        
        # Specific combinations indicate organized fraud
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
        """Train the ensemble fraud detection model"""
        print("Training ensemble model...")
        
        # The integrated dataset already has the feature columns
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
        
        # Map fraud labels to binary
        label_mapping = {
            'legitimate': 0,
            'suspicious': 1,
            'fraudulent': 1
        }
        y = training_data['overall_fraud_label'].map(label_mapping).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble model
        self.ensemble_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.ensemble_model.fit(X_scaled, y)
        
        # Save models
        joblib.dump(self.ensemble_model, 'listing_fraud_ensemble_model.pkl')
        joblib.dump(self.scaler, 'listing_fraud_scaler.pkl')
        
        print("Ensemble model trained successfully!")
    
    def load_models(self):
        """Load all pre-trained models"""
        try:
            self.ensemble_model = joblib.load('listing_fraud_ensemble_model.pkl')
            self.scaler = joblib.load('listing_fraud_scaler.pkl')
            
            # Load sub-detector models
            self.mismatch_detector.load_model()
            self.evolution_tracker.load_model()
            self.seo_detector.load_model()
            self.variation_detector.load_model()
            self.hijack_detector.load_model()
            
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please train models first using train_all_models()")


class AmazonCategoryTaxonomy:
    """
    Complete Amazon Category Taxonomy for accurate mismatch detection
    """
    
    def __init__(self):
        # COMPLETE Amazon category structure
        self.categories = {
            # Electronics & Technology
            'electronics': {
                'name': 'Electronics',
                'keywords': ['electronic', 'device', 'gadget', 'tech', 'digital'],
                'subcategories': {
                    'computers': ['laptop', 'desktop', 'pc', 'computer', 'notebook', 'chromebook', 'macbook'],
                    'phones': ['phone', 'smartphone', 'iphone', 'android', 'mobile', 'cellular'],
                    'tablets': ['tablet', 'ipad', 'kindle', 'surface'],
                    'cameras': ['camera', 'dslr', 'mirrorless', 'webcam', 'gopro', 'camcorder'],
                    'audio': ['headphone', 'earphone', 'speaker', 'microphone', 'earbuds', 'airpods'],
                    'tv_video': ['tv', 'television', 'monitor', 'projector', 'display', 'screen'],
                    'gaming': ['playstation', 'xbox', 'nintendo', 'console', 'controller', 'gaming'],
                    'accessories': ['cable', 'charger', 'adapter', 'battery', 'case', 'cover', 'mount']
                },
                'incompatible': ['food', 'pet_food', 'plants', 'seeds', 'live_animals']
            },
            
            # Fashion & Apparel
            'clothing': {
                'name': 'Clothing, Shoes & Jewelry',
                'keywords': ['clothing', 'apparel', 'fashion', 'wear', 'outfit'],
                'subcategories': {
                    'mens_clothing': ['mens', 'men', 'shirt', 'pants', 'jacket', 'suit', 'tie'],
                    'womens_clothing': ['womens', 'women', 'dress', 'blouse', 'skirt', 'top'],
                    'kids_clothing': ['kids', 'children', 'boys', 'girls', 'baby'],
                    'shoes': ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'loafer', 'slipper'],
                    'jewelry': ['ring', 'necklace', 'bracelet', 'earring', 'pendant', 'chain'],
                    'watches': ['watch', 'smartwatch', 'timepiece'],
                    'accessories': ['belt', 'wallet', 'purse', 'handbag', 'scarf', 'hat', 'gloves']
                },
                'incompatible': ['electronics', 'appliances', 'automotive', 'tools', 'industrial']
            },
            
            # Home & Kitchen
            'home_kitchen': {
                'name': 'Home & Kitchen',
                'keywords': ['home', 'kitchen', 'household', 'domestic'],
                'subcategories': {
                    'furniture': ['chair', 'table', 'sofa', 'bed', 'desk', 'cabinet', 'shelf'],
                    'kitchen': ['cookware', 'pot', 'pan', 'knife', 'utensil', 'appliance'],
                    'bedding': ['mattress', 'pillow', 'sheet', 'blanket', 'comforter', 'duvet'],
                    'bath': ['towel', 'shower', 'bathroom', 'toilet', 'sink'],
                    'decor': ['decoration', 'lamp', 'vase', 'frame', 'artwork', 'rug'],
                    'storage': ['container', 'box', 'organizer', 'basket', 'bin'],
                    'cleaning': ['vacuum', 'mop', 'cleaner', 'detergent', 'sponge']
                },
                'incompatible': ['clothing', 'automotive', 'industrial']
            },
            
            # Books & Media
            'books': {
                'name': 'Books',
                'keywords': ['book', 'novel', 'read', 'literature', 'publication'],
                'subcategories': {
                    'fiction': ['novel', 'story', 'fiction', 'fantasy', 'mystery', 'romance'],
                    'nonfiction': ['biography', 'history', 'science', 'self-help', 'cookbook'],
                    'textbooks': ['textbook', 'educational', 'academic', 'study'],
                    'childrens': ['children', 'kids', 'picture book', 'young adult'],
                    'ebooks': ['kindle', 'ebook', 'digital book'],
                    'audiobooks': ['audiobook', 'audible', 'audio book']
                },
                'incompatible': ['electronics', 'tools', 'automotive', 'fresh_food']
            },
            
            # Sports & Outdoors
            'sports_outdoors': {
                'name': 'Sports & Outdoors',
                'keywords': ['sports', 'fitness', 'outdoor', 'exercise', 'athletic'],
                'subcategories': {
                    'exercise': ['gym', 'weight', 'yoga', 'fitness', 'workout', 'exercise'],
                    'outdoor': ['camping', 'hiking', 'tent', 'backpack', 'outdoor'],
                    'sports_equipment': ['ball', 'racket', 'bat', 'golf', 'tennis', 'basketball'],
                    'cycling': ['bike', 'bicycle', 'cycling', 'helmet'],
                    'water_sports': ['swimming', 'surf', 'kayak', 'fishing'],
                    'winter_sports': ['ski', 'snowboard', 'ice', 'hockey']
                },
                'incompatible': ['books', 'office_supplies', 'baby_formula']
            },
            
            # Health & Beauty
            'health_beauty': {
                'name': 'Health & Personal Care',
                'keywords': ['health', 'beauty', 'personal care', 'wellness'],
                'subcategories': {
                    'vitamins': ['vitamin', 'supplement', 'mineral', 'protein'],
                    'medical': ['medical', 'first aid', 'bandage', 'thermometer'],
                    'beauty': ['makeup', 'cosmetic', 'skincare', 'beauty'],
                    'personal_care': ['shampoo', 'soap', 'toothpaste', 'deodorant'],
                    'health_devices': ['blood pressure', 'glucose', 'oximeter', 'scale']
                },
                'incompatible': ['tools', 'automotive', 'industrial', 'raw_materials']
            },
            
            # Food & Grocery
            'grocery': {
                'name': 'Grocery & Gourmet Food',
                'keywords': ['food', 'grocery', 'eat', 'drink', 'consumable'],
                'subcategories': {
                    'fresh': ['fresh', 'fruit', 'vegetable', 'meat', 'dairy'],
                    'pantry': ['canned', 'pasta', 'rice', 'cereal', 'snack'],
                    'beverages': ['coffee', 'tea', 'juice', 'soda', 'water'],
                    'gourmet': ['gourmet', 'specialty', 'organic', 'artisan'],
                    'baby_food': ['formula', 'baby food', 'toddler']
                },
                'incompatible': ['electronics', 'tools', 'automotive', 'clothing']
            },
            
            # Automotive
            'automotive': {
                'name': 'Automotive',
                'keywords': ['car', 'auto', 'vehicle', 'automotive', 'truck'],
                'subcategories': {
                    'parts': ['part', 'engine', 'brake', 'filter', 'spark plug'],
                    'accessories': ['seat cover', 'floor mat', 'phone mount', 'dash cam'],
                    'tools': ['wrench', 'jack', 'diagnostic', 'tool'],
                    'care': ['oil', 'wax', 'cleaner', 'polish'],
                    'tires': ['tire', 'wheel', 'rim']
                },
                'incompatible': ['clothing', 'food', 'books', 'baby', 'beauty']
            },
            
            # Tools & Home Improvement
            'tools': {
                'name': 'Tools & Home Improvement',
                'keywords': ['tool', 'hardware', 'improvement', 'repair', 'build'],
                'subcategories': {
                    'power_tools': ['drill', 'saw', 'sander', 'grinder', 'power tool'],
                    'hand_tools': ['hammer', 'screwdriver', 'wrench', 'pliers'],
                    'hardware': ['screw', 'nail', 'bolt', 'fastener', 'hinge'],
                    'electrical': ['wire', 'outlet', 'switch', 'electrical'],
                    'plumbing': ['pipe', 'faucet', 'valve', 'plumbing'],
                    'paint': ['paint', 'brush', 'roller', 'primer']
                },
                'incompatible': ['clothing', 'food', 'baby', 'books', 'beauty']
            },
            
            # Toys & Games
            'toys_games': {
                'name': 'Toys & Games',
                'keywords': ['toy', 'game', 'play', 'fun', 'entertainment'],
                'subcategories': {
                    'toys': ['toy', 'doll', 'action figure', 'lego', 'plush'],
                    'games': ['board game', 'card game', 'puzzle', 'game'],
                    'outdoor_play': ['swing', 'slide', 'trampoline', 'playhouse'],
                    'educational': ['learning', 'stem', 'educational toy'],
                    'arts_crafts': ['craft', 'art', 'coloring', 'clay']
                },
                'incompatible': ['tools', 'automotive', 'industrial', 'medical']
            },
            
            # Pet Supplies
            'pet_supplies': {
                'name': 'Pet Supplies',
                'keywords': ['pet', 'dog', 'cat', 'animal', 'pet supply'],
                'subcategories': {
                    'dog': ['dog', 'puppy', 'canine'],
                    'cat': ['cat', 'kitten', 'feline'],
                    'fish': ['fish', 'aquarium', 'tank'],
                    'bird': ['bird', 'cage', 'perch'],
                    'small_animal': ['hamster', 'rabbit', 'guinea pig']
                },
                'incompatible': ['human_food', 'clothing', 'electronics', 'beauty']
            },
            
            # Baby
            'baby': {
                'name': 'Baby',
                'keywords': ['baby', 'infant', 'newborn', 'toddler'],
                'subcategories': {
                    'feeding': ['bottle', 'formula', 'baby food', 'high chair'],
                    'diapering': ['diaper', 'wipe', 'changing'],
                    'nursery': ['crib', 'bassinet', 'mobile', 'monitor'],
                    'clothing': ['onesie', 'baby clothes', 'bib'],
                    'travel': ['stroller', 'car seat', 'carrier'],
                    'toys': ['rattle', 'teether', 'baby toy']
                },
                'incompatible': ['tools', 'automotive', 'industrial', 'adult_items']
            },
            
            # Office Products
            'office': {
                'name': 'Office Products',
                'keywords': ['office', 'desk', 'work', 'business', 'stationery'],
                'subcategories': {
                    'supplies': ['pen', 'paper', 'stapler', 'tape', 'folder'],
                    'furniture': ['desk', 'chair', 'filing cabinet', 'bookshelf'],
                    'technology': ['printer', 'scanner', 'shredder', 'calculator'],
                    'organization': ['organizer', 'planner', 'calendar', 'label']
                },
                'incompatible': ['food', 'pet_supplies', 'automotive', 'baby']
            },
            
            # Garden & Outdoor
            'garden_outdoor': {
                'name': 'Garden & Outdoor',
                'keywords': ['garden', 'lawn', 'outdoor', 'yard', 'patio'],
                'subcategories': {
                    'gardening': ['plant', 'seed', 'soil', 'fertilizer', 'pot'],
                    'lawn_care': ['mower', 'trimmer', 'sprinkler', 'hose'],
                    'outdoor_living': ['grill', 'patio furniture', 'umbrella', 'fire pit'],
                    'pest_control': ['pesticide', 'trap', 'repellent']
                },
                'incompatible': ['electronics', 'clothing', 'books', 'office']
            },
            
            # Industrial & Scientific
            'industrial': {
                'name': 'Industrial & Scientific',
                'keywords': ['industrial', 'scientific', 'commercial', 'professional'],
                'subcategories': {
                    'lab': ['beaker', 'microscope', 'lab equipment', 'chemical'],
                    'safety': ['safety', 'protective', 'gloves', 'goggles', 'mask'],
                    'janitorial': ['cleaning', 'janitorial', 'commercial cleaning'],
                    'material_handling': ['pallet', 'forklift', 'warehouse'],
                    'fasteners': ['industrial fastener', 'bulk hardware']
                },
                'incompatible': ['toys', 'baby', 'food', 'clothing', 'home_decor']
            }
        }
        
        # Build reverse mappings for fast lookup
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build fast lookup tables"""
        self.keyword_to_category = {}
        self.subcategory_to_main = {}
        
        for main_cat, data in self.categories.items():
            # Map keywords to categories
            for keyword in data['keywords']:
                self.keyword_to_category[keyword] = main_cat
            
            # Map subcategories
            for subcat, keywords in data['subcategories'].items():
                self.subcategory_to_main[subcat] = main_cat
                for keyword in keywords:
                    self.keyword_to_category[keyword] = main_cat
    
    def identify_category(self, text: str) -> Tuple[str, List[str]]:
        """Identify primary category and all mentioned categories"""
        text_lower = text.lower()
        mentioned_categories = set()
        
        # Check all keywords
        for keyword, category in self.keyword_to_category.items():
            if keyword in text_lower:
                mentioned_categories.add(category)
        
        # Determine primary category (most relevant)
        if mentioned_categories:
            # Return first found as primary
            primary = list(mentioned_categories)[0]
            return primary, list(mentioned_categories)
        
        return 'unknown', []
    
    def are_categories_compatible(self, cat1: str, cat2: str) -> bool:
        """Check if two categories are compatible"""
        if cat1 == cat2:
            return True
        
        # Check incompatibility lists
        if cat1 in self.categories:
            if cat2 in self.categories[cat1].get('incompatible', []):
                return False
        
        if cat2 in self.categories:
            if cat1 in self.categories[cat2].get('incompatible', []):
                return False
        
        return True


class ReviewProductMismatchDetector:
    """
    Production-ready mismatch detection across ALL categories
    """
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.category_taxonomy = AmazonCategoryTaxonomy()
        self.threshold = 0.3
        self.classifier = None
        self.sentence_model.to(device)

        
    def detect(self, listing: Dict, reviews: List[Dict]) -> Dict:
        """Comprehensive mismatch detection"""
        results = {
            'score': 0,
            'evidence': [],
            'mismatch_count': 0,
            'mismatch_types': defaultdict(int),
            'features': []  # For ML model
        }
        
        if not reviews:
            results['features'] = [0, 0, 0, 0, 0]  # Default features
            return results
        
        # Extract listing information
        listing_text = self._create_listing_text(listing)
        listing_embedding = self.sentence_model.encode(listing_text)
        listing_category, listing_all_cats = self.category_taxonomy.identify_category(listing_text)
        
        mismatches = []
        
        for review in reviews:
            review_text = review.get('text', '')
            if not review_text:
                continue
            
            # 1. Semantic similarity check
            review_embedding = self.sentence_model.encode(review_text)
            similarity = cosine_similarity([listing_embedding], [review_embedding])[0][0]
            
            # 2. Category mismatch check
            review_category, review_all_cats = self.category_taxonomy.identify_category(review_text)
            
            mismatch_info = {
                'review_id': review.get('id'),
                'similarity_score': float(similarity),
                'listing_category': listing_category,
                'review_mentions': review_all_cats,
                'mismatches': []
            }
            
            # Check each mentioned category
            for rev_cat in review_all_cats:
                if not self.category_taxonomy.are_categories_compatible(listing_category, rev_cat):
                    mismatch_info['mismatches'].append({
                        'type': 'incompatible_category',
                        'severity': 'high',
                        'detail': f'{listing_category} incompatible with {rev_cat}'
                    })
                    results['mismatch_types']['incompatible_category'] += 1
            
            # 3. Temporal mismatch check
            temporal_mismatch = self._check_temporal_mismatch(listing, review_text)
            if temporal_mismatch:
                mismatch_info['mismatches'].append(temporal_mismatch)
                results['mismatch_types']['temporal'] += 1
            
            # 4. Feature mismatch check
            feature_mismatch = self._check_feature_mismatch(listing, review_text)
            if feature_mismatch:
                mismatch_info['mismatches'].append(feature_mismatch)
                results['mismatch_types']['feature'] += 1
            
            # 5. Brand confusion check
            brand_mismatch = self._check_brand_confusion(listing, review_text)
            if brand_mismatch:
                mismatch_info['mismatches'].append(brand_mismatch)
                results['mismatch_types']['brand'] += 1
            
            # Add to mismatches if any issues found
            if mismatch_info['mismatches'] or similarity < self.threshold:
                mismatches.append(mismatch_info)
        
        # Calculate results
        results['mismatch_count'] = len(mismatches)
        results['score'] = len(mismatches) / len(reviews) if reviews else 0
        results['evidence'] = sorted(mismatches, 
                                   key=lambda x: len(x['mismatches']), 
                                   reverse=True)[:10]
        
        # Extract features for ML
        results['features'] = [
            results['score'],
            results['mismatch_types']['incompatible_category'] / len(reviews),
            results['mismatch_types']['temporal'] / len(reviews),
            results['mismatch_types']['feature'] / len(reviews),
            results['mismatch_types']['brand'] / len(reviews)
        ]
        
        return results
    
    def _create_listing_text(self, listing: Dict) -> str:
        """Create comprehensive text representation of listing"""
        parts = [
            listing.get('title', ''),
            listing.get('brand', ''),
            listing.get('category', ''),
            listing.get('subcategory', ''),
            ' '.join(listing.get('bullet_points', [])),
            listing.get('description', '')[:500]  # First 500 chars
        ]
        return ' '.join(filter(None, parts))
    
    def _check_temporal_mismatch(self, listing: Dict, review_text: str) -> Optional[Dict]:
        """Check for temporal inconsistencies"""
        review_lower = review_text.lower()
        
        # Extract years mentioned
        year_pattern = r'\b(19|20)\d{2}\b'
        mentioned_years = [int(y) for y in re.findall(year_pattern, review_text)]
        
        if not mentioned_years:
            return None
        
        # Check against listing launch date
        launch_date = listing.get('launch_date')
        if launch_date:
            if isinstance(launch_date, str):
                launch_year = int(launch_date[:4])
            else:
                launch_year = launch_date.year
            
            # Check for reviews mentioning years before product existed
            for year in mentioned_years:
                if year < launch_year - 1:  # Allow 1 year buffer for pre-orders
                    return {
                        'type': 'temporal_mismatch',
                        'severity': 'high',
                        'detail': f'Review mentions {year} but product launched {launch_year}'
                    }
        
        return None
    
    def _check_feature_mismatch(self, listing: Dict, review_text: str) -> Optional[Dict]:
        """Check if review mentions features the product doesn't have"""
        review_lower = review_text.lower()
        
        # Common feature mismatches
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
            # Check if review mentions feature
            review_has_feature = any(kw in review_lower for kw in keywords)
            # Check if listing has feature
            listing_has_feature = any(kw in listing_text for kw in keywords)
            
            if review_has_feature and not listing_has_feature:
                return {
                    'type': 'feature_mismatch',
                    'severity': 'medium',
                    'detail': f'Review mentions {feature_type} but listing does not'
                }
        
        return None
    
    def _check_brand_confusion(self, listing: Dict, review_text: str) -> Optional[Dict]:
        """Check if review mentions wrong brand"""
        listing_brand = listing.get('brand', '').lower()
        if not listing_brand:
            return None
        
        # Common brand confusions
        brand_groups = [
            ['nike', 'adidas', 'puma', 'reebok', 'under armour'],
            ['apple', 'samsung', 'google', 'oneplus', 'xiaomi'],
            ['sony', 'bose', 'jbl', 'beats', 'sennheiser'],
            ['dell', 'hp', 'lenovo', 'asus', 'acer'],
            ['coca cola', 'pepsi', 'sprite', 'fanta'],
            ['pampers', 'huggies', 'luvs', 'honest']
        ]
        
        review_lower = review_text.lower()
        
        # Find which group the listing brand belongs to
        for group in brand_groups:
            if listing_brand in group:
                # Check if review mentions a different brand from same group
                for other_brand in group:
                    if other_brand != listing_brand and other_brand in review_lower:
                        return {
                            'type': 'brand_confusion',
                            'severity': 'high',
                            'detail': f'Listing is {listing_brand} but review mentions {other_brand}'
                        }
        
        return None
    
    def train(self, training_data: pd.DataFrame):
        """Train mismatch detection model"""
        print("Training Review-Product Mismatch Detector...")
        
        # Force CPU for sentence transformer
        device = 'cpu'
        self.sentence_model.to(device)
        
        # Create listing_text by combining the columns you have
        training_data['listing_text'] = (
            training_data['listing_title'].fillna('') + ' ' +
            training_data['listing_category'].fillna('') + ' ' +
            training_data['listing_brand'].fillna('') + ' ' +
            training_data['listing_description'].fillna('')
        )
        
        # Process in batches to avoid memory issues
        batch_size = 100
        X = []
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data.iloc[i:i+batch_size]
            
            # Encode in batches
            listing_texts = batch['listing_text'].tolist()
            review_texts = batch['review_text'].tolist()
            
            listing_embeddings = self.sentence_model.encode(
                listing_texts, device=device
            )
      


            review_embeddings = self.sentence_model.encode(
                review_texts, device=device
            )
          

            print("STEP 3: Computing features")
            
            # Calculate features for batch
            for j in range(len(batch)):
                similarity = cosine_similarity([listing_embeddings[j]], [review_embeddings[j]])[0][0]
                
                # Category compatibility
                listing_cat, _ = self.category_taxonomy.identify_category(listing_texts[j])
                review_cat, _ = self.category_taxonomy.identify_category(review_texts[j])
                cat_compatible = 1 if self.category_taxonomy.are_categories_compatible(listing_cat, review_cat) else 0
                
                X.append([similarity, cat_compatible])
        
        X = np.array(X)
        y = training_data['is_mismatch'].values
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        self.classifier.fit(X, y)
        
        # Save model
        joblib.dump(self.classifier, 'mismatch_detector_model.pkl')
        print("Mismatch detector trained!")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.classifier = joblib.load('mismatch_detector_model.pkl')
        except:
            pass


class ListingEvolutionTracker:
    """
    Tracks listing changes over time to detect fraud
    """
    
    def __init__(self):
        self.suspicious_patterns = {
            'brand_injection': {'score': 0.9, 'threshold': 1},
            'category_hopping': {'score': 0.8, 'threshold': 2},
            'product_transformation': {'score': 0.85, 'threshold': 1},
            'price_manipulation': {'score': 0.7, 'threshold': 0.7},
            'seller_change': {'score': 0.95, 'threshold': 1},
            'location_change': {'score': 0.8, 'threshold': 1}
        }
        
        # Premium brands that fraudsters target
        self.premium_brands = [
            'nike', 'adidas', 'apple', 'samsung', 'sony', 'bose', 
            'louis vuitton', 'gucci', 'prada', 'rolex', 'omega',
            'north face', 'patagonia', 'yeti', 'dyson', 'vitamix'
        ]
        
    def analyze(self, current_listing: Dict, historical_data: List[Dict]) -> Dict:
        """Analyze listing evolution for fraud patterns"""
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
        
        # Create timeline including current listing
        timeline = sorted(historical_data + [current_listing], 
                         key=lambda x: x.get('timestamp', datetime.now()))
        
        suspicious_changes = []
        change_counts = defaultdict(int)
        
        # Analyze each transition
        for i in range(1, len(timeline)):
            prev = timeline[i-1]
            curr = timeline[i]
            
            changes = self._detect_changes(prev, curr)
            
            for change in changes:
                suspicious_changes.append(change)
                change_counts[change['type']] += 1
                
                # Add to risk timeline
                results['risk_timeline'].append({
                    'timestamp': curr.get('timestamp'),
                    'change_type': change['type'],
                    'risk_score': change['risk_score']
                })
        
        # Calculate overall score
        max_risk = 0
        for pattern_type, config in self.suspicious_patterns.items():
            if change_counts[pattern_type] >= config['threshold']:
                max_risk = max(max_risk, config['score'])
        
        results['score'] = max_risk
        results['suspicious_changes'] = suspicious_changes
        results['evidence'] = sorted(suspicious_changes, 
                                   key=lambda x: x['risk_score'], 
                                   reverse=True)[:5]
        
        # Extract features for ML
        results['features'] = [
            change_counts['brand_injection'] / len(timeline),
            change_counts['category_hopping'] / len(timeline),
            change_counts['product_transformation'] / len(timeline),
            change_counts['price_manipulation'] / len(timeline),
            change_counts['seller_change'] / len(timeline)
        ]
        
        return results
    
    def _detect_changes(self, prev: Dict, curr: Dict) -> List[Dict]:
        """Detect all suspicious changes between two snapshots"""
        changes = []
        
        # 1. Brand injection
        brand_change = self._check_brand_injection(prev, curr)
        if brand_change:
            changes.append(brand_change)
        
        # 2. Category hopping
        category_change = self._check_category_hopping(prev, curr)
        if category_change:
            changes.append(category_change)
        
        # 3. Product transformation
        transformation = self._check_product_transformation(prev, curr)
        if transformation:
            changes.append(transformation)
        
        # 4. Price manipulation
        price_change = self._check_price_manipulation(prev, curr)
        if price_change:
            changes.append(price_change)
        
        # 5. Seller change
        seller_change = self._check_seller_change(prev, curr)
        if seller_change:
            changes.append(seller_change)
        
        # 6. Location change
        location_change = self._check_location_change(prev, curr)
        if location_change:
            changes.append(location_change)
        
        return changes
    
    def _check_brand_injection(self, prev: Dict, curr: Dict) -> Optional[Dict]:
        """Check for brand injection fraud"""
        prev_brand = prev.get('brand', '').lower().strip()
        curr_brand = curr.get('brand', '').lower().strip()
        
        # No brand -> Premium brand
        if (not prev_brand or prev_brand in ['generic', 'unbranded', 'no brand']) and curr_brand:
            if curr_brand in self.premium_brands:
                return {
                    'type': 'brand_injection',
                    'timestamp': curr.get('timestamp'),
                    'change': f'No brand → {curr_brand}',
                    'risk_score': 0.95,
                    'severity': 'critical'
                }
        
        # Different brand
        if prev_brand and curr_brand and prev_brand != curr_brand:
            # Check if upgrading to premium
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
        """Check for category gaming"""
        prev_cat = prev.get('category', '').lower()
        curr_cat = curr.get('category', '').lower()
        
        if prev_cat != curr_cat and prev_cat and curr_cat:
            # Moving from competitive to less competitive categories
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
        """Check if product fundamentally changed"""
        prev_title = prev.get('title', '').lower()
        curr_title = curr.get('title', '').lower()
        
        if not prev_title or not curr_title:
            return None
        
        # Calculate word overlap
        prev_words = set(prev_title.split())
        curr_words = set(curr_title.split())
        
        if len(prev_words) > 0:
            overlap = len(prev_words.intersection(curr_words)) / len(prev_words)
            
            # Less than 30% overlap suggests transformation
            if overlap < 0.3:
                # Check for specific transformation patterns
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
        """Check for suspicious price changes"""
        prev_price = prev.get('price', 0)
        curr_price = curr.get('price', 0)
        
        if prev_price > 0 and curr_price > 0:
            price_change_pct = abs(curr_price - prev_price) / prev_price
            
            # Extreme price drop (>70%)
            if curr_price < prev_price and price_change_pct > 0.7:
                return {
                    'type': 'price_manipulation',
                    'timestamp': curr.get('timestamp'),
                    'change': f'${prev_price:.2f} → ${curr_price:.2f}',
                    'risk_score': 0.8,
                    'severity': 'high',
                    'detail': f'{price_change_pct*100:.0f}% price drop'
                }
            
            # Extreme price increase (>300%)
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
        """Check for seller changes (hijacking indicator)"""
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
        """Check for shipping location changes"""
        prev_location = prev.get('ship_from', '').lower()
        curr_location = curr.get('ship_from', '').lower()
        
        if prev_location != curr_location and prev_location and curr_location:
            # High risk location changes
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
        """Train evolution detection model"""
        # For now, using rule-based approach
        # Could implement ML model if needed
        pass
    
    def load_model(self):
        """Load pre-trained model"""
        pass


class SEOManipulationDetector:
    """
    Detects all forms of SEO manipulation and keyword abuse
    """
    
    def __init__(self):
        self.keyword_threshold = 5
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.classifier = None
        
        # Comprehensive spam patterns
        self.spam_patterns = [
            # Repetition patterns
            (r'\b(\w+)\s+\1\s+\1\b', 'triple_repetition'),
            (r'(.{10,})\1{2,}', 'phrase_repetition'),
            
            # Keyword stuffing
            (r'(?i)(best|cheap|buy|discount|sale|free shipping){5,}', 'promotional_stuffing'),
            (r'(?i)(nike|adidas|apple|samsung|sony){4,}', 'brand_stuffing'),
            
            # Hidden text indicators
            (r'\s{5,}', 'excessive_spaces'),
            (r'[\u200b\u200c\u200d\ufeff]', 'invisible_unicode'),
            
            # ALL CAPS abuse
            (r'[A-Z\s]{30,}', 'excessive_caps'),
            
            # Special character abuse
            (r'[★☆✓✗✔✘]{5,}', 'special_char_spam'),
            
            # URL spam
            (r'(?i)(www\.|http:|\.com|\.net){3,}', 'url_spam')
        ]
        
        # SEO manipulation keywords
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
        """Comprehensive SEO manipulation detection"""
        results = {
            'score': 0,
            'evidence': [],
            'manipulation_types': [],
            'features': []
        }
        
        # Combine all text fields
        text = self._extract_all_text(listing)
        
        manipulations = []
        
        # 1. Keyword stuffing analysis
        stuffing_results = self._detect_keyword_stuffing(text)
        if stuffing_results:
            manipulations.extend(stuffing_results)
        
        # 2. Spam pattern detection
        spam_results = self._detect_spam_patterns(text)
        if spam_results:
            manipulations.extend(spam_results)
        
        # 3. Hidden text detection
        hidden_results = self._detect_hidden_text(listing)
        if hidden_results:
            manipulations.extend(hidden_results)
        
        # 4. Category/brand abuse
        category_abuse = self._detect_category_abuse(text, listing)
        if category_abuse:
            manipulations.extend(category_abuse)
        
        # 5. Manipulation keyword detection
        keyword_manipulation = self._detect_manipulation_keywords(text)
        if keyword_manipulation:
            manipulations.extend(keyword_manipulation)
        
        # 6. Title manipulation
        title_manipulation = self._detect_title_manipulation(listing.get('title', ''))
        if title_manipulation:
            manipulations.extend(title_manipulation)
        
        # Calculate score
        results['manipulation_types'] = manipulations
        results['evidence'] = manipulations[:10]  # Top 10
        results['score'] = min(len(manipulations) * 0.15, 1.0)
        
        # Extract features for ML
        results['features'] = [
            len(stuffing_results) / 10.0,
            len(spam_results) / 5.0,
            len(hidden_results) / 3.0,
            len(category_abuse) / 5.0,
            len(keyword_manipulation) / 10.0
        ]
        
        return results
    
    def _extract_all_text(self, listing: Dict) -> str:
        """Extract all text from listing"""
        parts = [
            listing.get('title', ''),
            listing.get('description', ''),
            ' '.join(listing.get('bullet_points', [])),
            ' '.join(listing.get('search_terms', [])),
            listing.get('backend_keywords', '')
        ]
        return ' '.join(filter(None, parts))
    
    def _detect_keyword_stuffing(self, text: str) -> List[Dict]:
        """Detect keyword stuffing"""
        manipulations = []
        words = text.lower().split()
        word_counts = Counter(words)
        
        # Find over-repeated keywords
        for word, count in word_counts.items():
            if len(word) > 3 and count > self.keyword_threshold:
                density = count / len(words) if words else 0
                
                if density > 0.05:  # More than 5% of text
                    manipulations.append({
                        'type': 'keyword_stuffing',
                        'severity': 'high' if density > 0.1 else 'medium',
                        'keyword': word,
                        'count': count,
                        'density': f'{density*100:.1f}%'
                    })
        
        # Check for brand name stuffing
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
        """Detect spam patterns using regex"""
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
        """Detect hidden text manipulation"""
        manipulations = []
        
        description = listing.get('description', '')
        
        # Check for excessive whitespace
        if '     ' in description:  # 5+ spaces
            manipulations.append({
                'type': 'hidden_text_whitespace',
                'severity': 'medium',
                'detail': 'Excessive whitespace detected'
            })
        
        # Check for invisible Unicode
        invisible_chars = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2060']
        for char in invisible_chars:
            if char in description:
                manipulations.append({
                    'type': 'hidden_text_unicode',
                    'severity': 'high',
                    'detail': 'Invisible Unicode characters detected'
                })
                break
        
        # Check backend keywords (if available)
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
        """Detect category stuffing"""
        manipulations = []
        text_lower = text.lower()
        
        # All major Amazon categories
        categories = [
            'electronics', 'clothing', 'shoes', 'jewelry', 'books',
            'toys', 'games', 'sports', 'outdoors', 'automotive',
            'tools', 'home', 'kitchen', 'garden', 'pet supplies',
            'beauty', 'health', 'grocery', 'baby', 'industrial',
            'handmade', 'music', 'movies', 'software'
        ]
        
        # Count category mentions
        mentioned_categories = [cat for cat in categories if cat in text_lower]
        
        # Check against actual category
        actual_category = listing.get('category', '').lower()
        
        # Too many categories mentioned
        if len(mentioned_categories) > 5:
            manipulations.append({
                'type': 'category_stuffing',
                'severity': 'high',
                'count': len(mentioned_categories),
                'categories': mentioned_categories[:10]
            })
        
        # Mentioning unrelated categories
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
        """Detect manipulation keywords"""
        manipulations = []
        text_lower = text.lower()
        
        for manipulation_type, keywords in self.manipulation_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            
            if len(matches) > 2:  # Multiple manipulation keywords
                manipulations.append({
                    'type': f'keyword_manipulation_{manipulation_type}',
                    'severity': 'medium',
                    'keywords_found': matches,
                    'count': len(matches)
                })
        
        return manipulations
    
    def _detect_title_manipulation(self, title: str) -> List[Dict]:
        """Detect title-specific manipulation"""
        manipulations = []
        
        # Title length check
        if len(title) > 200:
            manipulations.append({
                'type': 'title_too_long',
                'severity': 'medium',
                'length': len(title),
                'detail': 'Title exceeds recommended length'
            })
        
        # Special character abuse in title
        special_chars = re.findall(r'[|/\-,]', title)
        if len(special_chars) > 5:
            manipulations.append({
                'type': 'title_separator_abuse',
                'severity': 'low',
                'count': len(special_chars),
                'detail': 'Excessive separators in title'
            })
        
        # All caps words
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
        """Train SEO manipulation detection model"""
        print("Training SEO Manipulation Detector...")
        
        # Create listing_text by combining the columns you have
        training_data['listing_text'] = (
            training_data['title'].fillna('') + ' ' +
            training_data['description'].fillna('') + ' ' +
            training_data['bullet_points'].fillna('') + ' ' +
            training_data['backend_keywords'].fillna('')
        )
        
        # Extract text
        texts = training_data['listing_text'].values
        labels = training_data['has_seo_manipulation'].values
        
        # TF-IDF features
        X = self.tfidf.fit_transform(texts)
        
        # Train classifier
        from sklearn.svm import SVC
        self.classifier = SVC(probability=True, random_state=42)
        self.classifier.fit(X, labels)
        
        # Save models
        joblib.dump(self.classifier, 'seo_detector_model.pkl')
        joblib.dump(self.tfidf, 'seo_tfidf.pkl')
        
        print("SEO detector trained!")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.classifier = joblib.load('seo_detector_model.pkl')
            self.tfidf = joblib.load('seo_tfidf.pkl')
        except:
            pass


class VariationAbuseDetector:
    """
    Detects all forms of variation abuse
    """
    
    def __init__(self):
        self.max_reasonable_variations = 100
        self.price_variance_threshold = 10
        self.stock_out_threshold = 0.8
        
    def detect(self, listing: Dict) -> Dict:
        """Comprehensive variation abuse detection"""
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
        
        # 1. Unrelated variations
        unrelated_results = self._detect_unrelated_variations(variations, listing)
        if unrelated_results:
            abuses.extend(unrelated_results)
        
        # 2. Price manipulation
        price_abuse = self._detect_price_manipulation(variations)
        if price_abuse:
            abuses.extend(price_abuse)
        
        # 3. Fake variations
        fake_variations = self._detect_fake_variations(variations)
        if fake_variations:
            abuses.extend(fake_variations)
        
        # 4. Review pooling abuse
        review_pooling = self._detect_review_pooling_abuse(variations)
        if review_pooling:
            abuses.extend(review_pooling)
        
        # 5. Excessive variations
        if len(variations) > self.max_reasonable_variations:
            abuses.append({
                'type': 'excessive_variations',
                'severity': 'medium',
                'count': len(variations),
                'threshold': self.max_reasonable_variations
            })
        
        # 6. SEO variation abuse
        seo_abuse = self._detect_seo_variation_abuse(variations)
        if seo_abuse:
            abuses.extend(seo_abuse)
        
        # Calculate results
        results['abuse_types'] = abuses
        results['evidence'] = sorted(abuses, 
                                   key=lambda x: x.get('severity_score', 0), 
                                   reverse=True)[:5]
        results['score'] = min(len(abuses) * 0.2, 1.0)
        
        # Features for ML
        results['features'] = [
            len(unrelated_results) / len(variations) if variations else 0,
            1 if price_abuse else 0,
            len(fake_variations) / len(variations) if variations else 0,
            1 if review_pooling else 0,
            len(variations) / 100.0  # Normalized variation count
        ]
        
        return results
    
    def _detect_unrelated_variations(self, variations: List[Dict], listing: Dict) -> List[Dict]:
        """Detect unrelated products as variations"""
        abuses = []
        
        if not variations:
            return abuses
        
        # Get base product category from listing
        base_category = listing.get('category', '').lower()
        base_title = listing.get('title', '').lower()
        
        # Extract product types from variations
        variation_types = defaultdict(list)
        
        for var in variations:
            var_name = var.get('name', '').lower()
            var_type = self._extract_product_type(var_name)
            variation_types[var_type].append(var)
        
        # If too many different product types, it's abuse
        if len(variation_types) > 3:
            # Find the most different ones
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
        """Extract product type from name"""
        # Common product types
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
        """Check if two product types are related"""
        # Related product groups
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
        """Detect price manipulation through variations"""
        abuses = []
        
        prices = [var.get('price', 0) for var in variations if var.get('price', 0) > 0]
        
        if not prices:
            return abuses
        
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        
        # Check for extreme price variance
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
        
        # Check for price anchoring (one very low price)
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
        """Detect fake variations (always out of stock, etc.)"""
        abuses = []
        
        # Count stock status
        in_stock = 0
        out_of_stock = 0
        
        for var in variations:
            if var.get('in_stock', True) or var.get('quantity', 1) > 0:
                in_stock += 1
            else:
                out_of_stock += 1
        
        total = len(variations)
        
        # Most variations out of stock
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
        
        # Check for placeholder variations
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
        """Detect review pooling abuse"""
        abuses = []
        
        # Check if variations are too different for review pooling
        product_types = set()
        price_ranges = []
        
        for var in variations:
            var_type = self._extract_product_type(var.get('name', ''))
            product_types.add(var_type)
            
            price = var.get('price', 0)
            if price > 0:
                price_ranges.append(price)
        
        # Multiple unrelated product types sharing reviews
        if len(product_types) > 3:
            abuses.append({
                'type': 'review_pooling_abuse',
                'severity': 'high',
                'severity_score': 0.9,
                'product_types': list(product_types),
                'detail': 'Unrelated products sharing review pool'
            })
        
        # Extreme price differences sharing reviews
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
        """Detect SEO abuse through variations"""
        abuses = []
        
        # Check for keyword stuffing in variation names
        all_variation_text = ' '.join([var.get('name', '') for var in variations])
        
        # Brand stuffing
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
        
        # Duplicate variation names (with minor changes)
        variation_names = [var.get('name', '').lower().strip() for var in variations]
        
        # Find near-duplicates
        duplicates = 0
        for i, name1 in enumerate(variation_names):
            for name2 in variation_names[i+1:]:
                # Simple similarity check
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
        """Check if two variation names are suspiciously similar"""
        # Remove common variation indicators
        for pattern in ['size', 'color', 'style', 'model']:
            name1 = re.sub(r'\b\w*' + pattern + r'\w*\b', '', name1)
            name2 = re.sub(r'\b\w*' + pattern + r'\w*\b', '', name2)
        
        # Remove numbers
        name1 = re.sub(r'\d+', '', name1)
        name2 = re.sub(r'\d+', '', name2)
        
        # Check similarity
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1.intersection(words2))
            similarity = overlap / min(len(words1), len(words2))
            return similarity > 0.8
        
        return False
    
    def train(self, training_data: pd.DataFrame):
        """Train variation abuse model"""
        # Currently using rule-based approach
        pass
    
    def load_model(self):
        """Load pre-trained model"""
        pass


class ListingHijackingDetector:
    """
    Detects listing hijacking attempts
    """
    
    def __init__(self):
        self.critical_changes = {
            'seller_change': {'score': 0.95, 'severity': 'critical'},
            'brand_change': {'score': 0.9, 'severity': 'critical'},
            'location_change': {'score': 0.85, 'severity': 'high'},
            'price_crash': {'score': 0.8, 'severity': 'high'},
            'bulk_change': {'score': 0.9, 'severity': 'critical'}
        }
        
    def detect(self, current_listing: Dict, historical_data: List[Dict]) -> Dict:
        """Detect listing hijacking"""
        results = {
            'score': 0,
            'evidence': [],
            'hijack_indicators': [],
            'risk_timeline': [],
            'features': []
        }
        
        if not historical_data:
            results['features'] = [0, 0, 0, 0, 0]
            return results
        
        # Get original listing
        original = min(historical_data, 
                      key=lambda x: x.get('timestamp', datetime.now()))
        
        # Get recent history
        recent_history = sorted(historical_data[-10:], 
                              key=lambda x: x.get('timestamp', datetime.now()))
        
        indicators = []
        
        # 1. Check critical changes
        critical_changes = self._detect_critical_changes(original, current_listing)
        indicators.extend(critical_changes)
        
        # 2. Check velocity of changes
        velocity_indicators = self._detect_change_velocity(recent_history + [current_listing])
        indicators.extend(velocity_indicators)
        
        # 3. Check for patterns
        pattern_indicators = self._detect_hijack_patterns(historical_data, current_listing)
        indicators.extend(pattern_indicators)
        
        # 4. Check for preparation signs
        prep_indicators = self._detect_preparation_signs(recent_history, current_listing)
        indicators.extend(prep_indicators)
        
        # Calculate score
        if indicators:
            results['score'] = max([ind['risk_score'] for ind in indicators])
        
        results['hijack_indicators'] = indicators
        results['evidence'] = sorted(indicators, 
                                   key=lambda x: x['risk_score'], 
                                   reverse=True)[:5]
        
        # Build risk timeline
        results['risk_timeline'] = self._build_risk_timeline(historical_data, indicators)
        
        # Extract features
        results['features'] = [
            1 if any(ind['type'] == 'seller_change' for ind in indicators) else 0,
            1 if any(ind['type'] == 'brand_change' for ind in indicators) else 0,
            1 if any(ind['type'] == 'location_change' for ind in indicators) else 0,
            1 if any(ind['type'] == 'price_crash' for ind in indicators) else 0,
            len(indicators) / 10.0
        ]
        
        return results
    
    def _detect_critical_changes(self, original: Dict, current: Dict) -> List[Dict]:
        """Detect critical changes indicating hijacking"""
        indicators = []
        
        # Seller change
        if original.get('seller_id') != current.get('seller_id'):
            indicators.append({
                'type': 'seller_change',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': self.critical_changes['seller_change']['score'],
                'severity': self.critical_changes['seller_change']['severity'],
                'original_seller': original.get('seller_id'),
                'current_seller': current.get('seller_id'),
                'detail': 'Seller account changed - possible hijacking'
            })
        
        # Brand change
        orig_brand = original.get('brand', '').lower()
        curr_brand = current.get('brand', '').lower()
        
        if orig_brand and curr_brand and orig_brand != curr_brand:
            indicators.append({
                'type': 'brand_change',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': self.critical_changes['brand_change']['score'],
                'severity': self.critical_changes['brand_change']['severity'],
                'original_brand': orig_brand,
                'current_brand': curr_brand,
                'detail': 'Brand changed - high hijacking risk'
            })
        
        # Location change
        location_change = self._check_location_change(original, current)
        if location_change:
            indicators.append(location_change)
        
        # Price crash
        price_crash = self._check_price_crash(original, current)
        if price_crash:
            indicators.append(price_crash)
        
        return indicators
    
    def _check_location_change(self, original: Dict, current: Dict) -> Optional[Dict]:
        """Check for suspicious location changes"""
        orig_location = original.get('ship_from', '').lower()
        curr_location = current.get('ship_from', '').lower()
        
        if orig_location != curr_location:
            # High risk combinations
            high_risk = [
                ('united states', 'china'),
                ('usa', 'china'),
                ('fulfilled by amazon', 'seller fulfilled'),
                ('domestic', 'international')
            ]
            
            risk_score = 0.5  # Default
            
            for orig_pattern, curr_pattern in high_risk:
                if orig_pattern in orig_location and curr_pattern in curr_location:
                    risk_score = self.critical_changes['location_change']['score']
                    break
            
            return {
                'type': 'location_change',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': risk_score,
                'severity': 'high' if risk_score > 0.7 else 'medium',
                'original_location': orig_location,
                'current_location': curr_location,
                'detail': 'Shipping location changed'
            }
        
        return None
    
    def _check_price_crash(self, original: Dict, current: Dict) -> Optional[Dict]:
        """Check for suspicious price drops"""
        orig_price = original.get('price', 0)
        curr_price = current.get('price', 0)
        
        if orig_price > 0 and curr_price > 0:
            price_drop_pct = (orig_price - curr_price) / orig_price
            
            # More than 70% drop is suspicious
            if price_drop_pct > 0.7:
                return {
                    'type': 'price_crash',
                    'timestamp': current.get('timestamp', datetime.now()),
                    'risk_score': self.critical_changes['price_crash']['score'],
                    'severity': 'high',
                    'original_price': orig_price,
                    'current_price': curr_price,
                    'drop_percentage': price_drop_pct * 100,
                    'detail': f'Price dropped {price_drop_pct*100:.0f}% - possible hijacking'
                }
        
        return None
    
    def _detect_change_velocity(self, recent_history: List[Dict]) -> List[Dict]:
        """Detect rapid changes indicating hijacking"""
        indicators = []
        
        if len(recent_history) < 2:
            return indicators
        
        # Count changes in last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_changes = []
        
        for i in range(1, len(recent_history)):
            prev = recent_history[i-1]
            curr = recent_history[i]
            
            if curr.get('timestamp', datetime.now()) > seven_days_ago:
                changes = self._count_changes(prev, curr)
                if changes > 0:
                    recent_changes.append({
                        'timestamp': curr.get('timestamp'),
                        'change_count': changes
                    })
        
        # Many changes in short time
        if len(recent_changes) > 5:
            indicators.append({
                'type': 'rapid_changes',
                'timestamp': datetime.now(),
                'risk_score': 0.7,
                'severity': 'medium',
                'change_count': len(recent_changes),
                'detail': 'Multiple rapid changes detected'
            })
        
        return indicators
    
    def _count_changes(self, prev: Dict, curr: Dict) -> int:
        """Count number of significant changes"""
        changes = 0
        
        # Fields to check
        fields = ['title', 'brand', 'price', 'category', 'seller_id', 'ship_from']
        
        for field in fields:
            if prev.get(field) != curr.get(field):
                changes += 1
        
        return changes
    
    def _detect_hijack_patterns(self, historical_data: List[Dict], 
                               current: Dict) -> List[Dict]:
        """Detect known hijacking patterns"""
        indicators = []
        
        # Pattern 1: Dormant listing suddenly active
        if self._is_dormant_activation(historical_data, current):
            indicators.append({
                'type': 'dormant_activation',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': 0.75,
                'severity': 'high',
                'detail': 'Dormant listing suddenly reactivated'
            })
        
        # Pattern 2: Bulk changes at once
        if self._has_bulk_changes(historical_data[-1] if historical_data else {}, current):
            indicators.append({
                'type': 'bulk_changes',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': 0.8,
                'severity': 'high',
                'detail': 'Multiple critical fields changed simultaneously'
            })
        
        return indicators
    
    def _is_dormant_activation(self, historical_data: List[Dict], 
                              current: Dict) -> bool:
        """Check if dormant listing was reactivated"""
        if len(historical_data) < 2:
            return False
        
        # Check for long period of no changes
        sorted_history = sorted(historical_data, 
                              key=lambda x: x.get('timestamp', datetime.now()))
        
        if len(sorted_history) >= 2:
            last_change = sorted_history[-1].get('timestamp', datetime.now())
            prev_change = sorted_history[-2].get('timestamp', datetime.now())
            
            # If no changes for 6+ months, then sudden activity
            if isinstance(last_change, datetime) and isinstance(prev_change, datetime):
                gap = (last_change - prev_change).days
                if gap > 180:  # 6 months
                    return True
        
        return False
    
    def _has_bulk_changes(self, prev: Dict, curr: Dict) -> bool:
        """Check if multiple critical fields changed at once"""
        if not prev:
            return False
        
        critical_fields = ['seller_id', 'brand', 'title', 'category', 'ship_from']
        changes = sum(1 for field in critical_fields 
                     if prev.get(field) != curr.get(field))
        
        return changes >= 3
    
    def _detect_preparation_signs(self, recent_history: List[Dict], 
                                 current: Dict) -> List[Dict]:
        """Detect preparation for hijacking"""
        indicators = []
        
        # Sign 1: Inventory spike
        if self._has_inventory_spike(recent_history, current):
            indicators.append({
                'type': 'inventory_spike',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': 0.6,
                'severity': 'medium',
                'detail': 'Sudden inventory increase - possible exit scam preparation'
            })
        
        # Sign 2: Review decline
        if self._has_review_decline(recent_history):
            indicators.append({
                'type': 'review_quality_decline',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': 0.5,
                'severity': 'medium',
                'detail': 'Recent review quality decline'
            })
        
        return indicators
    
    def _has_inventory_spike(self, recent_history: List[Dict], 
                           current: Dict) -> bool:
        """Check for sudden inventory increases"""
        curr_inventory = current.get('inventory_count', 0)
        
        if recent_history and curr_inventory > 0:
            avg_inventory = sum(h.get('inventory_count', 0) for h in recent_history) / len(recent_history)
            
            # Inventory more than 3x average
            if avg_inventory > 0 and curr_inventory > avg_inventory * 3:
                return True
        
        return False
    
    def _has_review_decline(self, recent_history: List[Dict]) -> bool:
        """Check for declining review quality"""
        if len(recent_history) < 3:
            return False
        
        # Check average ratings trend
        ratings = [h.get('average_rating', 0) for h in recent_history if h.get('average_rating')]
        
        if len(ratings) >= 3:
            # Check if ratings declining
            recent_avg = sum(ratings[-3:]) / 3
            older_avg = sum(ratings[:-3]) / len(ratings[:-3]) if len(ratings) > 3 else 0
            
            if older_avg > 0 and recent_avg < older_avg * 0.8:
                return True
        
        return False
    
    def _build_risk_timeline(self, historical_data: List[Dict], 
                           indicators: List[Dict]) -> List[Dict]:
        """Build timeline of risk evolution"""
        timeline = []
        
        # Add all changes with risk scores
        for i in range(1, len(historical_data)):
            prev = historical_data[i-1]
            curr = historical_data[i]
            
            changes = self._identify_changes(prev, curr)
            if changes:
                timeline.append({
                    'timestamp': curr.get('timestamp', datetime.now()),
                    'changes': changes,
                    'risk_level': self._calculate_risk_level(changes)
                })
        
        # Add current indicators
        for indicator in indicators:
            timeline.append({
                'timestamp': indicator.get('timestamp', datetime.now()),
                'event': indicator['type'],
                'risk_level': indicator['risk_score']
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        return timeline
    
    def _identify_changes(self, prev: Dict, curr: Dict) -> List[str]:
        """Identify what changed between snapshots"""
        changes = []
        
        fields_to_check = [
            'seller_id', 'brand', 'title', 'price', 
            'category', 'ship_from', 'description'
        ]
        
        for field in fields_to_check:
            if prev.get(field) != curr.get(field):
                changes.append(field)
        
        return changes
    
    def _calculate_risk_level(self, changes: List[str]) -> float:
        """Calculate risk level based on changes"""
        risk_weights = {
            'seller_id': 0.9,
            'brand': 0.8,
            'ship_from': 0.7,
            'price': 0.5,
            'title': 0.4,
            'category': 0.6,
            'description': 0.3
        }
        
        total_risk = sum(risk_weights.get(change, 0.2) for change in changes)
        return min(total_risk, 1.0)
    
    def train(self, training_data: pd.DataFrame):
        """Train hijacking detection model"""
        # Currently using rule-based approach
        # Could implement ML model for pattern recognition
        pass
    
    def load_model(self):
        """Load pre-trained model"""
        pass

class ListingFraudModelTrainer:
    """
    Complete training system for all listing fraud models
    """
    
    def __init__(self):
        self.detector = ListingFraudDetector()
        
    def train_all_models(self, datasets: Dict[str, pd.DataFrame]):
        """Train all sub-models with provided datasets"""
        print("\n" + "="*50)
        print("TRAINING LISTING FRAUD DETECTION MODELS")
        print("="*50)
        
        # 1. Train Mismatch Detector
        if 'mismatch_data' in datasets:
            print("\n[1/5] Training Review-Product Mismatch Detector...")
            print(f"Dataset size: {len(datasets['mismatch_data'])} samples")
            self.detector.mismatch_detector.train(datasets['mismatch_data'])
            print("✓ Mismatch Detector trained successfully!")
        
        # 2. Train SEO Detector
        if 'seo_data' in datasets:
            print("\n[2/5] Training SEO Manipulation Detector...")
            print(f"Dataset size: {len(datasets['seo_data'])} samples")
            self.detector.seo_detector.train(datasets['seo_data'])
            print("✓ SEO Detector trained successfully!")
        
        # 3. Evolution Tracker (rule-based, no training needed)
        print("\n[3/5] Evolution Tracker configured (rule-based)")
        print("✓ Using predefined patterns for evolution detection")
        
        # 4. Variation Detector (rule-based, no training needed)
        print("\n[4/5] Variation Abuse Detector configured (rule-based)")
        print("✓ Using predefined patterns for variation abuse")
        
        # 5. Hijacking Detector (rule-based, no training needed)
        print("\n[5/5] Hijacking Detector configured (rule-based)")
        print("✓ Using predefined patterns for hijacking detection")
        
        # 6. Train Ensemble Model
        if 'integrated_data' in datasets:
            print("\n[FINAL] Training Ensemble Model...")
            print(f"Dataset size: {len(datasets['integrated_data'])} samples")
            self.detector.train_ensemble_model(datasets['integrated_data'])
            print("✓ Ensemble Model trained successfully!")
        
        print("\n" + "="*50)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*50)
        
        # Save configuration
        self._save_training_metadata(datasets)
    
    def _save_training_metadata(self, datasets: Dict[str, pd.DataFrame]):
        """Save training metadata for tracking"""
        metadata = {
            'training_date': datetime.now().isoformat(),
            'dataset_sizes': {name: len(df) for name, df in datasets.items()},
            'model_versions': {
                'mismatch_detector': '1.0',
                'seo_detector': '1.0',
                'evolution_tracker': '1.0',
                'variation_detector': '1.0',
                'hijacking_detector': '1.0',
                'ensemble_model': '1.0'
            }
        }
        
        with open('listing_fraud_training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def evaluate_models(self, test_datasets: Dict[str, pd.DataFrame]):
        """Evaluate all models on test data"""
        print("\n" + "="*50)
        print("EVALUATING LISTING FRAUD DETECTION MODELS")
        print("="*50)
        
        results = {}
        
        # Evaluate each model
        # Implementation would include precision, recall, F1 scores
        
        return results
    
def main():
    """
    Main function to train and test the Listing Fraud Detector
    """
    print("\n" + "="*70)
    print("TRUSTSIGHT - LISTING FRAUD DETECTOR")
    print("Amazon HackOn 2024")
    print("="*70)
    
    # Step 1: Load datasets
    print("\n[Step 1] Loading datasets...")
    try:
        datasets = {
            'mismatch_data': pd.read_csv('review_product_mismatch_50k.csv'),
            'seo_data': pd.read_csv('seo_manipulated_listings.csv'),
            'integrated_data': pd.read_csv('listing_fraud_assessments.csv')
        }
        print("✓ Datasets loaded successfully!")
        print(f"  - Mismatch data: {len(datasets['mismatch_data'])} rows")
        print(f"  - SEO data: {len(datasets['seo_data'])} rows")
        print(f"  - Integrated data: {len(datasets['integrated_data'])} rows")
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        return
    
    # Step 2: Train models
    print("\n[Step 2] Training models...")
    trainer = ListingFraudModelTrainer()
    trainer.train_all_models(datasets)
    
    # Step 3: Initialize detector with trained models
    print("\n[Step 3] Initializing detector with trained models...")
    detector = ListingFraudDetector()
    detector.load_models()
    print("✓ Detector ready!")
    
    # Step 4: Test with a comprehensive fraud case
    print("\n[Step 4] Testing detector with fraud case...")
    
    # Test case: Product with multiple fraud indicators
    test_listing = {
        'id': 'TEST_PROD_001',
        'asin': 'B08TESTTEST',
        'title': 'Nike Air Max Running Shoes Best Price Cheap Adidas Puma',
        'category': 'Shoes',
        'brand': 'Nike',
        'price': 29.99,  # Suspiciously low for Nike
        'seller_id': 'NEW_SELLER_123',
        'ship_from': 'China',
        'description': 'Best shoes nike adidas puma reebok ' * 20,  # Keyword stuffing
        'bullet_points': [
            'High quality shoes',
            'Fast shipping guaranteed',
            'Better than all competitors'
        ],
        'launch_date': datetime(2024, 1, 1),
        'variations': [
            {'name': 'Size 8 Black', 'price': 29.99, 'in_stock': True},
            {'name': 'Size 9 Black', 'price': 29.99, 'in_stock': False},
            {'name': 'iPhone Case', 'price': 19.99, 'in_stock': True},  # Unrelated!
            {'name': 'Watch Band', 'price': 15.99, 'in_stock': False}   # Unrelated!
        ]
    }
    
    test_reviews = [
        {
            'id': 'REV001',
            'text': 'This watch keeps perfect time! Love the design.',
            'rating': 5
        },
        {
            'id': 'REV002', 
            'text': 'Great product! Fast shipping! Highly recommend!',
            'rating': 5
        },
        {
            'id': 'REV003',
            'text': 'The phone case fits my iPhone perfectly.',
            'rating': 5
        }
    ]
    
    historical_data = [
        {
            'timestamp': datetime(2024, 1, 1),
            'title': 'Generic Running Shoes',
            'brand': '',
            'price': 39.99,
            'seller_id': 'OLD_SELLER_456',
            'ship_from': 'USA'
        }
    ]
    
    # Run detection
    results = detector.detect_listing_fraud(test_listing, test_reviews, historical_data)
    
    # Step 5: Display results
    print("\n" + "="*70)
    print("FRAUD DETECTION RESULTS")
    print("="*70)
    
    print(f"\nListing: {test_listing['title'][:50]}...")
    print(f"ASIN: {results['asin']}")
    
    # Risk visualization
    risk_level = results['overall_risk']
    risk_bar = "█" * int(risk_level * 20) + "░" * (20 - int(risk_level * 20))
    print(f"\nOverall Risk: [{risk_bar}] {risk_level:.1%}")
    print(f"Confidence: {results['confidence']:.1%}")
    
    # Fraud types detected
    if results['fraud_types_detected']:
        print("\n🚨 Fraud Types Detected:")
        for fraud_type in results['fraud_types_detected']:
            print(f"   • {fraud_type.replace('_', ' ').title()}")
    
    # Individual scores
    print("\nDetailed Analysis:")
    for score_type, score in results['fraud_scores'].items():
        if score > 0:
            indicator = "🔴" if score > 0.7 else "🟡" if score > 0.3 else "🟢"
            print(f"   {indicator} {score_type.replace('_', ' ').title()}: {score:.1%}")
    
    # Top evidence
    print("\n📊 Top Evidence:")
    for i, evidence in enumerate(results['evidence'][:3], 1):
        print(f"   {i}. {evidence}")
    
    # Recommended action
    action = results['action_recommended']
    print(f"\n⚡ Recommended Action: {action['severity']}")
    for act in action['actions'][:3]:
        print(f"   → {act}")
    
    # Network indicators
    if results['network_indicators']:
        print("\n🕸️  Network Indicators:")
        for indicator in results['network_indicators']:
            print(f"   ⚠️  {indicator}")
    
    print("\n" + "="*70)
    print("Detection complete! Models saved and ready for production use.")
    print("="*70)


if __name__ == "__main__":
    # Run the main function
    main()