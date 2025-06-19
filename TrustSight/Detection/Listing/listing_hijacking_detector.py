import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class ListingHijackingDetector:
    """Detects listing hijacking attempts through analysis of critical changes and suspicious patterns."""
    
    def __init__(self):
        self.critical_changes = {
            'seller_change': {'score': 0.95, 'severity': 'critical'},
            'brand_change': {'score': 0.9, 'severity': 'critical'},
            'location_change': {'score': 0.85, 'severity': 'high'},
            'price_crash': {'score': 0.8, 'severity': 'high'},
            'bulk_change': {'score': 0.9, 'severity': 'critical'}
        }
        
    def detect(self, current_listing: Dict, historical_data: List[Dict]) -> Dict:
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
        
        original = min(historical_data, 
                      key=lambda x: x.get('timestamp', datetime.now()))
        
        recent_history = sorted(historical_data[-10:], 
                              key=lambda x: x.get('timestamp', datetime.now()))
        
        indicators = []
        
        critical_changes = self._detect_critical_changes(original, current_listing)
        indicators.extend(critical_changes)
        
        velocity_indicators = self._detect_change_velocity(recent_history + [current_listing])
        indicators.extend(velocity_indicators)
        
        pattern_indicators = self._detect_hijack_patterns(historical_data, current_listing)
        indicators.extend(pattern_indicators)
        
        prep_indicators = self._detect_preparation_signs(recent_history, current_listing)
        indicators.extend(prep_indicators)
        
        if indicators:
            results['score'] = max([ind['risk_score'] for ind in indicators])
        
        results['hijack_indicators'] = indicators
        results['evidence'] = sorted(indicators, 
                                   key=lambda x: x['risk_score'], 
                                   reverse=True)[:5]
        
        results['risk_timeline'] = self._build_risk_timeline(historical_data, indicators)
        
        results['features'] = [
            1 if any(ind['type'] == 'seller_change' for ind in indicators) else 0,
            1 if any(ind['type'] == 'brand_change' for ind in indicators) else 0,
            1 if any(ind['type'] == 'location_change' for ind in indicators) else 0,
            1 if any(ind['type'] == 'price_crash' for ind in indicators) else 0,
            len(indicators) / 10.0
        ]
        
        return results
    
    def _detect_critical_changes(self, original: Dict, current: Dict) -> List[Dict]:
        indicators = []
        
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
        
        location_change = self._check_location_change(original, current)
        if location_change:
            indicators.append(location_change)
        
        price_crash = self._check_price_crash(original, current)
        if price_crash:
            indicators.append(price_crash)
        
        return indicators
    
    def _check_location_change(self, original: Dict, current: Dict) -> Optional[Dict]:
        orig_location = original.get('ship_from', '').lower()
        curr_location = current.get('ship_from', '').lower()
        
        if orig_location != curr_location:
            high_risk = [
                ('united states', 'china'),
                ('usa', 'china'),
                ('fulfilled by amazon', 'seller fulfilled'),
                ('domestic', 'international')
            ]
            
            risk_score = 0.5
            
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
        orig_price = original.get('price', 0)
        curr_price = current.get('price', 0)
        
        if orig_price > 0 and curr_price > 0:
            price_drop_pct = (orig_price - curr_price) / orig_price
            
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
        indicators = []
        
        if len(recent_history) < 2:
            return indicators
        
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
        changes = 0
        
        fields = ['title', 'brand', 'price', 'category', 'seller_id', 'ship_from']
        
        for field in fields:
            if prev.get(field) != curr.get(field):
                changes += 1
        
        return changes
    
    def _detect_hijack_patterns(self, historical_data: List[Dict], 
                               current: Dict) -> List[Dict]:
        indicators = []
        
        if self._is_dormant_activation(historical_data, current):
            indicators.append({
                'type': 'dormant_activation',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': 0.75,
                'severity': 'high',
                'detail': 'Dormant listing suddenly reactivated'
            })
        
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
        if len(historical_data) < 2:
            return False
        
        sorted_history = sorted(historical_data, 
                              key=lambda x: x.get('timestamp', datetime.now()))
        
        if len(sorted_history) >= 2:
            last_change = sorted_history[-1].get('timestamp', datetime.now())
            prev_change = sorted_history[-2].get('timestamp', datetime.now())
            
            if isinstance(last_change, datetime) and isinstance(prev_change, datetime):
                gap = (last_change - prev_change).days
                if gap > 180:
                    return True
        
        return False
    
    def _has_bulk_changes(self, prev: Dict, curr: Dict) -> bool:
        if not prev:
            return False
        
        critical_fields = ['seller_id', 'brand', 'title', 'category', 'ship_from']
        changes = sum(1 for field in critical_fields 
                     if prev.get(field) != curr.get(field))
        
        return changes >= 3
    
    def _detect_preparation_signs(self, recent_history: List[Dict], 
                                 current: Dict) -> List[Dict]:
        indicators = []
        
        if self._has_inventory_spike(recent_history, current):
            indicators.append({
                'type': 'inventory_spike',
                'timestamp': current.get('timestamp', datetime.now()),
                'risk_score': 0.6,
                'severity': 'medium',
                'detail': 'Sudden inventory increase - possible exit scam preparation'
            })
        
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
        curr_inventory = current.get('inventory_count', 0)
        
        if recent_history and curr_inventory > 0:
            avg_inventory = sum(h.get('inventory_count', 0) for h in recent_history) / len(recent_history)
            
            if avg_inventory > 0 and curr_inventory > avg_inventory * 3:
                return True
        
        return False
    
    def _has_review_decline(self, recent_history: List[Dict]) -> bool:
        if len(recent_history) < 3:
            return False
        
        ratings = [h.get('average_rating', 0) for h in recent_history if h.get('average_rating')]
        
        if len(ratings) >= 3:
            recent_avg = sum(ratings[-3:]) / 3
            older_avg = sum(ratings[:-3]) / len(ratings[:-3]) if len(ratings) > 3 else 0
            
            if older_avg > 0 and recent_avg < older_avg * 0.8:
                return True
        
        return False
    
    def _build_risk_timeline(self, historical_data: List[Dict], 
                           indicators: List[Dict]) -> List[Dict]:
        timeline = []
        
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
        
        for indicator in indicators:
            timeline.append({
                'timestamp': indicator.get('timestamp', datetime.now()),
                'event': indicator['type'],
                'risk_level': indicator['risk_score']
            })
        
        timeline.sort(key=lambda x: x['timestamp'])
        
        return timeline
    
    def _identify_changes(self, prev: Dict, curr: Dict) -> List[str]:
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
        pass
    
    def load_model(self):
        pass
