import asyncio
import json
import logging
import os
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from dataclasses import asdict
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import MemoryBankConfig
from .enums import DecisionStatus, VerificationMethod, StorageTier
from .models import Decision, VerificationSignal, Appeal
from .database_models import Base, DecisionRecord

logger = logging.getLogger(__name__)

class FraudDecisionMemoryBank:
    """Core memory bank for tracking, verifying, and learning from all fraud decisions"""
    
    def __init__(self, config: MemoryBankConfig = None):
        self.config = config or MemoryBankConfig()
        
        self._initialize_storage()
        
        self.verification_queue = asyncio.Queue()
        self.appeal_queue = asyncio.Queue()
        
        self.learning_queue = asyncio.Queue()
        self.model_versions = deque(maxlen=self.config.MODEL_VERSION_RETENTION)
        
        self.metrics = {
            'total_decisions': 0,
            'verified_decisions': 0,
            'false_positives': 0,
            'true_positives': 0,
            'appeals_received': 0,
            'appeals_upheld': 0
        }
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        self.scheduler = AsyncIOScheduler()
        self._setup_scheduled_tasks()
        
        logger.info("Fraud Decision Memory Bank initialized")
    
    def _initialize_storage(self):
        engine = create_engine(
            self.config.DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        Base.metadata.create_all(bind=engine)
        self.SessionLocal = sessionmaker(bind=engine)
        
        self.evidence_storage_path = "./evidence_storage"
        os.makedirs(self.evidence_storage_path, exist_ok=True)
        
        self.archive_path = "./archive_storage"
        os.makedirs(self.archive_path, exist_ok=True)
    
    def _setup_scheduled_tasks(self):
        self.scheduler.add_job(
            self.process_verification_queue,
            trigger=CronTrigger(minute='*/5'),
            id='verification_processor'
        )
        
        self.scheduler.add_job(
            self.process_expired_decisions,
            trigger=CronTrigger(hour='*/1'),
            id='ttl_processor'
        )
        
        self.scheduler.add_job(
            self.migrate_storage_tiers,
            trigger=CronTrigger(hour='2'),
            id='storage_migration'
        )
        
        self.scheduler.add_job(
            self.process_learning_pipeline,
            trigger=CronTrigger(hour='4'),
            id='learning_pipeline'
        )
        
        self.scheduler.start()
    
    async def record_decision(self, fraud_signal: Dict, detection_results: Dict) -> str:
        decision = Decision(
            detection_type=fraud_signal.get('signal_type', ''),
            entity_type=fraud_signal.get('entity_type', ''),
            entity_id=fraud_signal.get('entity_id', ''),
            fraud_score=detection_results.get('overall_fraud_score', 0.0),
            confidence_score=detection_results.get('confidence', 0.0),
            evidence={
                'primary_indicators': detection_results.get('primary_evidence', []),
                'supporting_signals': detection_results.get('supporting_signals', []),
                'network_connections': detection_results.get('network_map', {}),
                'model_versions': detection_results.get('model_versions', {})
            },
            action_taken=detection_results.get('action', '')
        )
        
        decision.ttl_days = decision.calculate_ttl(self.config)
        
        db = self.SessionLocal()
        try:
            db_record = DecisionRecord(
                decision_id=decision.decision_id,
                timestamp=decision.timestamp,
                detection_type=decision.detection_type,
                entity_type=decision.entity_type,
                entity_id=decision.entity_id,
                fraud_score=decision.fraud_score,
                confidence_score=decision.confidence_score,
                primary_evidence=decision.evidence.get('primary_indicators', []),
                supporting_signals=decision.evidence.get('supporting_signals', []),
                network_map=decision.evidence.get('network_connections', {}),
                model_versions=decision.evidence.get('model_versions', {}),
                status=decision.status.value,
                action_taken=decision.action_taken,
                ttl=decision.ttl_days,
                expires_at=datetime.utcnow() + timedelta(days=decision.ttl_days)
            )
            
            db.add(db_record)
            db.commit()
            
            await self._store_evidence(decision.decision_id, decision.evidence)
            
            self.metrics['total_decisions'] += 1
            
            if decision.confidence_score < 0.95:
                await self.verification_queue.put((decision.confidence_score, decision.decision_id))
            
            logger.info(f"Recorded decision {decision.decision_id} with TTL {decision.ttl_days} days")
            
            return decision.decision_id
            
        except Exception as e:
            logger.error(f"Error recording decision: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    async def _store_evidence(self, decision_id: str, evidence: Dict):
        evidence_path = os.path.join(self.evidence_storage_path, decision_id)
        os.makedirs(evidence_path, exist_ok=True)
        
        with open(os.path.join(evidence_path, 'evidence.json'), 'w') as f:
            json.dump(evidence, f, indent=2, default=str)
        
        if 'binary_evidence' in evidence:
            for filename, data in evidence['binary_evidence'].items():
                with open(os.path.join(evidence_path, filename), 'wb') as f:
                    f.write(data)
    
    async def submit_appeal(self, appeal: Appeal) -> str:
        db = self.SessionLocal()
        try:
            decision = db.query(DecisionRecord).filter_by(
                decision_id=appeal.decision_id
            ).first()
            
            if not decision:
                raise ValueError(f"Decision {appeal.decision_id} not found")
            
            hours_since_decision = (datetime.utcnow() - decision.timestamp).total_seconds() / 3600
            if hours_since_decision > self.config.APPEAL_WINDOW:
                raise ValueError(f"Appeal window expired ({self.config.APPEAL_WINDOW} hours)")
            
            decision.status = DecisionStatus.UNDER_APPEAL.value
            decision.appeal_submitted = True
            decision.appeal_timestamp = datetime.utcnow()
            decision.appeal_details = {
                'appeal_id': appeal.appeal_id,
                'seller_id': appeal.seller_id,
                'reason': appeal.reason,
                'evidence': appeal.evidence
            }
            
            db.commit()
            
            await self.verification_queue.put((1.0, appeal.decision_id))
            
            self.metrics['appeals_received'] += 1
            
            logger.info(f"Appeal {appeal.appeal_id} submitted for decision {appeal.decision_id}")
            
            return appeal.appeal_id
            
        except Exception as e:
            logger.error(f"Error submitting appeal: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    async def process_verification_queue(self):
        batch_size = 50
        decisions_to_verify = []
        
        for _ in range(batch_size):
            try:
                priority, decision_id = await asyncio.wait_for(
                    self.verification_queue.get(), 
                    timeout=0.1
                )
                decisions_to_verify.append(decision_id)
            except asyncio.TimeoutError:
                break
        
        if not decisions_to_verify:
            return
        
        logger.info(f"Processing {len(decisions_to_verify)} decisions for verification")
        
        for decision_id in decisions_to_verify:
            try:
                await self.verify_decision(decision_id)
            except Exception as e:
                logger.error(f"Error verifying decision {decision_id}: {e}")
    
    async def verify_decision(self, decision_id: str) -> Dict:
        db = self.SessionLocal()
        try:
            decision = db.query(DecisionRecord).filter_by(decision_id=decision_id).first()
            if not decision:
                raise ValueError(f"Decision {decision_id} not found")
            
            verification_signals = []
            
            if decision.appeal_submitted:
                appeal_result = await self._process_seller_appeal(decision)
                verification_signals.append(appeal_result)
            
            customer_reports = await self._check_customer_reports(decision)
            if customer_reports:
                verification_signals.append(customer_reports)
            
            behavioral_check = await self._verify_through_behavior(decision)
            if behavioral_check:
                verification_signals.append(behavioral_check)
            
            if (decision.expires_at - datetime.utcnow()).days <= 2:
                time_based = await self._time_based_verification(decision)
                verification_signals.append(time_based)
            
            final_verdict = self._calculate_verdict(verification_signals)
            
            decision.status = final_verdict['status']
            decision.verification_result = final_verdict['result']
            decision.verification_timestamp = datetime.utcnow()
            decision.verification_details = {
                'signals': [asdict(s) for s in verification_signals],
                'verdict': final_verdict
            }
            
            db.commit()
            
            if final_verdict['status'] == DecisionStatus.CONFIRMED_FRAUD.value:
                await self._process_confirmed_fraud(decision)
            elif final_verdict['status'] == DecisionStatus.FALSE_POSITIVE.value:
                await self._process_false_positive(decision)
            
            self.metrics['verified_decisions'] += 1
            if final_verdict['status'] == DecisionStatus.CONFIRMED_FRAUD.value:
                self.metrics['true_positives'] += 1
            elif final_verdict['status'] == DecisionStatus.FALSE_POSITIVE.value:
                self.metrics['false_positives'] += 1
            
            logger.info(f"Decision {decision_id} verified: {final_verdict['result']}")
            
            return final_verdict
            
        except Exception as e:
            logger.error(f"Error in verification: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    async def _process_seller_appeal(self, decision: DecisionRecord) -> VerificationSignal:
        appeal_details = decision.appeal_details
        
        evidence_valid = await self._validate_appeal_evidence(appeal_details.get('evidence', {}))
        
        seller_history = await self._get_seller_history(appeal_details.get('seller_id'))
        seller_credibility = self._calculate_seller_credibility(seller_history)
        
        if evidence_valid and seller_credibility > 0.7:
            result = "appeal_upheld"
            confidence = 0.9
        else:
            result = "appeal_rejected"
            confidence = 0.8
        
        return VerificationSignal(
            method=VerificationMethod.SELLER_APPEAL,
            result=result,
            confidence=confidence,
            details={
                'evidence_valid': evidence_valid,
                'seller_credibility': seller_credibility,
                'appeal_reason': appeal_details.get('reason', '')
            }
        )
    
    async def _check_customer_reports(self, decision: DecisionRecord) -> Optional[VerificationSignal]:
        entity_id = decision.entity_id
        
        complaints = await self._query_customer_complaints(entity_id)
        
        if len(complaints) >= 3:
            return VerificationSignal(
                method=VerificationMethod.CUSTOMER_REPORTS,
                result="fraud_confirmed",
                confidence=0.85,
                details={
                    'complaint_count': len(complaints),
                    'complaint_types': [c['type'] for c in complaints]
                }
            )
        elif len(complaints) == 0:
            return VerificationSignal(
                method=VerificationMethod.CUSTOMER_REPORTS,
                result="no_complaints",
                confidence=0.6,
                details={'complaint_count': 0}
            )
        
        return None
    
    async def _verify_through_behavior(self, decision: DecisionRecord) -> Optional[VerificationSignal]:
        if decision.entity_type == 'seller':
            seller_activity = await self._check_seller_activity(decision.entity_id, decision.timestamp)
            
            if seller_activity['abandoned']:
                return VerificationSignal(
                    method=VerificationMethod.BEHAVIORAL_VERIFICATION,
                    result="fraud_likely",
                    confidence=0.75,
                    details={'behavior': 'account_abandoned_after_detection'}
                )
            elif seller_activity['increased_activity']:
                return VerificationSignal(
                    method=VerificationMethod.BEHAVIORAL_VERIFICATION,
                    result="legitimate_likely",
                    confidence=0.65,
                    details={'behavior': 'continued_normal_operations'}
                )
        
        return None
    
    async def _time_based_verification(self, decision: DecisionRecord) -> VerificationSignal:
        return VerificationSignal(
            method=VerificationMethod.TIME_BASED,
            result="no_dispute_received",
            confidence=0.6,
            details={
                'days_elapsed': (datetime.utcnow() - decision.timestamp).days,
                'ttl': decision.ttl
            }
        )
    
    def _calculate_verdict(self, signals: List[VerificationSignal]) -> Dict:
        if not signals:
            return {
                'status': DecisionStatus.PENDING_VERIFICATION.value,
                'result': 'insufficient_signals',
                'confidence': 0.0
            }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        fraud_indicators = 0
        legitimate_indicators = 0
        
        for signal in signals:
            weight = self.config.VERIFICATION_WEIGHTS.get(signal.method.value, 0.5)
            
            if 'fraud' in signal.result or 'rejected' in signal.result:
                weighted_score += signal.confidence * weight
                fraud_indicators += 1
            elif 'legitimate' in signal.result or 'upheld' in signal.result:
                weighted_score -= signal.confidence * weight
                legitimate_indicators += 1
            
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0
        
        if final_score > self.config.VERIFICATION_THRESHOLD:
            status = DecisionStatus.CONFIRMED_FRAUD.value
            result = 'fraud_confirmed'
        elif final_score < -self.config.VERIFICATION_THRESHOLD:
            status = DecisionStatus.FALSE_POSITIVE.value
            result = 'false_positive_confirmed'
        elif legitimate_indicators > fraud_indicators:
            status = DecisionStatus.FALSE_POSITIVE.value
            result = 'likely_false_positive'
        else:
            status = DecisionStatus.CONFIRMED_FRAUD.value
            result = 'likely_fraud'
        
        appeal_signal = next((s for s in signals if s.method == VerificationMethod.SELLER_APPEAL), None)
        if appeal_signal and appeal_signal.result == 'appeal_upheld':
            status = DecisionStatus.APPEAL_UPHELD.value
            result = 'appeal_successful'
        
        return {
            'status': status,
            'result': result,
            'confidence': abs(final_score),
            'fraud_indicators': fraud_indicators,
            'legitimate_indicators': legitimate_indicators
        }
    
    async def _process_confirmed_fraud(self, decision: DecisionRecord):
        await self.learning_queue.put({
            'decision_id': decision.decision_id,
            'label': 'fraud',
            'confidence': decision.verification_details['verdict']['confidence']
        })
        
        await self._strengthen_similar_detections(decision)
        
        logger.info(f"Processed confirmed fraud: {decision.decision_id}")
    
    async def _process_false_positive(self, decision: DecisionRecord):
        await self.learning_queue.put({
            'decision_id': decision.decision_id,
            'label': 'legitimate',
            'confidence': 1.0
        })
        
        await self._remediate_false_positive(decision)
        
        await self._reduce_similar_detections(decision)
        
        root_cause = await self._analyze_false_positive_cause(decision)
        
        if await self._detect_systematic_false_positives(root_cause):
            await self._adjust_detection_thresholds(root_cause)
        
        if decision.appeal_submitted:
            decision.appeal_outcome = 'upheld'
            self.metrics['appeals_upheld'] += 1
        
        logger.info(f"Processed false positive: {decision.decision_id}")
    
    async def _remediate_false_positive(self, decision: DecisionRecord):
        remediation_actions = {
            'restore_listing': decision.entity_type in ['product', 'listing'],
            'unfreeze_account': decision.entity_type == 'seller',
            'restore_reviews': decision.detection_type == 'review_fraud',
            'send_apology': True,
            'priority_support': True
        }
        
        logger.info(f"Remediation actions for {decision.decision_id}: {remediation_actions}")
    
    async def process_expired_decisions(self):
        db = self.SessionLocal()
        try:
            expired = db.query(DecisionRecord).filter(
                DecisionRecord.expires_at <= datetime.utcnow(),
                DecisionRecord.status == DecisionStatus.PENDING_VERIFICATION.value
            ).limit(100).all()
            
            for decision in expired:
                decision.status = DecisionStatus.EXPIRED_NO_DISPUTE.value
                decision.verification_result = 'auto_confirmed'
                decision.verification_timestamp = datetime.utcnow()
                
                await self.learning_queue.put({
                    'decision_id': decision.decision_id,
                    'label': 'fraud',
                    'confidence': 0.7
                })
            
            db.commit()
            logger.info(f"Processed {len(expired)} expired decisions")
            
        except Exception as e:
            logger.error(f"Error processing expired decisions: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def migrate_storage_tiers(self):
        db = self.SessionLocal()
        try:
            hot_to_warm = db.query(DecisionRecord).filter(
                DecisionRecord.storage_tier == StorageTier.HOT.value,
                DecisionRecord.timestamp < datetime.utcnow() - timedelta(days=30)
            ).limit(1000).all()
            
            for decision in hot_to_warm:
                await self._migrate_to_warm_storage(decision)
                decision.storage_tier = StorageTier.WARM.value
            
            warm_to_cold = db.query(DecisionRecord).filter(
                DecisionRecord.storage_tier == StorageTier.WARM.value,
                DecisionRecord.timestamp < datetime.utcnow() - timedelta(days=730)
            ).limit(100).all()
            
            for decision in warm_to_cold:
                await self._migrate_to_cold_storage(decision)
                decision.storage_tier = StorageTier.COLD.value
            
            db.commit()
            logger.info(f"Migrated {len(hot_to_warm)} to warm, {len(warm_to_cold)} to cold storage")
            
        except Exception as e:
            logger.error(f"Error in storage migration: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def _migrate_to_warm_storage(self, decision: DecisionRecord):
        evidence_path = os.path.join(self.evidence_storage_path, decision.decision_id)
        if os.path.exists(evidence_path):
            archive_path = os.path.join(self.archive_path, f"{decision.decision_id}.tar.gz")
            logger.debug(f"Migrated {decision.decision_id} to warm storage")
    
    async def _migrate_to_cold_storage(self, decision: DecisionRecord):
        logger.debug(f"Migrated {decision.decision_id} to cold storage")
    
    async def process_learning_pipeline(self):
        batch_size = 1000
        training_batch = []
        
        for _ in range(batch_size):
            try:
                decision_data = await asyncio.wait_for(
                    self.learning_queue.get(),
                    timeout=0.1
                )
                training_batch.append(decision_data)
            except asyncio.TimeoutError:
                break
        
        if len(training_batch) < self.config.MIN_VERIFICATIONS_FOR_TRAINING:
            for item in training_batch:
                await self.learning_queue.put(item)
            return
        
        logger.info(f"Processing learning pipeline with {len(training_batch)} decisions")
        
        fraud_examples = [d for d in training_batch if d['label'] == 'fraud']
        legitimate_examples = [d for d in training_batch if d['label'] == 'legitimate']
        
        balanced_batch = self._create_balanced_batch(fraud_examples, legitimate_examples)
        
        training_data = await self._prepare_training_data(balanced_batch)
        
        model_version = await self._version_current_models()
        
        try:
            new_models = await self._train_models(training_data)
            
            if await self._validate_models(new_models, model_version):
                await self._deploy_new_models(new_models)
                logger.info("Successfully deployed updated models")
            else:
                await self._rollback_models(model_version)
                logger.warning("Model validation failed, rolled back")
                
        except Exception as e:
            logger.error(f"Training pipeline error: {e}")
            await self._rollback_models(model_version)
    
    def _create_balanced_batch(self, fraud_examples: List, legitimate_examples: List) -> List:
        min_count = min(len(fraud_examples), len(legitimate_examples))
        
        balanced = []
        balanced.extend(fraud_examples[:min_count])
        balanced.extend(legitimate_examples[:min_count])
        
        np.random.shuffle(balanced)
        
        return balanced
    
    async def _prepare_training_data(self, batch: List) -> Dict:
        db = self.SessionLocal()
        training_data = {
            'features': [],
            'labels': [],
            'metadata': []
        }
        
        try:
            for item in batch:
                decision = db.query(DecisionRecord).filter_by(
                    decision_id=item['decision_id']
                ).first()
                
                if decision:
                    features = await self._extract_features(decision)
                    
                    training_data['features'].append(features)
                    training_data['labels'].append(1 if item['label'] == 'fraud' else 0)
                    training_data['metadata'].append({
                        'decision_id': decision.decision_id,
                        'confidence': item['confidence'],
                        'detection_type': decision.detection_type
                    })
            
            return training_data
            
        finally:
            db.close()
    
    async def _extract_features(self, decision: DecisionRecord) -> np.ndarray:
        features = [
            decision.fraud_score,
            decision.confidence_score,
            len(decision.primary_evidence or []),
            len(decision.supporting_signals or []),
            1 if decision.network_map else 0
        ]
        
        return np.array(features)
    
    async def _version_current_models(self) -> str:
        version_id = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.model_versions.append({
            'version_id': version_id,
            'timestamp': datetime.utcnow(),
            'models': {}
        })
        
        return version_id
    
    async def _train_models(self, training_data: Dict) -> Dict:
        logger.info(f"Training models with {len(training_data['features'])} examples")
        
        await asyncio.sleep(2)
        
        return {
            'review_model': 'updated',
            'seller_model': 'updated',
            'listing_model': 'updated',
            'counterfeit_model': 'updated'
        }
    
    async def _validate_models(self, new_models: Dict, baseline_version: str) -> bool:
        validation_passed = True
        
        return validation_passed
    
    async def _deploy_new_models(self, new_models: Dict):
        logger.info("Deploying new models")
    
    async def _rollback_models(self, version_id: str):
        logger.warning(f"Rolling back to model version {version_id}")
    
    def get_metrics(self) -> Dict:
        db = self.SessionLocal()
        try:
            total_decisions = db.query(DecisionRecord).count()
            
            status_counts = {}
            for status in DecisionStatus:
                count = db.query(DecisionRecord).filter_by(status=status.value).count()
                status_counts[status.value] = count
            
            storage_distribution = {}
            for tier in StorageTier:
                count = db.query(DecisionRecord).filter_by(storage_tier=tier.value).count()
                storage_distribution[tier.value] = count
            
            return {
                'total_decisions': total_decisions,
                'status_distribution': status_counts,
                'storage_distribution': storage_distribution,
                'metrics': self.metrics,
                'queue_sizes': {
                    'verification': self.verification_queue.qsize(),
                    'learning': self.learning_queue.qsize()
                }
            }
            
        finally:
            db.close()
    
    async def _validate_appeal_evidence(self, evidence: Dict) -> bool:
        return bool(evidence) and len(evidence) > 2
    
    async def _get_seller_history(self, seller_id: str) -> Dict:
        return {
            'account_age_days': np.random.randint(30, 1000),
            'total_sales': np.random.randint(100, 10000),
            'previous_violations': np.random.randint(0, 3)
        }
    
    def _calculate_seller_credibility(self, history: Dict) -> float:
        score = 0.5
        
        if history['account_age_days'] > 365:
            score += 0.2
        if history['total_sales'] > 1000:
            score += 0.2
        if history['previous_violations'] == 0:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _query_customer_complaints(self, entity_id: str) -> List[Dict]:
        if np.random.random() > 0.7:
            return [
                {'type': 'quality', 'severity': 'high'},
                {'type': 'authenticity', 'severity': 'high'},
                {'type': 'delivery', 'severity': 'medium'}
            ]
        return []
    
    async def _check_seller_activity(self, seller_id: str, since: datetime) -> Dict:
        return {
            'abandoned': np.random.random() > 0.7,
            'increased_activity': np.random.random() > 0.5
        }
    
    async def _strengthen_similar_detections(self, decision: DecisionRecord):
        logger.info(f"Strengthening similar detections to {decision.decision_id}")
    
    async def _reduce_similar_detections(self, decision: DecisionRecord):
        logger.info(f"Reducing similar detections to {decision.decision_id}")
    
    async def _analyze_false_positive_cause(self, decision: DecisionRecord) -> Dict:
        return {
            'primary_cause': 'threshold_too_low',
            'contributing_factors': ['new_seller', 'unusual_pattern'],
            'pattern_frequency': np.random.randint(1, 10)
        }
    
    async def _detect_systematic_false_positives(self, root_cause: Dict) -> bool:
        return root_cause['pattern_frequency'] > 5
    
    async def _adjust_detection_thresholds(self, root_cause: Dict):
        logger.info(f"Adjusting thresholds based on pattern: {root_cause['primary_cause']}")