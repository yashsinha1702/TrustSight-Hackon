// TrustSightIndicator.js
import React, { useState, useEffect } from 'react';
import trustSightService from '../services/trustSightService';
import './trustSightIndicator.css';

const TrustSightIndicator = ({ productId, sellerId, showDetails = false }) => {
  const [trustData, setTrustData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    loadTrustData();
  }, [productId, sellerId]);

  const loadTrustData = async () => {
    setLoading(true);
    try {
      const productTrust = await trustSightService.getProductTrustScore(productId);
      const sellerRisk = sellerId ? await trustSightService.getSellerRisk(sellerId) : null;
      
      setTrustData({
        productTrust,
        sellerRisk
      });
    } catch (error) {
      console.error('Error loading trust data:', error);
    }
    setLoading(false);
  };

  const getTrustColor = (score) => {
    if (score >= 80) return '#00a862'; // Green
    if (score >= 60) return '#ff9900'; // Orange
    return '#d13212'; // Red
  };

  const getRiskLabel = (score) => {
    if (score >= 80) return 'Verified';
    if (score >= 60) return 'Caution';
    return 'High Risk';
  };

  if (loading) {
    return <div className="trust-loading">Checking...</div>;
  }

  if (!trustData || !trustData.productTrust) {
    return null;
  }

  const { trust_score, fraud_indicators, verification_status } = trustData.productTrust;
  const riskColor = getTrustColor(trust_score);
  const riskLabel = getRiskLabel(trust_score);

  return (
    <div className="trust-indicator-container">
      <div 
        className="trust-badge" 
        style={{ borderColor: riskColor }}
        onClick={() => showDetails && setExpanded(!expanded)}
      >
        <div className="trust-shield" style={{ backgroundColor: riskColor }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
            <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>
          </svg>
        </div>
        <div className="trust-info">
          <span className="trust-label">{riskLabel}</span>
          <span className="trust-score">{trust_score}%</span>
        </div>
      </div>

      {showDetails && expanded && (
        <div className="trust-details">
          <h4>TrustSight Analysis</h4>
          
          {fraud_indicators && fraud_indicators.length > 0 && (
            <div className="fraud-indicators">
              <p className="indicator-title">⚠️ Risk Indicators:</p>
              <ul>
                {fraud_indicators.map((indicator, index) => (
                  <li key={index} className="risk-item">
                    {indicator.type}: {indicator.description}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {trustData.sellerRisk && (
            <div className="seller-info">
              <p className="seller-title">Seller Trust Score:</p>
              <div className="seller-score">
                <div 
                  className="score-bar"
                  style={{
                    width: `${trustData.sellerRisk.trust_score}%`,
                    backgroundColor: getTrustColor(trustData.sellerRisk.trust_score)
                  }}
                />
                <span>{trustData.sellerRisk.trust_score}%</span>
              </div>
              {trustData.sellerRisk.is_authorized && (
                <span className="authorized-badge">✓ Authorized Seller</span>
              )}
            </div>
          )}

          {verification_status && (
            <div className="verification-status">
              <p className="status-title">Verification:</p>
              <span className={`status-badge ${verification_status.toLowerCase()}`}>
                {verification_status}
              </span>
            </div>
          )}

          <button 
            className="report-button"
            onClick={() => handleReport()}
          >
            Report Suspicious Activity
          </button>
        </div>
      )}
    </div>
  );

  async function handleReport() {
    const report = await trustSightService.reportSuspiciousActivity({
      product_id: productId,
      seller_id: sellerId,
      reason: 'User reported suspicious activity'
    });
    
    if (report.success) {
      alert('Thank you for your report. Our team will investigate.');
    }
  }
};

export default TrustSightIndicator;