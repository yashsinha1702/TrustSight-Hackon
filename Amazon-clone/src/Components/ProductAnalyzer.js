// Fixed ProductAnalyzer.js - Real-time analysis with backend integration
import React, { useState, useEffect } from 'react';
import trustSightService from '../services/trustSightService';
import './productAnalyzer.css';

const ProductAnalyzer = ({ product, seller, reviews, onAnalysisComplete }) => {
  const [analysisState, setAnalysisState] = useState('idle');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [completedSteps, setCompletedSteps] = useState([]);

  const analysisSteps = [
    { name: 'counterfeit', label: 'Checking for counterfeits', duration: 2000 },
    { name: 'seller', label: 'Analyzing seller reputation', duration: 1500 },
    { name: 'listing', label: 'Examining listing patterns', duration: 1000 },
    { name: 'reviews', label: 'Detecting fake reviews', duration: 2500 },
    { name: 'network', label: 'Tracing fraud networks', duration: 3000 },
    { name: 'final', label: 'Generating recommendations', duration: 1000 }
  ];

  const startAnalysis = async () => {
    setAnalysisState('analyzing');
    setProgress(0);
    setCompletedSteps([]);
    
    // Start the real backend analysis
    const analysisPromise = trustSightService.analyzeProduct(product, seller, reviews);
    
    // Simulate step-by-step progress
    for (let i = 0; i < analysisSteps.length; i++) {
      const step = analysisSteps[i];
      setCurrentStep(step.label);
      
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, step.duration));
      
      setCompletedSteps(prev => [...prev, step.name]);
      setProgress(((i + 1) / analysisSteps.length) * 100);
    }

    try {
      // Wait for the actual analysis result
      const result = await analysisPromise;
      console.log('Analysis complete:', result);
      
      setAnalysisResult(result);
      setAnalysisState('complete');
      
      // Small delay before calling completion handler
      setTimeout(() => {
        if (onAnalysisComplete) {
          onAnalysisComplete(result);
        }
      }, 1000);
      
    } catch (error) {
      console.error('Analysis failed:', error);
      setAnalysisState('error');
    }
  };

  const getTrustScoreColor = (score) => {
    if (score >= 75) return '#00a862';
    if (score >= 50) return '#ff9900';
    return '#d13212';
  };

  const getSeverityBadge = (severity) => {
    const colors = {
      'CRITICAL': '#d13212',
      'HIGH': '#ff5722',
      'MEDIUM': '#ff9900',
      'LOW': '#4caf50'
    };
    
    return (
      <span 
        className="severity-badge"
        style={{ backgroundColor: colors[severity] || '#666' }}
      >
        {severity}
      </span>
    );
  };

  const renderAnalysisDetails = () => {
    if (!analysisResult) return null;

    return (
      <div className="analysis-details">
        {/* Trust Score Display */}
        <div className="trust-score-result">
          <h3>Trust Score Analysis Complete</h3>
          <div className="score-display" style={{ borderColor: getTrustScoreColor(analysisResult.trustScore) }}>
            <div className="score-number">{analysisResult.trustScore}%</div>
            <div className="score-status">{analysisResult.verificationStatus}</div>
          </div>
          {analysisResult.backendAnalyzed && (
            <div className="analysis-method">
              🤖 AI Pipeline Analysis
            </div>
          )}
        </div>

        {/* Fraud Indicators */}
        {analysisResult.fraudIndicators && analysisResult.fraudIndicators.length > 0 && (
          <div className="indicators-section">
            <h4>🚨 Fraud Indicators Detected</h4>
            {analysisResult.fraudIndicators.map((indicator, index) => (
              <div key={index} className="indicator-item">
                {getSeverityBadge(indicator.severity)}
                <div className="indicator-content">
                  <strong>{indicator.type.replace(/_/g, ' ')}</strong>
                  <p>{indicator.description}</p>
                  {indicator.confidence && (
                    <small>Confidence: {Math.round(indicator.confidence * 100)}%</small>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Risk Factors */}
        {analysisResult.riskFactors && analysisResult.riskFactors.length > 0 && (
          <div className="risk-factors-section">
            <h4>⚠️ Risk Factors</h4>
            <ul>
              {analysisResult.riskFactors.map((factor, index) => (
                <li key={index}>{factor}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Network Analysis */}
        {analysisResult.network_analysis && (
          <div className="section network-analysis">
            <h4>🕸️ Fraud Network Analysis</h4>
            <div className="network-stats">
              <div className="network-stat">
                <span className="stat-label">Network Size:</span>
                <span className="stat-value">{analysisResult.network_analysis.graph_summary?.nodes || 0} entities</span>
              </div>
              <div className="network-stat">
                <span className="stat-label">Confidence:</span>
                <span className="stat-value">{Math.round((analysisResult.network_analysis.confidence_score || 0) * 100)}%</span>
              </div>
              <div className="network-stat">
                <span className="stat-label">Financial Impact:</span>
                <span className="stat-value">${(analysisResult.network_analysis.financial_impact || 0).toLocaleString()}</span>
              </div>
            </div>
          </div>
        )}

        {/* Detection Details */}
        {analysisResult.detections && (
          <div className="detection-details">
            <h4>🔍 Detection Results</h4>
            {analysisResult.detections.counterfeit && (
              <div className="detection-item">
                <h5>Counterfeit Detection</h5>
                <p className="detection-score">
                  Score: {Math.round((analysisResult.detections.counterfeit.confidence || 0) * 100)}%
                  <span className="confidence"> (Verdict: {analysisResult.detections.counterfeit.verdict})</span>
                </p>
              </div>
            )}
            {analysisResult.detections.seller && (
              <div className="detection-item">
                <h5>Seller Analysis</h5>
                <p className="detection-score">
                  Risk Score: {Math.round((analysisResult.detections.seller.risk_score || 0) * 100)}%
                  <span className="confidence"> (Level: {analysisResult.detections.seller.risk_level})</span>
                </p>
                {analysisResult.detections.seller.flags && analysisResult.detections.seller.flags.length > 0 && (
                  <p>Flags: {analysisResult.detections.seller.flags.join(', ')}</p>
                )}
              </div>
            )}
            {analysisResult.detections.review && (
              <div className="detection-item">
                <h5>Review Authenticity</h5>
                <p className="detection-score">
                  Fake Reviews: {Math.round(analysisResult.detections.review.fake_percentage || 0)}%
                  <span className="confidence"> ({analysisResult.detections.review.fake_count || 0} of {analysisResult.detections.review.total_reviews || 0})</span>
                </p>
              </div>
            )}
          </div>
        )}

        {/* Actions Taken */}
        {analysisResult.actions_taken && analysisResult.actions_taken.length > 0 && (
          <div className="actions-taken">
            <h4>⚡ Actions Taken</h4>
            <div className="actions-list">
              {analysisResult.actions_taken.map((action, index) => (
                <div key={index} className="action-item">
                  {action.action_type}: {action.description}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  if (analysisState === 'idle') {
    return (
      <div className="analysis-trigger">
        <button 
          className="analyze-button"
          onClick={startAnalysis}
        >
          🛡️ Run TrustSight Analysis
        </button>
        <p className="analysis-description">
          Get comprehensive fraud detection analysis including counterfeit detection, 
          seller verification, review authenticity, and fraud network mapping.
        </p>
      </div>
    );
  }

  if (analysisState === 'analyzing') {
    return (
      <div className="analysis-progress">
        <div className="progress-header">
          <h3>Analyzing Product...</h3>
          <span className="progress-percentage">{Math.round(progress)}%</span>
        </div>
        
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%` }}
          />
        </div>
        
        <p className="current-step">{currentStep}</p>
        
        <div className="analysis-steps">
          {analysisSteps.map((step, index) => (
            <div 
              key={step.name} 
              className={`step ${completedSteps.includes(step.name) ? 'completed' : ''}`}
            >
              <div className="step-indicator" />
              <span className="step-label">{step.label}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (analysisState === 'error') {
    return (
      <div className="analysis-error">
        <h3>Analysis Failed</h3>
        <p>Unable to complete the analysis. Please try again.</p>
        <button onClick={startAnalysis}>Retry Analysis</button>
      </div>
    );
  }

  if (analysisState === 'complete' && analysisResult) {
    return (
      <div className="analysis-complete">
        <h2>✅ TrustSight Analysis Complete</h2>
        {renderAnalysisDetails()}
        <div className="analysis-actions">
          <button 
            className="action-btn report"
            onClick={() => console.log('Report fraud')}
          >
            Report Fraud
          </button>
          <button 
            className="action-btn alternatives"
            onClick={() => console.log('Find alternatives')}
          >
            Find Safe Alternatives
          </button>
          <button 
            className="action-btn reanalyze"
            onClick={startAnalysis}
          >
            Run Again
          </button>
        </div>
      </div>
    );
  }

  return null;
};

export default ProductAnalyzer;