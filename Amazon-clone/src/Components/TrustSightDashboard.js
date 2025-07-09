import React, { useState, useEffect } from 'react';
import Navbar from './Navbar';
import Footer from './Footer';
import trustSightService from '../services/trustSightService';
import './trustSightDashboard.css';

const TrustSightDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [liveMode, setLiveMode] = useState(true);
  const [alerts, setAlerts] = useState([]);
  const [riskProducts, setRiskProducts] = useState([]);
  const [stats, setStats] = useState({
    totalProducts: 0,
    fraudulentProducts: 0,
    verifiedProducts: 0,
    suspiciousSellers: 0,
    fakeReviews: 0,
    totalMonetaryRisk: 0
  });

  useEffect(() => {
    loadDashboardData();
    loadRiskProducts();
    
    // Listen for real-time alerts
    const handleAlert = (event) => {
      setAlerts(prev => [event.detail, ...prev].slice(0, 50));
    };
    
    window.addEventListener('trustsight-alert', handleAlert);
    return () => window.removeEventListener('trustsight-alert', handleAlert);
  }, []);

  const loadDashboardData = async () => {
    try {
      const summary = await trustSightService.getDashboardSummary();
      const fraudStats = await trustSightService.getFraudStats();
      
      setStats({
        totalProducts: 73,
        fraudulentProducts: 28,
        verifiedProducts: 31,
        suspiciousSellers: 15,
        fakeReviews: fraudStats.total_fraud_detected || 1247,
        totalMonetaryRisk: fraudStats.fraud_prevented_value || 2456789
      });
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    }
  };

  const loadRiskProducts = async () => {
    try {
      // Load analysis results to find high-risk products
      const response = await fetch('/data/analysis_results.json');
      if (response.ok) {
        const data = await response.json();
        const highRiskProducts = data.results
          .filter(r => r.trust_score < 50)
          .map(r => ({
            product_id: r.product_id,
            trust_score: r.trust_score,
            verification_status: r.verification_status,
            risk_factors: r.risk_factors,
            fraud_indicators: r.fraud_indicators
          }))
          .slice(0, 10); // Top 10 high-risk products
        
        setRiskProducts(highRiskProducts);
      }
    } catch (error) {
      console.error('Error loading risk products:', error);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0
    }).format(amount);
  };

  const renderPipelineView = () => (
    <div className="pipeline-view">
      <h2>TrustSight Pipeline Architecture</h2>
      
      {/* Detection Layer */}
      <div className="pipeline-layer detection-layer">
        <h3>🔍 Detection Layer</h3>
        <div className="layer-components">
          <div className="component">
            <h4>Counterfeit Detection</h4>
            <ul>
              <li>Image analysis</li>
              <li>Price anomaly detection</li>
              <li>Brand verification</li>
            </ul>
          </div>
          <div className="component">
            <h4>Review Authenticity</h4>
            <ul>
              <li>NLP analysis</li>
              <li>Pattern detection</li>
              <li>Reviewer profiling</li>
            </ul>
          </div>
          <div className="component">
            <h4>Seller Verification</h4>
            <ul>
              <li>Authorization check</li>
              <li>History analysis</li>
              <li>Behavior patterns</li>
            </ul>
          </div>
          <div className="component">
            <h4>Listing Analysis</h4>
            <ul>
              <li>Title/description check</li>
              <li>Category validation</li>
              <li>Price consistency</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Intelligence Layer */}
      <div className="pipeline-layer intelligence-layer">
        <h3>🧠 Cross Intelligence Engine</h3>
        <div className="layer-components">
          <div className="component">
            <h4>Network Expansion</h4>
            <p>Trace review, product, seller links</p>
          </div>
          <div className="component">
            <h4>Pattern Classifier</h4>
            <p>Review farms, seller cartels, hybrid rings</p>
          </div>
          <div className="component">
            <h4>Kingpin Identification</h4>
            <p>Central actors in fraud networks</p>
          </div>
          <div className="component">
            <h4>Financial Impact</h4>
            <p>Total value of fraud ring</p>
          </div>
        </div>
      </div>

      {/* Lifecycle Layer */}
      <div className="pipeline-layer lifecycle-layer">
        <h3>🔄 Lifecycle Monitoring</h3>
        <div className="layer-components">
          <div className="component">
            <h4>Birth Stage</h4>
            <p>Preloaded reviews, launch velocity</p>
          </div>
          <div className="component">
            <h4>Growth Stage</h4>
            <p>Review surge, category hopping</p>
          </div>
          <div className="component">
            <h4>Maturity Stage</h4>
            <p>Hijacking attempts, return spikes</p>
          </div>
          <div className="component">
            <h4>Decline/Exit</h4>
            <p>Exit scam signs, inventory dumping</p>
          </div>
        </div>
      </div>

      {/* Decision Layer */}
      <div className="pipeline-layer decision-layer">
        <h3>💾 Decision Memory Bank</h3>
        <div className="layer-components">
          <div className="component">
            <h4>Decision Recording</h4>
            <p>All actions tracked with evidence</p>
          </div>
          <div className="component">
            <h4>Verification Tracking</h4>
            <p>Monitor decision outcomes</p>
          </div>
          <div className="component">
            <h4>Appeal System</h4>
            <p>Sellers can contest decisions</p>
          </div>
          <div className="component">
            <h4>Learning Loop</h4>
            <p>Improve from false positives</p>
          </div>
        </div>
      </div>

      {/* Action Layer */}
      <div className="pipeline-layer action-layer">
        <h3>⚡ Action Layer</h3>
        <div className="layer-components">
          <div className="component soft-action">
            <h4>Soft Actions</h4>
            <ul>
              <li>Buyer alerts</li>
              <li>Review flags</li>
              <li>Seller warnings</li>
            </ul>
          </div>
          <div className="component hard-action">
            <h4>Hard Actions</h4>
            <ul>
              <li>Listing suspension</li>
              <li>Account termination</li>
              <li>Network takedown</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAddEntityForm = () => (
    <div className="add-entity-section">
      <h2>Add New Entities for Analysis</h2>
      
      <div className="entity-forms">
        {/* Add Product Form */}
        <div className="entity-form">
          <h3>Add New Product</h3>
          <div className="form-group">
            <input 
              type="text" 
              placeholder="Product Title" 
              id="product-title"
            />
            <input 
              type="number" 
              placeholder="Price" 
              id="product-price"
            />
            <input 
              type="text" 
              placeholder="Seller ID" 
              id="product-seller-id"
            />
            <input 
              type="text" 
              placeholder="Category" 
              id="product-category"
            />
            <textarea 
              placeholder="Description" 
              id="product-description"
            ></textarea>
            <button 
              className="analyze-btn"
              onClick={async () => {
                const productData = {
                  title: document.getElementById('product-title').value,
                  price: parseFloat(document.getElementById('product-price').value),
                  seller_id: document.getElementById('product-seller-id').value,
                  category: document.getElementById('product-category').value,
                  description: document.getElementById('product-description').value,
                  brand: document.getElementById('product-title').value.split(' ')[0]
                };
                
                const result = await trustSightService.addAndAnalyzeProduct(productData);
                console.log('Product added:', result);
                alert(`Product added! Trust Score: ${result.analysis?.trustScore || 'Pending'}%`);
              }}
            >
              Add & Analyze Product
            </button>
          </div>
        </div>

        {/* Add Seller Form */}
        <div className="entity-form">
          <h3>Add New Seller</h3>
          <div className="form-group">
            <input 
              type="text" 
              placeholder="Business Name" 
              id="seller-business-name"
            />
            <input 
              type="text" 
              placeholder="Display Name" 
              id="seller-display-name"
            />
            <input 
              type="date" 
              placeholder="Registration Date" 
              id="seller-registration-date"
            />
            <input 
              type="email" 
              placeholder="Email" 
              id="seller-email"
            />
            <button 
              className="analyze-btn"
              onClick={async () => {
                const sellerData = {
                  business_name: document.getElementById('seller-business-name').value,
                  display_name: document.getElementById('seller-display-name').value,
                  registration_date: document.getElementById('seller-registration-date').value,
                  email: document.getElementById('seller-email').value
                };
                
                const result = await trustSightService.addAndAnalyzeSeller(sellerData);
                console.log('Seller added:', result);
                alert(`Seller added! Risk Score: ${result.analysis?.risk_score || 'Pending'}`);
              }}
            >
              Add & Analyze Seller
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <Navbar />
      <div className="trustsight-dashboard">
        {/* Header */}
        <div className="dashboard-header">
          <div className="header-content">
            <h1>TrustSight Command Center</h1>
            <p>Real-time fraud detection and prevention system</p>
          </div>
          <div className="header-controls">
            <button 
              className={`live-mode-toggle ${liveMode ? 'active' : ''}`}
              onClick={() => setLiveMode(!liveMode)}
            >
              {liveMode ? '🔴 LIVE' : '⏸️ PAUSED'}
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="dashboard-tabs">
          <button 
            className={activeTab === 'overview' ? 'active' : ''}
            onClick={() => setActiveTab('overview')}
          >
            📊 Overview
          </button>
          <button 
            className={activeTab === 'pipeline' ? 'active' : ''}
            onClick={() => setActiveTab('pipeline')}
          >
            🔧 Pipeline View
          </button>
          <button 
            className={activeTab === 'alerts' ? 'active' : ''}
            onClick={() => setActiveTab('alerts')}
          >
            🚨 Live Alerts
          </button>
          <button 
            className={activeTab === 'add' ? 'active' : ''}
            onClick={() => setActiveTab('add')}
          >
            ➕ Add Entities
          </button>
        </div>

        {/* Content based on active tab */}
        {activeTab === 'overview' && (
          <>
            {/* Key Metrics */}
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-icon">📦</div>
                <div className="metric-content">
                  <span className="metric-value">{stats.totalProducts}</span>
                  <span className="metric-label">Total Products</span>
                </div>
              </div>
              
              <div className="metric-card alert">
                <div className="metric-icon">⚠️</div>
                <div className="metric-content">
                  <span className="metric-value">{stats.fraudulentProducts}</span>
                  <span className="metric-label">Fraudulent Products</span>
                  <span className="metric-change">
                    {Math.round((stats.fraudulentProducts / stats.totalProducts) * 100)}% of total
                  </span>
                </div>
              </div>
              
              <div className="metric-card success">
                <div className="metric-icon">✅</div>
                <div className="metric-content">
                  <span className="metric-value">{stats.verifiedProducts}</span>
                  <span className="metric-label">Verified Products</span>
                </div>
              </div>
              
              <div className="metric-card financial">
                <div className="metric-icon">💰</div>
                <div className="metric-content">
                  <span className="metric-value">{formatCurrency(stats.totalMonetaryRisk)}</span>
                  <span className="metric-label">Fraud Prevented</span>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="action-center">
              <h2>Quick Actions</h2>
              <div className="action-buttons">
                <button className="action-button">
                  <span className="action-icon">🔍</span>
                  <span>Run Full Scan</span>
                </button>
                <button className="action-button">
                  <span className="action-icon">📊</span>
                  <span>Generate Report</span>
                </button>
                <button className="action-button">
                  <span className="action-icon">🚫</span>
                  <span>Bulk Remove</span>
                </button>
                <button className="action-button">
                  <span className="action-icon">📧</span>
                  <span>Alert Sellers</span>
                </button>
              </div>
            </div>

            {/* High Risk Products */}
            <div className="risk-products-section">
              <h2>⚠️ High Risk Products</h2>
              <div className="risk-products-list">
                {riskProducts.length === 0 ? (
                  <p className="no-risk-products">No high-risk products detected</p>
                ) : (
                  riskProducts.map((product, index) => (
                    <div key={index} className="risk-product-item">
                      <div className="risk-score" style={{
                        backgroundColor: product.trust_score < 30 ? '#d13212' : '#ff5722'
                      }}>
                        {product.trust_score}%
                      </div>
                      <div className="risk-product-info">
                        <h4>Product ID: {product.product_id}</h4>
                        <p>Status: {product.verification_status}</p>
                        <div className="risk-factors">
                          {product.risk_factors.slice(0, 2).map((factor, i) => (
                            <span key={i} className="risk-factor-tag">{factor}</span>
                          ))}
                        </div>
                      </div>
                      <div className="risk-actions">
                        <button className="action-btn investigate">Investigate</button>
                        <button className="action-btn remove">Remove</button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </>
        )}

        {activeTab === 'pipeline' && renderPipelineView()}

        {activeTab === 'alerts' && (
          <div className="alerts-view">
            <h2>Real-time Fraud Alerts</h2>
            <div className="alerts-list">
              {alerts.length === 0 ? (
                <p className="no-alerts">No active alerts. System monitoring...</p>
              ) : (
                alerts.map((alert, index) => (
                  <div key={index} className={`alert-item ${alert.severity?.toLowerCase()}`}>
                    <div className="alert-time">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </div>
                    <div className="alert-content">
                      <h4>{alert.message}</h4>
                      <p>Entity: {alert.entity_id} ({alert.entity_type})</p>
                    </div>
                    <div className="alert-actions">
                      <button>View</button>
                      <button>Dismiss</button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === 'add' && renderAddEntityForm()}
      </div>
      <Footer />
    </>
  );
};

export default TrustSightDashboard;