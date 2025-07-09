// Products.js - Quick Fix Version with Essential Inline Styles
import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import Navbar from "./Navbar";
import Footer from "./Footer";
import "./products.css";

// Import datasets
import productsData from "../data/products.json";
import sellersData from "../data/sellers.json";
import reviewsData from "../data/reviews.json";

function Products() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [showTrustFilter, setShowTrustFilter] = useState(false);

  document.title = "Amazon - Shop with TrustSight Protection";

  useEffect(() => {
    loadProductsWithAnalysis();
  }, []);

  const loadProductsWithAnalysis = async () => {
    console.log("Loading products with pre-computed analysis...");
    
    try {
      // Load the pre-analyzed results
      const analysisResponse = await fetch('/data/analysis_results.json');
      let analysisMap = {};
      
      if (analysisResponse.ok) {
        const analysisData = await analysisResponse.json();
        
        // Create a map of analysis results by product_id
        if (analysisData && analysisData.results) {
          analysisData.results.forEach(result => {
            analysisMap[result.product_id] = result;
          });
        }
      }
      
      // Merge product data with analysis results
      const enhancedProducts = productsData.map(product => {
        const seller = sellersData.find(s => s.seller_id === product.seller_id);
        const productReviews = reviewsData.filter(r => r.product_id === product.product_id);
        const analysis = analysisMap[product.product_id];
        
        // Use the actual trust score from analysis
        let trustScore = 75;
        let verificationStatus = 'CAUTION';
        let fraudIndicators = [];
        
        if (analysis) {
          trustScore = analysis.trust_score;
          verificationStatus = analysis.verification_status;
          fraudIndicators = analysis.fraud_indicators || [];
        }
        
        return {
          ...product,
          id: product.product_id,
          seller,
          reviews: productReviews,
          trustScore,
          fraudIndicators,
          verificationStatus,
          hasAnalysis: !!analysis
        };
      });

      setProducts(enhancedProducts);
      
    } catch (error) {
      console.error("Error loading analysis:", error);
      // Fallback
      const basicProducts = productsData.map(product => ({
        ...product,
        id: product.product_id,
        trustScore: 75,
        verificationStatus: 'PENDING',
        fraudIndicators: [],
        hasAnalysis: false
      }));
      setProducts(basicProducts);
    }
    
    setLoading(false);
  };

  const filterProducts = () => {
    if (filter === 'all') return products;
    if (filter === 'verified') return products.filter(p => p.trustScore >= 75);
    if (filter === 'caution') return products.filter(p => p.trustScore >= 50 && p.trustScore < 75);
    if (filter === 'high-risk') return products.filter(p => p.trustScore < 50);
    return products;
  };

  if (loading) {
    return (
      <>
        <Navbar />
        <div style={{ textAlign: 'center', padding: '100px 20px' }}>
          <p>Loading products...</p>
        </div>
      </>
    );
  }

  const filteredProducts = filterProducts();

  return (
    <>
      <Navbar />
      <div style={{ minHeight: '100vh', backgroundColor: '#f7f7f7', paddingTop: '80px' }}>
        
        {/* Status Banner */}
        <div style={{
          background: 'linear-gradient(135deg, #00a862 0%, #008c51 100%)',
          color: 'white',
          padding: '16px 30px',
          marginBottom: '20px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', maxWidth: '1400px', margin: '0 auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <span style={{ fontSize: '24px' }}>🛡️</span>
              <div>
                <div style={{ fontWeight: 'bold' }}>TrustSight Protection Active</div>
                <div style={{ fontSize: '14px', opacity: 0.9 }}>
                  {products.length} products analyzed with AI pipeline
                </div>
              </div>
            </div>
            <div style={{
              background: 'white',
              color: '#00a862',
              padding: '8px 16px',
              borderRadius: '20px',
              fontWeight: 'bold',
              fontSize: '12px'
            }}>
              ✓ PRE-ANALYZED DATASET
            </div>
          </div>
        </div>

        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '20px 30px',
          maxWidth: '1400px',
          margin: '0 auto'
        }}>
          <h1 style={{ fontSize: '28px', margin: 0, color: '#232f3e' }}>
            Products Protected by TrustSight™
          </h1>
          
          {/* Filter */}
          <div style={{ position: 'relative' }}>
            <button 
              onClick={() => setShowTrustFilter(!showTrustFilter)}
              style={{
                background: '#667eea',
                color: 'white',
                border: 'none',
                padding: '10px 20px',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              🛡️ Filter: {filter === 'all' ? 'All Products' : filter}
            </button>
            
            {showTrustFilter && (
              <div style={{
                position: 'absolute',
                top: '100%',
                right: 0,
                marginTop: '8px',
                background: 'white',
                borderRadius: '8px',
                boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                zIndex: 100,
                minWidth: '200px',
                overflow: 'hidden'
              }}>
                {['all', 'verified', 'caution', 'high-risk'].map(option => (
                  <button
                    key={option}
                    onClick={() => { setFilter(option); setShowTrustFilter(false); }}
                    style={{
                      display: 'block',
                      width: '100%',
                      padding: '12px 20px',
                      border: 'none',
                      background: filter === option ? '#667eea' : 'white',
                      color: filter === option ? 'white' : 'black',
                      textAlign: 'left',
                      cursor: 'pointer'
                    }}
                  >
                    {option === 'all' && 'All Products'}
                    {option === 'verified' && '✅ Verified (75%+)'}
                    {option === 'caution' && '⚠️ Caution (50-74%)'}
                    {option === 'high-risk' && '❌ High Risk (<50%)'}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Products Grid - KEY FIX HERE */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '25px',
          padding: '0 30px 60px',
          maxWidth: '1400px',
          margin: '0 auto'
        }}>
          {filteredProducts.map((product) => (
            <div key={product.id} style={{
              backgroundColor: 'white',
              borderRadius: '12px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
              overflow: 'hidden',
              transition: 'all 0.3s ease',
              position: 'relative'
            }}>
              <Link to={`/product/${product.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
                {/* Image Container */}
                <div style={{
                  position: 'relative',
                  padding: '20px',
                  background: '#f8f8f8',
                  textAlign: 'center',
                  height: '200px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <img 
                    src={product.image_urls?.[0] || "/placeholder.jpg"}
                    alt={product.title}
                    style={{ maxHeight: '160px', maxWidth: '100%', objectFit: 'contain' }}
                  />
                  
                  {/* Trust Score Badge */}
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    width: '60px',
                    height: '60px',
                    borderRadius: '50%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 'bold',
                    backgroundColor: product.trustScore >= 75 ? '#00a862' : 
                                    product.trustScore >= 50 ? '#ff9900' : '#d13212',
                    color: 'white',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                  }}>
                    <div style={{ fontSize: '20px' }}>{product.trustScore}</div>
                    <div style={{ fontSize: '9px', textTransform: 'uppercase' }}>Score</div>
                  </div>
                </div>

                {/* Product Info */}
                <div style={{ padding: '20px' }}>
                  <h3 style={{
                    fontSize: '16px',
                    fontWeight: '500',
                    marginBottom: '10px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical'
                  }}>
                    {product.title}
                  </h3>
                  
                  {/* Price */}
                  <div style={{ marginBottom: '10px' }}>
                    <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#c45500' }}>
                      ${product.price}
                    </span>
                    {product.market_price && product.market_price > product.price && (
                      <span style={{ marginLeft: '10px', textDecoration: 'line-through', color: '#666' }}>
                        ${product.market_price}
                      </span>
                    )}
                  </div>
                  
                  {/* Seller */}
                  <div style={{ fontSize: '13px', color: '#666' }}>
                    Sold by: {product.seller?.display_name || 'Unknown'}
                  </div>
                </div>
              </Link>
            </div>
          ))}
        </div>
      </div>
      <Footer />
    </>
  );
}

export default Products;