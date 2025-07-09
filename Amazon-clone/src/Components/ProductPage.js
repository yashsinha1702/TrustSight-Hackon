// Updated ProductPage.js with Add Review functionality
import React, { useState, useEffect, useRef } from "react";
import { useParams, Link } from "react-router-dom";
import Navbar from "./Navbar";
import Footer from "./Footer";
import TrustSightIndicator from "./TrustSightIndicator";
import ProductAnalyzer from "./ProductAnalyzer";
import AddReviewForm from "./AddReviewForm"; // New import
import trustSightService from "../services/trustSightService";
import "./productpage.css";
import "./trustSightStyles.css";
import Rating from "../imgs/rating.png";
import added from "../imgs/added.png";
import add from "../imgs/not-added.png";
import { AddToCart, RemoveCart } from "../action/Cart";
import { useSelector, useDispatch } from "react-redux";
import VanillaTilt from "vanilla-tilt";

// Import fraud datasets
import productsData from "../data/products.json";
import sellersData from "../data/sellers.json";
import reviewsData from "../data/reviews.json";
import reviewersData from "../data/reviewers.json";

function ProductPage() {
  const { id } = useParams();
  const [product, setProduct] = useState(null);
  const [seller, setSeller] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [fraudAnalysis, setFraudAnalysis] = useState(null);
  const [showFraudDetails, setShowFraudDetails] = useState(false);
  const [showAnalyzer, setShowAnalyzer] = useState(false);
  const [showAddReview, setShowAddReview] = useState(false); // New state
  const [Size, setSize] = useState("");
  const [AddedIds, setAddedIds] = useState([]);
  const [loading, setLoading] = useState(true);
  const Quantity = 1;

  const tiltRef = useRef(null);

  const CartItems = useSelector((state) => state.CartItemsAdded.CartItems);
  const dispatch = useDispatch();

  useEffect(() => {
    loadProductData();
    window.scrollTo(0, 0);
  }, [id]);

  useEffect(() => {
    const ids = CartItems.map((item) => item.id);
    setAddedIds(ids);
  }, [CartItems]);

  useEffect(() => {
    if (product && product.image_urls?.[0] && tiltRef.current) {
      VanillaTilt.init(tiltRef.current, {
        max: 25,
        speed: 400,
        glare: true,
        "max-glare": 0.5,
      });

      return () => {
        if (tiltRef.current && tiltRef.current.vanillaTilt) {
          tiltRef.current.vanillaTilt.destroy();
        }
      };
    }
  }, [product]);

  const loadProductData = async () => {
    setLoading(true);
    
    const foundProduct = productsData.find(p => p.product_id === id);
    if (foundProduct) {
      setProduct(foundProduct);
      
      const foundSeller = sellersData.find(s => s.seller_id === foundProduct.seller_id);
      setSeller(foundSeller || null);
      
      const productReviews = reviewsData.filter(r => r.product_id === id);
      setReviews(productReviews);
      
      // Run basic fraud analysis
      const analysis = await runBasicFraudAnalysis(foundProduct, foundSeller, productReviews);
      setFraudAnalysis(analysis);
    }
    
    setLoading(false);
  };

  const runBasicFraudAnalysis = async (product, seller, reviews) => {
    // Use TrustSight service for analysis
    return await trustSightService.analyzeProduct(product, seller, reviews);
  };

  const handleAnalysisComplete = (result) => {
    setFraudAnalysis(result);
    setShowAnalyzer(false);
    setShowFraudDetails(true);
  };

  const handleReviewAdded = (newReview, analysis) => {
    // Add the new review to the reviews list
    setReviews([newReview, ...reviews]);
    
    // Show notification
    alert(`Review added! Authenticity score: ${analysis.authenticityScore}%`);
    
    // Close the form
    setShowAddReview(false);
    
    // Re-run fraud analysis with new review
    runBasicFraudAnalysis(product, seller, [newReview, ...reviews]).then(result => {
      setFraudAnalysis(result);
    });
  };

  const isAdded = (id) => AddedIds.includes(id);

  const AddItem = () => {
    if (Size === "" && product.category === "Fashion") {
      alert("Please select a size");
    } else {
      const item = {
        id: product.id,
        title: product.title,
        price: product.price,
        image: product.image_urls?.[0] || product.image,
        Size: Size || "N/A",
        quantity: Quantity,
      };
      dispatch(AddToCart(item));
    }
  };

  const RemoveItem = (id) => {
    dispatch(RemoveCart(id));
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (!product) {
    return (
      <>
        <Navbar />
        <div className="error-container">
          <h2>Product not found</h2>
          <Link to="/products">Back to products</Link>
        </div>
        <Footer />
      </>
    );
  }

  return (
    <>
      <Navbar />
      <div className="product-page">
        {/* Add Review Form Modal */}
        {showAddReview && (
          <AddReviewForm
            productId={product.product_id}
            onReviewAdded={handleReviewAdded}
            onClose={() => setShowAddReview(false)}
          />
        )}

        <div className="productpage-main">
          {/* Product Images */}
          <div className="productpage-left">
            <div className="product-image-container" ref={tiltRef}>
              <img
                src={product.image_urls?.[0] || "/placeholder.jpg"}
                alt={product.title}
                className="main-product-image"
              />
              <TrustSightIndicator 
                product={product} 
                fraudAnalysis={fraudAnalysis}
                onViewDetails={() => setShowFraudDetails(true)}
                size="large"
              />
            </div>
          </div>

          {/* Product Details */}
          <div className="productpage-right">
            <h1>{product.title}</h1>
            
            {/* Trust Score Badge */}
            <div className="trust-score-section">
              <button 
                className="analyze-real-time-btn"
                onClick={() => setShowAnalyzer(true)}
              >
                🔍 Run Real-Time Analysis
              </button>
            </div>

            <p className="product-description">{product.description}</p>

            {/* Price Section */}
            <div className="price-wrapper">
              <p className="product-price">${product.price}</p>
              {product.market_price && (
                <p className="market-price">${product.market_price}</p>
              )}
            </div>

            {/* Rating */}
            <div className="rating-section">
              <img src={Rating} alt="rating" className="rating-img" />
              <p>({reviews.length} reviews)</p>
            </div>

            {/* Seller Info */}
            {seller && (
              <div className="seller-info">
                <p>Sold by: <strong>{seller.display_name}</strong></p>
                {seller.seller_name?.includes("unauthorized") && (
                  <p className="unauthorized-warning">⚠️ Unauthorized Seller</p>
                )}
              </div>
            )}

            {/* Size Selection */}
            {product.category === "Fashion" && (
              <div className="size-selection">
                <p>Size:</p>
                <div className="size-buttons">
                  {["S", "M", "L", "XL"].map((size) => (
                    <button
                      key={size}
                      className={Size === size ? "selected" : ""}
                      onClick={() => setSize(size)}
                    >
                      {size}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Add to Cart Button */}
            <div className="productpage-cart">
              <button
                onClick={() => isAdded(product.id) ? RemoveItem(product.id) : AddItem()}
                className="add-to-cart-btn"
              >
                <img src={isAdded(product.id) ? added : add} className="cart-img" alt="cart" />
                <p style={{ marginLeft: "8px" }} className="cart-text">
                  {isAdded(product.id) ? "Added" : "Add"}
                </p>
              </button>
            </div>
          </div>
        </div>

        {/* ProductAnalyzer Component */}
        {showAnalyzer && (
          <div className="analyzer-section">
            <ProductAnalyzer
              product={product}
              seller={seller}
              reviews={reviews}
              onAnalysisComplete={handleAnalysisComplete}
            />
          </div>
        )}

        {/* Reviews Section */}
        <div className="reviews-section">
          <div className="reviews-header">
            <h2>Customer Reviews ({reviews.length})</h2>
            <button 
              className="add-review-btn"
              onClick={() => setShowAddReview(true)}
            >
              ✍️ Write a Review
            </button>
          </div>

          <div className="reviews-list">
            {reviews.length === 0 ? (
              <p className="no-reviews">No reviews yet. Be the first to review!</p>
            ) : (
              reviews.map((review, index) => (
                <div key={review.review_id || index} className="review-item">
                  <div className="review-header">
                    <div>
                      <span className="reviewer-name">{review.reviewer_name}</span>
                      <div className="review-rating">
                        {'⭐'.repeat(review.rating)}
                      </div>
                    </div>
                    <span className="review-date">
                      {new Date(review.review_timestamp).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="review-text">{review.review_text}</p>
                  {review.verified_purchase && (
                    <span className="verified-badge">✓ Verified Purchase</span>
                  )}
                  {/* Show if review is flagged as fake */}
                  {fraudAnalysis?.detections?.review?.individual_reviews?.find(
                    r => r.review_id === review.review_id && r.is_fake
                  ) && (
                    <span className="fake-review-badge">⚠️ Suspicious Review</span>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Fraud Analysis Modal */}
        {showFraudDetails && fraudAnalysis && !showAnalyzer && (
          <div className="fraud-modal-overlay" onClick={() => setShowFraudDetails(false)}>
            <div className="fraud-modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h2>TrustSight Analysis Report</h2>
                <button className="close-btn" onClick={() => setShowFraudDetails(false)}>×</button>
              </div>
              
              <div className="modal-content">
                {/* Trust Score */}
                <div className="trust-score-section">
                  <h3>Overall Trust Score</h3>
                  <div className="trust-score-display">
                    <div className="score-circle" style={{
                      borderColor: fraudAnalysis.trustScore >= 80 ? '#00a862' : 
                                   fraudAnalysis.trustScore >= 50 ? '#ff9900' : '#d13212'
                    }}>
                      <span className="score-value">{fraudAnalysis.trustScore}%</span>
                    </div>
                    <p className="verification-status">{fraudAnalysis.verificationStatus}</p>
                  </div>
                </div>

                {/* Analysis Method Badge */}
                <div className="analysis-method">
                  {fraudAnalysis.backendAnalyzed ? (
                    <span className="method-badge backend">🤖 AI Pipeline Analysis</span>
                  ) : (
                    <span className="method-badge fallback">📊 Basic Analysis</span>
                  )}
                </div>

                {/* Risk Factors */}
                {fraudAnalysis.riskFactors && fraudAnalysis.riskFactors.length > 0 && (
                  <div className="risk-factors-section">
                    <h3>Risk Factors Detected</h3>
                    {fraudAnalysis.riskFactors.map((factor, index) => (
                      <div key={index} className="risk-factor">
                        <span className="risk-icon">⚠️</span>
                        <span>{factor}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Fraud Indicators */}
                {fraudAnalysis.fraudIndicators && fraudAnalysis.fraudIndicators.length > 0 && (
                  <div className="fraud-indicators-section">
                    <h3>Fraud Indicators</h3>
                    {fraudAnalysis.fraudIndicators.map((indicator, index) => (
                      <div key={index} className="fraud-indicator">
                        <div className="indicator-header">
                          <span className={`severity-badge ${indicator.severity?.toLowerCase()}`}>
                            {indicator.severity}
                          </span>
                          <span className="indicator-type">{indicator.type}</span>
                        </div>
                        <p className="indicator-description">{indicator.description}</p>
                      </div>
                    ))}
                  </div>
                )}

                {/* Actions */}
                <div className="modal-actions">
                  <button 
                    className="full-scan-btn"
                    onClick={() => {
                      setShowFraudDetails(false);
                      setShowAnalyzer(true);
                    }}
                  >
                    Run Full Analysis
                  </button>
                  <button className="report-fraud-btn">
                    Report Fraud
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      <Footer />
    </>
  );
}

export default ProductPage;