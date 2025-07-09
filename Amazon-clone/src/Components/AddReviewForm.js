import React, { useState } from 'react';

const AddReviewForm = ({ productId, onReviewAdded, onClose }) => {
  const [formData, setFormData] = useState({
    reviewerName: '',
    rating: 5,
    reviewText: '',
    verifiedPurchase: false
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleSubmit = async () => {
    if (!formData.reviewerName || !formData.reviewText) {
      alert('Please fill in all required fields');
      return;
    }

    setIsAnalyzing(true);

    // Create new review object
    const newReview = {
      review_id: `R_NEW_${Date.now()}`,
      product_id: productId,
      reviewer_id: `REV_${formData.reviewerName.replace(/\s/g, '_')}`,
      reviewer_name: formData.reviewerName,
      rating: formData.rating,
      review_text: formData.reviewText,
      verified_purchase: formData.verifiedPurchase,
      review_timestamp: new Date().toISOString(),
      helpful_votes: 0,
      total_votes: 0
    };

    // Simulate analysis
    setTimeout(() => {
      const mockAnalysis = {
        isAuthentic: Math.random() > 0.3,
        authenticityScore: Math.floor(Math.random() * 40) + 60,
        patterns: ['genuine_language', 'detailed_experience']
      };
      
      setAnalysisResult(mockAnalysis);
      setIsAnalyzing(false);
      
      // Add review after showing analysis
      setTimeout(() => {
        onReviewAdded(newReview, mockAnalysis);
      }, 2000);
    }, 3000);
  };

  const styles = {
    overlay: {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 10000
    },
    modal: {
      background: 'white',
      borderRadius: '12px',
      width: '90%',
      maxWidth: '500px',
      maxHeight: '90vh',
      overflowY: 'auto',
      boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)'
    },
    header: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '24px',
      borderBottom: '1px solid #eee'
    },
    closeBtn: {
      background: 'none',
      border: 'none',
      fontSize: '24px',
      cursor: 'pointer',
      color: '#666',
      width: '32px',
      height: '32px',
      borderRadius: '50%'
    },
    content: {
      padding: '24px'
    },
    formGroup: {
      marginBottom: '20px'
    },
    label: {
      display: 'block',
      marginBottom: '8px',
      color: '#333',
      fontWeight: '500'
    },
    input: {
      width: '100%',
      padding: '10px',
      border: '1px solid #ddd',
      borderRadius: '6px',
      fontSize: '14px'
    },
    textarea: {
      width: '100%',
      padding: '10px',
      border: '1px solid #ddd',
      borderRadius: '6px',
      fontSize: '14px',
      resize: 'vertical'
    },
    ratingSelector: {
      display: 'flex',
      gap: '8px'
    },
    star: {
      fontSize: '24px',
      cursor: 'pointer',
      opacity: 0.3,
      transition: 'opacity 0.3s ease'
    },
    starFilled: {
      opacity: 1
    },
    checkbox: {
      marginRight: '8px'
    },
    actions: {
      display: 'flex',
      gap: '12px',
      marginTop: '24px'
    },
    submitBtn: {
      flex: 1,
      padding: '12px',
      border: 'none',
      borderRadius: '6px',
      background: '#667eea',
      color: 'white',
      fontWeight: '500',
      cursor: 'pointer'
    },
    cancelBtn: {
      flex: 1,
      padding: '12px',
      border: 'none',
      borderRadius: '6px',
      background: '#f0f0f0',
      color: '#333',
      fontWeight: '500',
      cursor: 'pointer'
    },
    analysisProgress: {
      textAlign: 'center',
      padding: '60px 24px'
    },
    spinner: {
      width: '60px',
      height: '60px',
      border: '4px solid #f0f0f0',
      borderTop: '4px solid #667eea',
      borderRadius: '50%',
      margin: '0 auto 24px',
      animation: 'spin 1s linear infinite'
    },
    analysisComplete: {
      padding: '40px 24px',
      textAlign: 'center'
    },
    authenticityScore: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '16px',
      borderRadius: '8px',
      margin: '20px 0'
    },
    scoreAuthentic: {
      background: '#e8f5e8',
      color: '#2d5a2d'
    },
    scoreSuspicious: {
      background: '#fff5f5',
      color: '#d13212'
    },
    scoreValue: {
      fontSize: '24px',
      fontWeight: 'bold'
    },
    warningMessage: {
      background: '#fff3cd',
      color: '#856404',
      padding: '12px',
      borderRadius: '6px',
      margin: '16px 0'
    },
    doneBtn: {
      background: '#00a862',
      color: 'white',
      padding: '12px 24px',
      border: 'none',
      borderRadius: '6px',
      cursor: 'pointer',
      fontWeight: '500'
    }
  };

  return (
    <div style={styles.overlay}>
      <div style={styles.modal}>
        <div style={styles.header}>
          <h2 style={{ margin: 0, color: '#333' }}>Add New Review</h2>
          <button style={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        {!isAnalyzing && !analysisResult ? (
          <div style={styles.content}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Your Name</label>
              <input
                style={styles.input}
                type="text"
                value={formData.reviewerName}
                onChange={(e) => setFormData({...formData, reviewerName: e.target.value})}
                placeholder="Enter your name"
              />
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Rating</label>
              <div style={styles.ratingSelector}>
                {[1, 2, 3, 4, 5].map(star => (
                  <span
                    key={star}
                    style={{
                      ...styles.star,
                      ...(formData.rating >= star ? styles.starFilled : {})
                    }}
                    onClick={() => setFormData({...formData, rating: star})}
                  >
                    ⭐
                  </span>
                ))}
              </div>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Review Text</label>
              <textarea
                style={styles.textarea}
                value={formData.reviewText}
                onChange={(e) => setFormData({...formData, reviewText: e.target.value})}
                rows="4"
                placeholder="Share your thoughts about this product..."
              />
            </div>

            <div style={styles.formGroup}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  style={styles.checkbox}
                  type="checkbox"
                  checked={formData.verifiedPurchase}
                  onChange={(e) => setFormData({...formData, verifiedPurchase: e.target.checked})}
                />
                Verified Purchase
              </label>
            </div>

            <div style={styles.actions}>
              <button style={styles.submitBtn} onClick={handleSubmit}>
                Submit Review & Analyze
              </button>
              <button style={styles.cancelBtn} onClick={onClose}>
                Cancel
              </button>
            </div>
          </div>
        ) : isAnalyzing ? (
          <div style={styles.analysisProgress}>
            <div style={styles.spinner}></div>
            <h3>Analyzing Review with TrustSight...</h3>
            <p>Checking for authenticity patterns...</p>
          </div>
        ) : (
          <div style={styles.analysisComplete}>
            <h3>Analysis Complete!</h3>
            <div style={{
              ...styles.authenticityScore,
              ...(analysisResult.isAuthentic ? styles.scoreAuthentic : styles.scoreSuspicious)
            }}>
              <span>Authenticity Score:</span>
              <span style={styles.scoreValue}>{analysisResult.authenticityScore}%</span>
            </div>
            {!analysisResult.isAuthentic && (
              <div style={styles.warningMessage}>
                ⚠️ This review shows signs of being inauthentic
              </div>
            )}
            <button style={styles.doneBtn} onClick={() => {
              setAnalysisResult(null);
              setFormData({
                reviewerName: '',
                rating: 5,
                reviewText: '',
                verifiedPurchase: false
              });
            }}>
              Add Another Review
            </button>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default AddReviewForm;