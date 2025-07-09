"""
TRUSTSIGHT TESTING UI - Enhanced Version
Comprehensive testing interface for all fraud detection components
"""

import streamlit as st
import requests
import json
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="TrustSight - Fraud Detection Testing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stAlert {
        margin-top: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fraud-high { color: #ff4b4b; }
    .fraud-medium { color: #ffa500; }
    .fraud-low { color: #00cc00; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Test Data Templates
TEST_SCENARIOS = {
    'fake_nike': {
        'name': '🎯 Fake Nike Product',
        'entity_type': 'product',
        'entity_data': {
            'title': 'Nike Air Max 2024 CHEAP BEST PRICE!!! LIMITED OFFER',
            'brand': 'Nike',
            'price': 29.99,
            'category': 'Shoes',
            'seller_id': 'SUSPICIOUS_SELLER_123',
            'description': 'Genuine Nike shoes cheap fast shipping from warehouse clearance sale',
            'images': ['fake_nike_1.jpg', 'fake_nike_2.jpg']
        },
        'expected_fraud_score': 0.85
    },
    'fake_review_network': {
        'name': '💬 Fake Review Network',
        'entity_type': 'review',
        'entity_data': {
            'review_text': 'Excellent product! Best quality! Fast shipping! Highly recommend to all! Five stars!',
            'rating': 5,
            'reviewer_id': 'REVIEWER_BOT_123',
            'product_id': 'PROD_NIKE_001',
            'verified_purchase': False,
            'timestamp': datetime.now().isoformat(),
            'review_history': [
                {'product_id': 'PROD_002', 'rating': 5, 'days_ago': 1},
                {'product_id': 'PROD_003', 'rating': 5, 'days_ago': 2},
                {'product_id': 'PROD_004', 'rating': 5, 'days_ago': 3}
            ]
        },
        'expected_fraud_score': 0.92
    },
    'seller_fraud_network': {
        'name': '🏪 Seller Fraud Network',
        'entity_type': 'seller',
        'entity_data': {
            'seller_name': 'SuperDeals2024',
            'registration_date': (datetime.now() - timedelta(days=5)).isoformat(),
            'address': '123 Industrial Park, Unit 5',
            'products_count': 247,
            'shared_attributes': {
                'bank_account': 'SHARED_123',
                'tax_id': 'FAKE_TAX_123',
                'phone': '+1234567890'
            },
            'connected_sellers': ['SuperDeals2023', 'BestDeals2024', 'MegaDeals2024']
        },
        'expected_fraud_score': 0.78
    },
    'listing_manipulation': {
        'name': '📄 Listing Manipulation',
        'entity_type': 'listing',
        'entity_data': {
            'title': 'Nike Adidas Puma Reebok Best Shoes All Brands Cheap Price Wholesale',
            'category': 'Shoes',
            'variations': 150,
            'seo_keywords': ['nike', 'adidas', 'puma', 'cheap', 'wholesale', 'best price'],
            'price_variations': {
                'min': 19.99,
                'max': 299.99,
                'avg': 45.99
            }
        },
        'expected_fraud_score': 0.88
    },
    'legitimate_product': {
        'name': '✅ Legitimate Product',
        'entity_type': 'product',
        'entity_data': {
            'title': 'Nike Air Max 270 React',
            'brand': 'Nike',
            'price': 150.00,
            'category': 'Shoes',
            'seller_id': 'VERIFIED_NIKE_RETAILER',
            'description': 'The Nike Air Max 270 React combines Nike\'s tallest Air unit yet with React foam.',
            'verified_seller': True
        },
        'expected_fraud_score': 0.15
    }
}

# ============= Helper Functions =============

def make_api_request(endpoint: str, method: str = "GET", data: dict = None):
    """Make API request with better error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        headers = {'Content-Type': 'application/json'}
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        else:
            response = requests.post(url, json=data, headers=headers, timeout=30)
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API")
        st.info("Please ensure the backend is running: `python integration.py`")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"🚨 API Error: {e}")
        if hasattr(e.response, 'text'):
            st.error(f"Details: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def create_trust_score_gauge(trust_score, fraud_score):
    """Enhanced gauge chart with dual metrics"""
    fig = go.Figure()
    
    # Trust Score Gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = trust_score,
        title = {'text': "Trust Score", 'font': {'size': 24}},
        delta = {'reference': 50, 'position': "bottom"},
        domain = {'x': [0, 0.45], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue", 'thickness': 0.8},
            'steps': [
                {'range': [0, 25], 'color': "#ffebee"},
                {'range': [25, 50], 'color': "#fff3e0"},
                {'range': [50, 75], 'color': "#e8f5e9"},
                {'range': [75, 100], 'color': "#c8e6c9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Fraud Score Gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = fraud_score * 100,
        title = {'text': "Fraud Risk", 'font': {'size': 24}},
        delta = {'reference': 50, 'position': "bottom"},
        domain = {'x': [0.55, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "red", 'thickness': 0.8},
            'steps': [
                {'range': [0, 25], 'color': "#c8e6c9"},
                {'range': [25, 50], 'color': "#fff3e0"},
                {'range': [50, 75], 'color': "#ffccbc"},
                {'range': [75, 100], 'color': "#ffcdd2"}
            ]
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

def create_detector_results_chart(detector_results):
    """Enhanced bar chart for detector results"""
    if not detector_results:
        return None
    
    # Prepare data
    detectors = []
    scores = []
    confidences = []
    
    for detector, result in detector_results.items():
        detectors.append(detector.replace('_', ' ').title())
        scores.append(result.get('score', 0) * 100)
        confidences.append(result.get('confidence', 0) * 100)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Fraud Score',
        x=detectors,
        y=scores,
        text=[f'{s:.1f}%' for s in scores],
        textposition='auto',
        marker_color='indianred'
    ))
    
    fig.add_trace(go.Bar(
        name='Confidence',
        x=detectors,
        y=confidences,
        text=[f'{c:.1f}%' for c in confidences],
        textposition='auto',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Detection Results by Component",
        xaxis_title="Detector",
        yaxis_title="Score (%)",
        barmode='group',
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_network_visualization(cross_intel_result):
    """Enhanced network visualization with more details"""
    if not cross_intel_result:
        return None
    
    network_size = cross_intel_result.get('network_size', 1)
    network_type = cross_intel_result.get('network_type', 'Unknown')
    key_players = cross_intel_result.get('key_players', [])
    
    # Create network graph
    fig = go.Figure()
    
    # Add edges first (so they appear behind nodes)
    edge_trace = []
    node_trace = []
    
    # Center node (detected entity)
    node_trace.append(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=60, color='red', symbol='star'),
        text=['Detected Entity'],
        textposition='bottom center',
        name='Primary Detection'
    ))
    
    # Create nodes in layers
    layers = min(3, (network_size - 1) // 10 + 1)
    nodes_per_layer = min(20, network_size - 1) // layers
    
    node_positions = [(0, 0)]  # Center node
    
    for layer in range(1, layers + 1):
        radius = layer * 2
        for i in range(min(nodes_per_layer, network_size - len(node_positions))):
            angle = 2 * np.pi * i / nodes_per_layer
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            node_positions.append((x, y))
            
            # Add edge
            edge_trace.append(go.Scatter(
                x=[0, x], y=[0, y],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
            
            # Determine node type
            node_color = 'orange'
            node_size = 40
            if i < len(key_players):
                node_color = 'darkred'
                node_size = 50
            
            # Add node
            node_trace.append(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=node_size, color=node_color),
                showlegend=False,
                hoverinfo='text',
                text=f'Entity {len(node_positions)}'
            ))
    
    # Add all traces
    for trace in edge_trace:
        fig.add_trace(trace)
    for trace in node_trace:
        fig.add_trace(trace)
    
    # Add network statistics
    annotations = [
        dict(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"<b>Network Type:</b> {network_type}<br>" +
                 f"<b>Total Entities:</b> {network_size}<br>" +
                 f"<b>Key Players:</b> {len(key_players)}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12),
            align="left"
        )
    ]
    
    fig.update_layout(
        title=f"Fraud Network Visualization - {network_size} Connected Entities",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        annotations=annotations,
        hovermode='closest'
    )
    
    return fig

def display_evidence(detector_results):
    """Display evidence in a structured format"""
    all_evidence = []
    
    for detector, result in detector_results.items():
        if 'evidence' in result and result['evidence']:
            for evidence in result['evidence']:
                all_evidence.append({
                    'Detector': detector.replace('_', ' ').title(),
                    'Type': evidence.get('type', 'Unknown'),
                    'Description': evidence.get('description', 'No description'),
                    'Score Impact': f"{result.get('score', 0) * 100:.1f}%"
                })
    
    if all_evidence:
        df = pd.DataFrame(all_evidence)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Score Impact': st.column_config.TextColumn(
                    'Impact',
                    help='Contribution to fraud score'
                )
            }
        )
    else:
        st.info("No detailed evidence available")

def display_lifecycle_analysis(lifecycle_data):
    """Display lifecycle analysis results"""
    if not lifecycle_data:
        st.info("No lifecycle analysis available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stage = lifecycle_data.get('current_stage', 'Unknown')
        stage_color = {
            'new': '🟢',
            'establishing': '🟡',
            'mature': '🟠',
            'declining': '🔴'
        }.get(stage, '⚪')
        st.metric("Current Stage", f"{stage_color} {stage.title()}")
    
    with col2:
        risk_level = lifecycle_data.get('risk_level', 'Unknown')
        st.metric("Risk Level", risk_level.title())
    
    with col3:
        indicators = lifecycle_data.get('indicators', [])
        st.metric("Risk Indicators", len(indicators))
    
    if indicators:
        st.subheader("Risk Indicators")
        for indicator in indicators:
            st.write(f"• {indicator}")

# ============= Main UI =============

def main():
    # Header with custom styling
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
        🛡️ TrustSight - Fraud Detection Testing Suite
    </h1>
    <p style='text-align: center; color: #666;'>
        Comprehensive testing interface for all fraud detection components
    </p>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'test_history' not in st.session_state:
        st.session_state.test_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check backend health
        with st.spinner("Checking backend..."):
            health = make_api_request("/health")
        
        if health:
            st.success("✅ Backend Connected")
            
            # Display component status
            components = health.get('components', {})
            st.subheader("Component Status")
            
            status_df = pd.DataFrame([
                {"Component": comp, "Status": "✅ Active" if status else "❌ Inactive"}
                for comp, status in components.items()
            ])
            st.dataframe(status_df, hide_index=True, use_container_width=True)
            
            # System metrics - with error handling
            try:
                metrics = make_api_request("/metrics")
                if metrics:
                    st.subheader("System Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Active", metrics.get('active_requests', 0))
                        st.metric("Queue", metrics.get('queue_size', 0))
                    with col2:
                        st.metric("Completed", metrics.get('completed_requests', 0))
                        st.metric("Cache", metrics.get('cache_size', 0))
                else:
                    st.warning("⚠️ Metrics endpoint not available")
            except Exception as e:
                st.warning("⚠️ Could not load metrics")
                logger.debug(f"Metrics error: {e}")
        else:
            st.error("❌ Backend Disconnected")
            st.stop()
        
        st.divider()
        
        # Quick test scenarios
        st.header("🚀 Quick Test Scenarios")
        
        for scenario_key, scenario in TEST_SCENARIOS.items():
            if st.button(scenario['name'], key=f"btn_{scenario_key}", use_container_width=True):
                st.session_state['selected_scenario'] = scenario_key
                st.rerun()
        
        st.divider()
        
        # Test history
        if st.session_state.test_history:
            st.header("📊 Recent Tests")
            for i, test in enumerate(reversed(st.session_state.test_history[-5:])):
                with st.expander(f"{test['entity_type']} - {test['time']}", expanded=False):
                    st.json(test['result'])
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Fraud Detection", 
        "📊 Results Analysis", 
        "🕸️ Network Intelligence",
        "📈 Lifecycle & Predictions",
        "🔧 Raw Data"
    ])
    
    with tab1:
        st.header("Submit Entity for Fraud Detection")
        
        # Load selected scenario if any
        if 'selected_scenario' in st.session_state:
            scenario = TEST_SCENARIOS[st.session_state.selected_scenario]
            entity_type = scenario['entity_type']
            entity_data = scenario['entity_data'].copy()
            st.success(f"Loaded scenario: {scenario['name']}")
            del st.session_state.selected_scenario
        else:
            entity_type = None
            entity_data = {}
        
        # Entity configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            entity_type = st.selectbox(
                "Entity Type",
                ["product", "review", "seller", "listing"],
                index=["product", "review", "seller", "listing"].index(entity_type) if entity_type else 0
            )
        
        with col2:
            entity_id = st.text_input(
                "Entity ID",
                value=f"{entity_type.upper()}_{int(time.time() * 1000)}"
            )
        
        with col3:
            priority = st.selectbox(
                "Priority",
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                index=1
            )
        
        # Dynamic form based on entity type
        st.subheader("Entity Details")
        
        if entity_type == "product":
            col1, col2 = st.columns(2)
            with col1:
                entity_data['title'] = st.text_input("Product Title", 
                    value=entity_data.get('title', 'Nike Air Max 2024'))
                entity_data['brand'] = st.text_input("Brand", 
                    value=entity_data.get('brand', 'Nike'))
                entity_data['category'] = st.text_input("Category", 
                    value=entity_data.get('category', 'Shoes'))
            with col2:
                entity_data['price'] = st.number_input("Price ($)", 
                    min_value=0.0, value=entity_data.get('price', 150.0))
                entity_data['seller_id'] = st.text_input("Seller ID", 
                    value=entity_data.get('seller_id', 'SELLER_001'))
            
            entity_data['description'] = st.text_area("Description", 
                value=entity_data.get('description', 'Genuine product description'))
        
        elif entity_type == "review":
            entity_data['review_text'] = st.text_area(
                "Review Text",
                value=entity_data.get('review_text', 'Great product!')
            )
            
            col1, col2 = st.columns(2)
            with col1:
                entity_data['rating'] = st.slider("Rating", 1, 5, 
                    value=entity_data.get('rating', 5))
                entity_data['reviewer_id'] = st.text_input("Reviewer ID", 
                    value=entity_data.get('reviewer_id', 'REVIEWER_001'))
            with col2:
                entity_data['product_id'] = st.text_input("Product ID", 
                    value=entity_data.get('product_id', 'PROD_001'))
                entity_data['verified_purchase'] = st.checkbox("Verified Purchase",
                    value=entity_data.get('verified_purchase', False))
            
            # entity_data['timestamp'] = datetime.now().isoformat()
            entity_data['review_timestamp'] = datetime.now().isoformat()
            entity_data['delivery_timestamp'] = (datetime.now() - timedelta(days=2)).isoformat()

        
        elif entity_type == "seller":
            col1, col2 = st.columns(2)
            with col1:
                entity_data['seller_name'] = st.text_input("Seller Name", 
                    value=entity_data.get('seller_name', 'BestDeals123'))
                entity_data['registration_date'] = st.date_input("Registration Date",
                    value=datetime.now().date()).isoformat()
            with col2:
                entity_data['address'] = st.text_input("Address", 
                    value=entity_data.get('address', '123 Business St'))
                entity_data['products_count'] = st.number_input("Products Count", 
                    min_value=0, value=entity_data.get('products_count', 50))
                    
            seller_features_input = st.text_area("Paste Seller Feature Vector (21 comma-separated values)", 
                                                value=','.join([str(round(0.05*i, 2)) for i in range(1, 22)]))
            if seller_features_input:
                try:
                    seller_features = [float(v.strip()) for v in seller_features_input.split(",")]
                    if len(seller_features) != 21:
                        st.error("❌ Please enter exactly 21 numeric values.")
                    else:
                        entity_data["seller_features"] = seller_features
                except:
                    st.error("❌ Invalid format. Provide 21 numbers separated by commas.")
        
        elif entity_type == "listing":
            entity_data['title'] = st.text_input("Listing Title", 
                value=entity_data.get('title', 'Product Listing Title'))
            entity_data['category'] = st.text_input("Category", 
                value=entity_data.get('category', 'General'))
            entity_data['variations'] = st.number_input("Number of Variations", 
                min_value=0, value=entity_data.get('variations', 1))
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                include_lifecycle = st.checkbox("Include Lifecycle Analysis", value=True)
                include_predictions = st.checkbox("Include Fraud Predictions", value=True)
            with col2:
                force_cross_intel = st.checkbox("Force Cross-Intelligence Analysis", value=False)
                detailed_evidence = st.checkbox("Request Detailed Evidence", value=True)
        
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze for Fraud", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing fraud detection..."):
                    # Prepare request
                    request_data = {
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "entity_data": entity_data,
                        "priority": priority
                    }
                    
                    # Make API request
                    start_time = time.time()
                    result = make_api_request("/detect", method="POST", data=request_data)
                    processing_time = time.time() - start_time
                    
                    if result:
                        # Store results
                        st.session_state['last_result'] = result
                        st.session_state['last_request'] = request_data
                        
                        # Add to history
                        st.session_state.test_history.append({
                            'entity_type': entity_type,
                            'entity_id': entity_id,
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'fraud_score': result.get('fraud_score', 0),
                            'result': result
                        })
                        
                        # Show success message
                        if result.get('fraud_detected', False):
                            st.error(f"🚨 FRAUD DETECTED - Score: {result.get('fraud_score', 0):.1%}")
                        else:
                            st.success(f"✅ Entity appears legitimate - Score: {result.get('fraud_score', 0):.1%}")
                        
                        st.info(f"⏱️ Processing completed in {processing_time:.2f} seconds")
                        
                        # Auto-switch to results tab
                        st.balloons()
    
    with tab2:
        st.header("Detection Results Analysis")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            request = st.session_state['last_request']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                trust_score = result.get('trust_score', 0)
                trust_color = "🟢" if trust_score > 75 else "🟡" if trust_score > 50 else "🔴"
                st.metric("Trust Score", f"{trust_color} {trust_score:.0f}/100")
            
            with col2:
                fraud_score = result.get('fraud_score', 0)
                fraud_color = "🔴" if fraud_score > 0.7 else "🟡" if fraud_score > 0.4 else "🟢"
                st.metric("Fraud Risk", f"{fraud_color} {fraud_score:.1%}")
            
            with col3:
                fraud_status = "🚨 FRAUD" if result.get('fraud_detected', False) else "✅ CLEAN"
                st.metric("Status", fraud_status)
            
            with col4:
                st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
            
            # Dual gauge chart
            st.plotly_chart(
                create_trust_score_gauge(trust_score, fraud_score),
                use_container_width=True
            )
            
            # Detector breakdown
            if 'detector_results' in result:
                st.subheader("🔍 Detection Component Analysis")
                
                # Chart
                fig = create_detector_results_chart(result['detector_results'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Evidence table
                with st.expander("📋 View Detailed Evidence", expanded=True):
                    display_evidence(result['detector_results'])
            
            # Fraud types detected
            fraud_types = set()
            for detector_result in result.get('detector_results', {}).values():
                fraud_types.update(detector_result.get('fraud_types', []))
            
            if fraud_types:
                st.subheader("🚨 Fraud Types Detected")
                cols = st.columns(min(len(fraud_types), 4))
                for i, fraud_type in enumerate(fraud_types):
                    with cols[i % len(cols)]:
                        st.error(f"• {fraud_type.replace('_', ' ').title()}")
            
            # Recommendations
            if result.get('recommendations'):
                st.subheader("💡 Recommended Actions")
                for i, rec in enumerate(result['recommendations']):
                    st.info(f"{i+1}. {rec}")
            
            # Priority actions
            if result.get('priority_actions'):
                st.subheader("⚡ Priority Actions Required")
                actions_df = pd.DataFrame(result['priority_actions'])
                st.dataframe(actions_df, use_container_width=True, hide_index=True)
        
        else:
            st.info("👆 Submit an entity for detection to see results")
    
    with tab3:
        st.header("Cross-Intelligence Network Analysis")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            if result.get('cross_intel_triggered'):
                # Network metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    network_size = result.get('network_size', 0)
                    st.metric("Network Size", f"{network_size:,} entities")
                
                with col2:
                    network_type = result.get('network_type', 'Unknown')
                    st.metric("Network Type", network_type)
                
                with col3:
                    financial_impact = 0
                    if 'cross_intel_result' in result:
                        financial_impact = result['cross_intel_result'].get('financial_impact', 0)
                    st.metric("Est. Financial Impact", f"${financial_impact:,.2f}")
                
                with col4:
                    confidence = 0
                    if 'cross_intel_result' in result:
                        confidence = result['cross_intel_result'].get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Network visualization
                if 'cross_intel_result' in result:
                    fig = create_network_visualization(result['cross_intel_result'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Key players
                if result.get('cross_intel_result', {}).get('key_players'):
                    st.subheader("🎯 Key Players Identified")
                    key_players = result['cross_intel_result']['key_players']
                    
                    cols = st.columns(min(len(key_players), 3))
                    for i, player in enumerate(key_players[:9]):  # Limit to 9
                        with cols[i % len(cols)]:
                            st.warning(f"• {player}")
                
                # Network indicators
                if result.get('cross_intel_result', {}).get('network_indicators'):
                    st.subheader("🔍 Network Indicators")
                    for indicator in result['cross_intel_result']['network_indicators']:
                        st.write(f"• {indicator}")
            
            else:
                st.info("📊 No fraud network detected")
                st.metric("Cross-Intel Threshold", "50%")
                st.write("Cross-intelligence analysis is triggered when fraud score exceeds 50%")
        
        else:
            st.info("👆 Submit an entity for detection to see network analysis")
    
    with tab4:
        st.header("Lifecycle Analysis & Predictions")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            # Lifecycle Analysis
            if 'lifecycle_stage' in result and result['lifecycle_stage']:
                st.subheader("📊 Entity Lifecycle Analysis")
                display_lifecycle_analysis({
                    'current_stage': result['lifecycle_stage'],
                    'risk_level': 'high' if result.get('fraud_score', 0) > 0.7 else 'medium',
                    'indicators': []
                })
            
            # Predictions
            if 'predictions' in result and result['predictions']:
                st.subheader("🔮 Fraud Predictions")
                
                predictions_df = pd.DataFrame(result['predictions'])
                if not predictions_df.empty:
                    st.dataframe(
                        predictions_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'probability': st.column_config.ProgressColumn(
                                'Probability',
                                help='Likelihood of this prediction',
                                format='%.1f%%',
                                min_value=0,
                                max_value=100,
                            ),
                        }
                    )
                else:
                    st.info("No predictions available")
            
            # Decision tracking
            if 'decision_id' in result and result['decision_id']:
                st.subheader("📝 Decision Tracking")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input("Decision ID", value=result['decision_id'], disabled=True)
                with col2:
                    st.info("This decision has been recorded in the Memory Bank")
            
        else:
            st.info("👆 Submit an entity for detection to see lifecycle analysis")
    
    with tab5:
        st.header("Raw API Response")
        
        if 'last_result' in st.session_state:
            # Request details
            with st.expander("📤 Request Details", expanded=False):
                st.json(st.session_state['last_request'])
            
            # Response details
            with st.expander("📥 Response Details", expanded=True):
                st.json(st.session_state['last_result'])
            
            # Copy button
            if st.button("📋 Copy Response to Clipboard"):
                st.write("```json")
                st.code(json.dumps(st.session_state['last_result'], indent=2))
                st.write("```")
        
        else:
            st.info("👆 Submit an entity for detection to see raw data")

# ============= Run App =============

if __name__ == "__main__":
    # Custom footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>TrustSight Testing Suite v1.0 | "
        "Built for comprehensive fraud detection testing</p>",
        unsafe_allow_html=True
    )
    
    main()