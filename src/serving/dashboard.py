"""
Streamlit Dashboard for Phase 6 Store Sales Forecasting
Interactive web interface for Pattern-Based Selection methodology
"""

import sys
from pathlib import Path
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from serving.config import config

# Page configuration
st.set_page_config(
    page_title="Store Sales Forecasting - Phase 6",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


class DashboardAPI:
    """API client for dashboard"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    
    def get_cases(self) -> Dict[str, Any]:
        """Get evaluation cases"""
        try:
            response = requests.get(f"{self.base_url}/cases", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_analysis(self, store_nbr: int, family: str) -> Dict[str, Any]:
        """Get pattern analysis"""
        try:
            response = requests.get(
                f"{self.base_url}/analysis/{store_nbr}/{family}",
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_prediction(self, store_nbr: int, family: str, horizon: int = 15) -> Dict[str, Any]:
        """Get prediction"""
        try:
            response = requests.post(
                f"{self.base_url}/predict/{store_nbr}/{family}",
                json={"forecast_horizon": horizon, "include_analysis": True},
                timeout=120  # Increased to 2 minutes
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": "Prediction generation timed out (120s). This store-family combination may require more processing time."}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection failed. Make sure the Flask API is running on port 8000."}
        except requests.exceptions.HTTPError as e:
            return {"error": f"API error: {e.response.status_code} - {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary"""
        try:
            response = requests.get(f"{self.base_url}/dashboard/summary", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_performance(self) -> Dict[str, Any]:
        """Get model performance"""
        try:
            response = requests.get(f"{self.base_url}/models/performance", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


# Initialize API client
@st.cache_resource
def get_api_client():
    return DashboardAPI(config.api_url)


def render_header():
    """Render dashboard header"""
    st.title("🏪 Store Sales Forecasting - Phase 6")
    st.markdown("""
    **Pattern-Based Selection with Adaptive Model Routing**
    
    This dashboard demonstrates the Phase 6 implementation that uses Coefficient of Variation (CV) analysis 
    to intelligently route forecasting tasks between Neural (LSTM) and Traditional (ARIMA) models.
    """)
    
    # API status check
    api = get_api_client()
    health = api.health_check()
    
    if health.get("status") == "healthy":
        st.success(f"✅ API Connected - Version {health.get('version', 'Unknown')}")
    else:
        st.error(f"❌ API Connection Failed: {health.get('detail', 'Unknown error')}")
        st.stop()


def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.header("🎛️ Controls")
    
    # Load evaluation cases
    api = get_api_client()
    cases_data = api.get_cases()
    
    if "error" in cases_data:
        st.sidebar.error(f"Failed to load cases: {cases_data['error']}")
        return None, None, None
    
    cases = cases_data.get("cases", [])
    
    if not cases:
        st.sidebar.warning("No evaluation cases available")
        return None, None, None
    
    # Case selection
    st.sidebar.subheader("📋 Select Store-Family")
    
    case_options = []
    for case in cases:
        display_name = f"Store {case['store_nbr']} - {case['family']}"
        quality = case.get('quality_score', 0)
        case_options.append({
            'display': f"{display_name} (Q: {quality:.1f})",
            'store_nbr': case['store_nbr'],
            'family': case['family'],
            'quality': quality
        })
    
    # Sort by quality score
    case_options.sort(key=lambda x: x['quality'], reverse=True)
    
    selected_option = st.sidebar.selectbox(
        "Choose evaluation case:",
        options=case_options,
        format_func=lambda x: x['display'],
        index=0
    )
    
    # Add performance indicator
    if selected_option:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏱️ Expected Processing Time")
        
        # Make a quick pattern analysis to estimate time
        try:
            api = get_api_client()
            analysis_data = api.get_analysis(selected_option['store_nbr'], selected_option['family'])
            
            if "error" not in analysis_data:
                pattern_type = analysis_data.get('pattern_type', 'UNKNOWN')
                cv = analysis_data.get('coefficient_variation', 0)
                
                if pattern_type == "REGULAR":
                    st.sidebar.success("🏃‍♂️ **Fast** (~5-15 seconds)\nUses Neural models (LSTM)")
                elif pattern_type == "VOLATILE":
                    st.sidebar.warning("⏳ **Moderate** (~30-60 seconds)\nUses Traditional models (ARIMA)")
                else:
                    st.sidebar.info("❓ **Unknown processing time**")
                    
                st.sidebar.caption(f"Pattern: {pattern_type} (CV: {cv:.3f})")
            else:
                st.sidebar.info("❓ Could not estimate processing time")
                
        except Exception:
            st.sidebar.info("❓ Could not estimate processing time")
    
    # Forecast horizon
    forecast_horizon = st.sidebar.slider(
        "📅 Forecast Horizon (days):",
        min_value=1,
        max_value=30,
        value=15,
        help="Number of days to forecast"
    )
    
    # Analysis options
    st.sidebar.subheader("⚙️ Options")
    show_pattern_details = st.sidebar.checkbox("Show Pattern Analysis Details", value=True)
    show_performance_comparison = st.sidebar.checkbox("Show Performance Comparison", value=True)
    
    return selected_option, forecast_horizon, {
        'show_pattern_details': show_pattern_details,
        'show_performance_comparison': show_performance_comparison
    }


def render_overview_metrics():
    """Render overview metrics"""
    st.header("📊 System Overview")
    
    api = get_api_client()
    summary = api.get_dashboard_summary()
    performance = api.get_performance()
    
    if "error" not in summary and "error" not in performance:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cases",
                summary.get('total_evaluation_cases', 0),
                help="Total evaluation cases available"
            )
        
        with col2:
            champion_rmsle = performance.get('champion_result', 0.4190)
            st.metric(
                "Champion RMSLE",
                f"{champion_rmsle:.4f}",
                help="Phase 3.6 champion result"
            )
        
        with col3:
            improvement = performance.get('improvement_vs_traditional', 0)
            st.metric(
                "vs Traditional",
                f"+{improvement:.1f}%",
                help="Improvement over Phase 2 baseline"
            )
        
        with col4:
            improvement_neural = performance.get('improvement_vs_neural', 0)
            st.metric(
                "vs Neural",
                f"+{improvement_neural:.1f}%",
                help="Improvement over Phase 3 baseline"
            )


def create_pattern_analysis_chart(analysis_data: Dict[str, Any]):
    """Create pattern analysis visualization"""
    
    # CV threshold visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Coefficient of Variation Analysis',
            'Pattern Classification',
            'Time Series Characteristics',
            'Model Routing Decision'
        ],
        specs=[[{"type": "indicator"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    cv_value = analysis_data.get('coefficient_variation', 0)
    threshold = 1.5
    pattern_type = analysis_data.get('pattern_type', 'UNKNOWN')
    confidence = analysis_data.get('confidence_score', 0)
    
    # CV Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=cv_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"CV = {cv_value:.3f}"},
            delta={'reference': threshold},
            gauge={
                'axis': {'range': [None, 3]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, threshold], 'color': "lightgreen"},
                    {'range': [threshold, 3], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ),
        row=1, col=1
    )
    
    # Pattern classification pie
    colors = ['#00cc44' if pattern_type == 'REGULAR' else '#ff6b6b', '#cccccc']
    fig.add_trace(
        go.Pie(
            labels=['Selected Pattern', 'Other'],
            values=[confidence, 1-confidence],
            marker_colors=colors,
            title=f"{pattern_type}<br>Confidence: {confidence:.1%}"
        ),
        row=1, col=2
    )
    
    # Time series characteristics
    characteristics = {
        'Zero Sales %': analysis_data.get('zero_sales_percentage', 0),
        'Seasonal Strength': analysis_data.get('seasonal_strength', 0) * 100,
        'Trend Strength': analysis_data.get('trend_strength', 0) * 100
    }
    
    fig.add_trace(
        go.Bar(
            x=list(characteristics.keys()),
            y=list(characteristics.values()),
            marker_color=['#ff7f0e', '#2ca02c', '#d62728']
        ),
        row=2, col=1
    )
    
    # Model routing decision
    recommended_model = analysis_data.get('recommended_model_type', 'UNKNOWN')
    model_color = "#00cc44" if recommended_model == "NEURAL" else "#1f77b4"
    
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=1,
            title={'text': f"Selected Model<br>{recommended_model}"},
            number={'font': {'color': model_color, 'size': 40}}
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Pattern Analysis Dashboard")
    return fig


def create_prediction_chart(prediction_data: Dict[str, Any]):
    """Create prediction visualization"""
    
    predictions = prediction_data.get('predictions', [])
    actuals = prediction_data.get('actuals', [])
    
    if not predictions or not actuals:
        st.warning("No prediction data available for visualization")
        return None
    
    # Create time series for visualization
    days = list(range(1, len(predictions) + 1))
    
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=days,
            y=actuals,
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        )
    )
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=days,
            y=predictions,
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8)
        )
    )
    
    # Add performance metrics as annotations
    rmsle = prediction_data.get('test_rmsle', 0)
    mae = prediction_data.get('test_mae', 0)
    model_name = prediction_data.get('selected_model_name', 'Unknown')
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Model: {model_name}<br>RMSLE: {rmsle:.4f}<br>MAE: {mae:.2f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f"Sales Forecast - Store {prediction_data.get('store_nbr')} - {prediction_data.get('family')}",
        xaxis_title="Forecast Day",
        yaxis_title="Sales Volume",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_performance_comparison_chart(prediction_data: Dict[str, Any]):
    """Create performance comparison chart"""
    
    current_rmsle = prediction_data.get('test_rmsle', 0)
    traditional_baseline = 0.4755
    neural_baseline = 0.5466
    
    models = ['Traditional\nBaseline', 'Neural\nBaseline', 'Phase 6\nCurrent']
    rmsle_values = [traditional_baseline, neural_baseline, current_rmsle]
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    # Determine best performer
    best_idx = rmsle_values.index(min(rmsle_values))
    colors[best_idx] = '#00cc44'  # Highlight best performer
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=rmsle_values,
            marker_color=colors,
            text=[f'{val:.4f}' for val in rmsle_values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="RMSLE Performance Comparison",
        yaxis_title="RMSLE (Lower is Better)",
        height=400,
        showlegend=False
    )
    
    return fig


def main():
    """Main dashboard function"""
    render_header()
    
    # Sidebar controls
    selected_case, forecast_horizon, options = render_sidebar()
    
    if selected_case is None:
        st.error("Cannot proceed without evaluation cases")
        return
    
    # Overview metrics
    render_overview_metrics()
    
    # Main content
    store_nbr = selected_case['store_nbr']
    family = selected_case['family']
    
    st.header(f"🎯 Analysis: Store {store_nbr} - {family}")
    
    # Get prediction with enhanced progress feedback
    with st.spinner(f"Generating predictions for Store {store_nbr} - {family}... This may take 1-2 minutes for complex cases."):
        api = get_api_client()
        
        # Show progress message
        progress_placeholder = st.empty()
        progress_placeholder.info("🧠 Analyzing time series patterns and selecting optimal model...")
        
        prediction_data = api.get_prediction(store_nbr, family, forecast_horizon)
        progress_placeholder.empty()
    
    if "error" in prediction_data:
        st.error(f"❌ Prediction Failed: {prediction_data['error']}")
        
        # Provide helpful suggestions
        st.info("""
        **Troubleshooting Tips:**
        - Try selecting a different store-family combination from the sidebar
        - Some combinations may require more processing time
        - Check that the Flask API is running properly
        - Consider reducing the forecast horizon if the issue persists
        """)
        
        # Show alternative cases suggestion based on pattern type
        st.subheader("🔄 Alternative Cases to Try")
        
        # Get pattern analysis to recommend similar complexity cases
        try:
            api = get_api_client()
            cases_data = api.get_cases()
            if "cases" not in cases_data:
                cases_data["cases"] = []
                
            st.info("**Fast Cases (Neural routing - REGULAR patterns):**")
            fast_cases = [
                {"store": 49, "family": "PET SUPPLIES", "type": "REGULAR"},
                {"store": 3, "family": "CLEANING", "type": "REGULAR"}, 
                {"store": 47, "family": "BOOKS", "type": "REGULAR"}
            ]
            
            col1, col2, col3 = st.columns(3)
            for i, case in enumerate(fast_cases):
                with [col1, col2, col3][i]:
                    if st.button(f"🏃‍♂️ Store {case['store']}\n{case['family'][:12]}...", key=f"fast_{case['store']}"):
                        st.rerun()
            
            st.info("**Complex Cases (Traditional routing - VOLATILE patterns):**")
            complex_cases = [
                {"store": 44, "family": "SCHOOL AND OFFICE SUPPLIES", "type": "VOLATILE"},
                {"store": 45, "family": "HARDWARE", "type": "VOLATILE"}
            ]
            
            st.warning("⚠️ These cases use Traditional models (ARIMA) and may take 1-2 minutes")
            
            for case in complex_cases:
                if st.button(f"⏳ Store {case['store']} - {case['family']}", key=f"complex_{case['store']}"):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Could not load alternative cases: {e}")
        
        return
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction chart
        pred_chart = create_prediction_chart(prediction_data)
        if pred_chart:
            st.plotly_chart(pred_chart, use_container_width=True)
    
    with col2:
        # Key metrics
        st.subheader("📈 Key Metrics")
        
        rmsle = prediction_data.get('test_rmsle', 0)
        mae = prediction_data.get('test_mae', 0)
        mape = prediction_data.get('test_mape', 0)
        
        st.metric("RMSLE", f"{rmsle:.4f}")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("MAPE", f"{mape:.1f}%")
        
        # Model info
        st.subheader("🤖 Model Selection")
        model_type = prediction_data.get('selected_model_type', 'Unknown')
        model_name = prediction_data.get('selected_model_name', 'Unknown')
        
        st.info(f"**Type:** {model_type}")
        st.info(f"**Model:** {model_name}")
        
        # Baseline comparison
        beats_traditional = prediction_data.get('beats_traditional', False)
        beats_neural = prediction_data.get('beats_neural', False)
        
        if beats_traditional and beats_neural:
            st.success("🏆 Beats both baselines!")
        elif beats_traditional:
            st.success("✅ Beats traditional baseline")
        elif beats_neural:
            st.success("✅ Beats neural baseline")
        else:
            st.warning("❌ Needs improvement")
    
    # Pattern analysis details
    if options['show_pattern_details']:
        st.header("🔍 Pattern Analysis Details")
        
        pattern_analysis = prediction_data.get('pattern_analysis', {})
        if pattern_analysis:
            pattern_chart = create_pattern_analysis_chart(pattern_analysis)
            st.plotly_chart(pattern_chart, use_container_width=True)
            
            # Additional details
            with st.expander("📊 Raw Pattern Metrics"):
                raw_metrics = pattern_analysis.get('raw_metrics', {})
                if raw_metrics:
                    df_metrics = pd.DataFrame([raw_metrics]).T
                    df_metrics.columns = ['Value']
                    st.dataframe(df_metrics)
    
    # Performance comparison
    if options['show_performance_comparison']:
        st.header("⚖️ Performance Comparison")
        
        perf_chart = create_performance_comparison_chart(prediction_data)
        st.plotly_chart(perf_chart, use_container_width=True)
        
        # Improvement details
        improvement_trad = prediction_data.get('improvement_vs_traditional_baseline', 0)
        improvement_neural = prediction_data.get('improvement_vs_neural_baseline', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "vs Traditional Baseline",
                f"{improvement_trad:+.1f}%",
                delta=improvement_trad
            )
        with col2:
            st.metric(
                "vs Neural Baseline", 
                f"{improvement_neural:+.1f}%",
                delta=improvement_neural
            )


if __name__ == "__main__":
    main()