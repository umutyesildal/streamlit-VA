import streamlit as st
from services.data_loader import load_data
from services.state_management import initialize_session_state, train_and_store_model, is_model_trained
from services.model import evaluate_model
from components.shared_sidebar import show_shared_sidebar

def get_available_classes(y_data):
    """Get available classes in the correct order"""
    custom_order = ["very_low", "low", "moderate", "high", "very_high"]
    all_classes = sorted(y_data['x'].unique())
    return [c for c in custom_order if c in all_classes]

def main():
    # Basic setup
    st.set_page_config(page_title="VA Streamlit App", page_icon="👋")
    initialize_session_state()
    
    # Load data
    X_data, y_data = load_data()
    
    # Show shared sidebar with training button
    with st.sidebar:
        if not is_model_trained():
            st.markdown("### Model Training")
            if st.button("Train Random Forest Model 🤖"):
                with st.spinner("Training model..."):
                    train_and_store_model(X_data, y_data)
                st.success("✅ Training Complete!")
    
    # Show shared sidebar content
    if is_model_trained():
        selected_classes = show_shared_sidebar(y_data)
    
    # Main content
    st.title("VA Lucas Carbon Classification with Streamlit")
    
    # Show dataset info
    st.write("### Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Features Shape**: {X_data.shape[0]} rows × {X_data.shape[1]} columns")
    with col2:
        st.info(f"**Target Shape**: {y_data.shape[0]} rows × {y_data.shape[1]} columns")
    
    # Description
    st.markdown("""
        This application uses a **Random Forest Classifier** to predict carbon levels in soil samples. 
        The model analyzes various spectral features to classify samples into different carbon level categories.
        
        ### Key Features:
        - **Interactive Visualizations**: Explore model performance through various plots and metrics
        - **Class Filtering**: Filter results by specific carbon level classes
        - **Feature Analysis**: Investigate feature importance and their impacts
        
        ### Available Pages:
        1. **Confusion Matrix & Metrics**: View model performance through confusion matrices and detailed metrics
        2. **Classwise Analysis**: Analyze per-class performance and distribution
        3. **Feature Analysis**: Explore feature importance and interval impacts
    """)
    
    # Evaluation results
    if is_model_trained():
        st.markdown("---")
        st.markdown("### 📊 Model Evaluation Results")
        
        eval_col1, eval_col2 = st.columns([3, 2])
        with eval_col1:
            st.markdown("#### Model Performance Metrics")
            evaluate_model(st.session_state.rf_model, 
                         st.session_state.X_test, 
                         st.session_state.y_test)
        
        with eval_col2:
            st.markdown("#### Next Steps")
            st.markdown("""
                ✨ Model is now ready! You can:
                
                🔍 **Explore Results:**
                - Filter specific classes using sidebar
                - View detailed visualizations
                
                📊 **Available Pages:**
                1. Confusion Matrix Analysis
                2. Classwise Performance
                3. Feature Importance Study
                
                💡 *Tip: Start with Confusion Matrix page*
            """)

if __name__ == "__main__":
    main()
