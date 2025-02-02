import streamlit as st
from services.state_management import initialize_session_state, get_filtered_predictions, is_model_trained
from services.plotting import (
    plot_classwise_metrics,
    plot_actual_vs_predicted_distributions
)
from services.data_loader import load_data
from components.shared_sidebar import show_shared_sidebar

def main():
    st.title("Classwise Analysis")
    initialize_session_state()
    
    if not is_model_trained():
        st.warning("Please train the model first in the main page.")
        return
        
    # Load data for sidebar
    _, y_data = load_data()
    
    # Show shared sidebar
    selected_classes = show_shared_sidebar(y_data)
    
    # Filter based on selected classes
    if not selected_classes:
        st.warning("Please select classes in the sidebar of the homepage.")
        return
        
    y_test_filtered, y_pred_filtered = get_filtered_predictions(selected_classes)
    
    if y_test_filtered is None or y_pred_filtered is None:
        st.error("No filtered data available.")
        return
        
    st.subheader("Classwise Metrics")
    plot_classwise_metrics(y_test_filtered, y_pred_filtered, st.session_state.rf_model)
    
    st.subheader("Actual vs Predicted Distribution")
    custom_order = ["very_low", "low", "moderate", "high", "very_high"]
    plot_actual_vs_predicted_distributions(y_test_filtered, y_pred_filtered, custom_order)

if __name__ == "__main__":
    main()
