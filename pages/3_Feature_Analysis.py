import streamlit as st
from services.state_management import initialize_session_state, is_model_trained
from services.plotting import (
    plot_feature_importance,
    show_interval_impact
)
from services.data_loader import load_data
from components.shared_sidebar import show_shared_sidebar

def main():
    st.title("Feature Analysis")
    initialize_session_state()
    
    # Load data for sidebar
    _, y_data = load_data()
    
    # Show shared sidebar
    selected_classes = show_shared_sidebar(y_data)
    
    if not is_model_trained():
        st.warning("Please train the model first in the main page.")
        return

    st.subheader("Feature Importance")
    plot_feature_importance(st.session_state.rf_model, 
                          st.session_state.X_train, 
                          threshold=0.005)
    
    st.subheader("Interval Impact")
    show_interval_impact(st.session_state.rf_model, 
                       st.session_state.X_test, 
                       st.session_state.y_test, 
                       st.session_state.rf_model.feature_importances_)

if __name__ == "__main__":
    main()
