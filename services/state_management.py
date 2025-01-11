import streamlit as st
from services.model import train_random_forest
from services.utils import filter_classes
from sklearn.model_selection import train_test_split

def initialize_session_state():
    """Initialize all session state variables"""
    if "rf_model" not in st.session_state:
        st.session_state["rf_model"] = None
    if "X_train" not in st.session_state:
        st.session_state["X_train"] = None
    if "X_test" not in st.session_state:
        st.session_state["X_test"] = None
    if "y_train" not in st.session_state:
        st.session_state["y_train"] = None
    if "y_test" not in st.session_state:
        st.session_state["y_test"] = None
    if "y_pred" not in st.session_state:
        st.session_state["y_pred"] = None
    if "selected_classes" not in st.session_state:
        st.session_state["selected_classes"] = None
    if "global_class_filter" not in st.session_state:
        st.session_state["global_class_filter"] = None

def train_and_store_model(X_data, y_data):
    """Train model and store all necessary data in session state"""
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=42)
    rf_model = train_random_forest(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Store everything in session state
    st.session_state["rf_model"] = rf_model
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred

def get_filtered_predictions(selected_classes):
    """Get filtered predictions based on selected classes"""
    if not all(key in st.session_state for key in ["y_test", "y_pred"]):
        return None, None
        
    y_test_filtered, y_pred_filtered = filter_classes(
        st.session_state["y_test"], 
        st.session_state["y_pred"], 
        selected_classes
    )
    return y_test_filtered, y_pred_filtered

def is_model_trained():
    """Check if model is trained"""
    return st.session_state["rf_model"] is not None
