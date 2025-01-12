import streamlit as st

def show_shared_sidebar(y_data):
    """Shared sidebar component for all pages"""
    with st.sidebar:
        st.title("Settings")
        
        # Get available classes in correct order
        custom_order = ["very_low", "low", "moderate", "high", "very_high"]
        all_classes = sorted(y_data['x'].unique())
        available_classes = [c for c in custom_order if c in all_classes]
        
        # Initialize global filter if not exists
        if "global_class_filter" not in st.session_state:
            st.session_state.global_class_filter = available_classes
        
        # Class selection
        st.markdown("### Class Selection")
        selected_classes = st.multiselect(
            "Select classes to include:",
            options=available_classes,
            default=st.session_state.global_class_filter,
            key="class_selector"
        )
        
        # Update global filter
        st.session_state.global_class_filter = selected_classes if selected_classes else available_classes
        
        return st.session_state.global_class_filter
