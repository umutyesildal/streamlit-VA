import streamlit as st

def show_sidebar(y_data):
    """Global sidebar component for class selection"""
    if 'global_class_filter' not in st.session_state:
        st.session_state.global_class_filter = []
        
    custom_order = ["very_low", "low", "moderate", "high", "very_high"]
    all_classes = sorted(y_data['x'].unique())
    final_options = [c for c in custom_order if c in all_classes]
    
    selected = st.sidebar.multiselect(
        "Select classes to include:",
        options=final_options,
        default=final_options,
        key="global_class_filter"
    )
    
    # Add class distribution if classes are selected
    if selected:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### Class Distribution")
        class_counts = y_data[y_data['x'].isin(selected)]['x'].value_counts()
        for cls in selected:
            count = class_counts.get(cls, 0)
            st.sidebar.text(f"• {cls}: {count} samples")
    
    return selected
