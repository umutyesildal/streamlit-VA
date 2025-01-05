import streamlit as st

st.set_page_config(
    page_title="VA Streamlit App",
    page_icon="👋",
)

st.title("VA Lucas Carbon Classification with Streamlit") 
st.sidebar.success("Select a page above.")

st.markdown(
    """
    ## Overview
    In this application, we explore two different encoding approaches 
    (label-encoding and no encoding) when training a Random Forest model.

    - **Label Encoding**: Converts categorical classes into numerical codes. 
      This sometimes skews how the model interprets the ordinal relationships 
      between classes.
    - **No Encoding**: Treats the data differently (often using one-hot-like 
      transformations behind the scenes or numeric features only). 

    These distinct approaches can lead to variations in:
    - Model performance.
    - Feature importance as seen in Heatmaps.
    
    You may observe that some features appear more or less important depending 
    on the encoding strategy.

    Please check the different pages for:
    - Data Overview
    - Model Training / Evaluation
    - Heatmaps for Feature Importance
    - Confusion Matrix

    We hope you find interesting results as you navigate through the pages!
    """
)
