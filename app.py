import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import train_random_forest, evaluate_model
from plotting import (
    show_confusion_matrix, show_confusion_matrix_normalized, show_metrics_table,
    plot_classwise_metrics,
    plot_actual_vs_predicted_distributions, plot_feature_importance, show_interval_impact,
     show_substitution_analysis, show_permutation_importance, show_interval_permutation_importance, plot_2d_heatmap
)
from utils import filter_classes, get_single_feature_importance

custom_order = ["very_low", "low", "moderate", "high", "very_high"]

def main():
    st.title("Random Forest Classification with Streamlit")

    # --- Setup session state variables ---
    if "rf_model" not in st.session_state:
        st.session_state["rf_model"] = None
    if "y_pred" not in st.session_state:
        st.session_state["y_pred"] = None

    # --- Load Data ---

    X_data, y_data = load_data()

    st.write("### Data Shapes")
    st.write("Features shape:", X_data.shape)
    st.write("Target shape:", y_data.shape)

    # --- Sidebar for Class Filter (Global) ---
    all_classes = sorted(y_data['x'].unique())
    st.sidebar.write("## Class Filter")

    # Intersect with whatever classes exist in y_data
    final_options = [c for c in custom_order if c in all_classes]

    selected_classes = st.sidebar.multiselect(
        "Select which classes to include in all plots:",
        options=final_options,  # in your custom order
        default=final_options
    )


    # ----------------------------------------------------------------
    # Button: Train the model and auto-run all plots once
    # ----------------------------------------------------------------
    if st.button("Train Random Forest"):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=42)
        rf_model = train_random_forest(X_train, y_train)
        feature_importances = rf_model.feature_importances_
        st.session_state["rf_model"] = rf_model
        st.session_state["y_pred"] = rf_model.predict(X_test)
        st.success("Model training complete. Automatically generating all plots...")

        # Evaluate Model (on full test set)
        evaluate_model(rf_model, X_test, y_test)

        # Filter for selected classes
        y_test_filtered, y_pred_filtered = filter_classes(y_test, st.session_state["y_pred"], selected_classes)


        # Filter for selected classes
        y_test_filtered, y_pred_filtered = filter_classes(y_test, st.session_state["y_pred"], selected_classes)

                                      
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix (Regular)")
            show_confusion_matrix(y_test_filtered, y_pred_filtered, rf_model, 
                             title="Confusion Matrix (Regular)", cmap='Reds')
        with col2:
            st.subheader("Normalized Confusion Matrix")
            show_confusion_matrix_normalized(y_test_filtered, y_pred_filtered, rf_model, 
                                         title="Normalized Confusion Matrix")
            
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Classwise Metrics")
            plot_classwise_metrics(y_test_filtered, y_pred_filtered, rf_model)
        with col4:
            st.subheader("Actual vs. Predicted Distribution")
            plot_actual_vs_predicted_distributions(y_test_filtered, y_pred_filtered, custom_order)    
            
            
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Metrics")
            show_metrics_table(y_test_filtered, y_pred_filtered, rf_model)
        with col6:
            st.subheader("Feature Importance")
            plot_feature_importance(rf_model, X_train, threshold=0.005)
            
        col7, col8 = st.columns(2)
        with col7:
            st.subheader("Interval Impact")
            show_interval_impact(rf_model, X_test, y_test, feature_importances, bins=10)
        with col8:
            st.subheader("2D Heatmap")
            plot_2d_heatmap(rf_model, X_test, y_test, feature1="2257.0", feature2="2252.0", bins=10)

        st.subheader("Substitution Analysis")
        show_substitution_analysis(rf_model, X_test, y_test, feature_name="2257.0")
        
        
        col11, col12 = st.columns(2)
        with col11:
            st.subheader("Interval Permutation Importance")
            show_interval_permutation_importance("2257.0", rf_model, X_test, y_test)
        with col12:
            st.subheader("2D Heatmap Interaction " )
            plot_2d_heatmap(rf_model, X_test, y_test, feature1="2257.0", feature2="2252.0")
        
    # ----------------------------------------------------------------
    # If the model is trained, present individual buttons for each plot
    # ----------------------------------------------------------------
    if st.session_state["rf_model"] is not None:

        st.write("---")
        st.write("## Individual Actions")
        
        if st.button("Confusion Matrix (Regular)"):
            show_confusion_matrix(y_test_filtered, y_pred_filtered, rf_model, 
                             title="Confusion Matrix (Regular)", cmap='Reds')
            
            
        if st.button("Normalized Confusion Matrix"):
            show_confusion_matrix_normalized(y_test_filtered, y_pred_filtered, rf_model, 
                                         title="Normalized Confusion Matrix")
            
        if st.button("Classwise Metrics"):
            plot_classwise_metrics(y_test_filtered, y_pred_filtered, rf_model)
            
        if st.button("Actual vs. Predicted Distribution"):
            plot_actual_vs_predicted_distributions(y_test_filtered, y_pred_filtered,custom_order)
            
        if st.button("Metrics"):
            show_metrics_table(y_test_filtered, y_pred_filtered, rf_model)
            
        if st.button("Feature Importance"):
            plot_feature_importance(rf_model, X_train, threshold=0.005)
            
            
        if st.button("Interval Impact"):
            show_interval_impact(rf_model, X_test, y_test, rf_model.feature_importances_, bins=10)
            
        if st.button("2D Heatmap"):
            plot_2d_heatmap(rf_model, X_test, y_test, rf_model.feature_importances_, bins=10)
            
        # Single Feature Importance Lookup
        st.subheader("Single Feature Importance Lookup")

        # This text input allows multiple features, separated by commas
        user_input = st.text_input(
            "Enter feature name(s) (comma-separated)",
            value=""
        )

        # Button to trigger the lookup
        if st.button("Get Single Feature Importance"):
            # Check if user typed anything
            if not user_input.strip():
                st.warning("Please enter at least one feature name.")
            else:
                # Split by comma, strip whitespace
                requested_features = [feat.strip() for feat in user_input.split(",")]

                # Loop through each requested feature and try to display its importance
                for feat in requested_features:
                    if feat in X_train.columns:
                        # Find importance by index
                        feature_index = X_train.columns.get_loc(feat)
                        importance_score = rf_model.feature_importances_[feature_index]
                        st.write(f"**{feat}**: {importance_score:.5f}")
                    else:
                        st.error(f"Feature '{feat}' not found in the dataset.")

        if st.button("Substitution Analysis"):
            chosen_feature = "2257.0"  # or from user input
            show_substitution_analysis(rf_model, X_test, y_test, feature_name=chosen_feature)

        if st.button("2D Feature Interaction Heatmap"):
            plot_2d_heatmap(rf_model, X_test, y_test, feature1="2257.0", feature2="2252.0", bins=10)
            
        if st.button("Interval Permutation Importance"):
            chosen_feature = "2257.0"
            show_interval_permutation_importance(rf_model, X_test, y_test, feature_name=chosen_feature, num_intervals=20)

        if st.button("Permutation Importance"):
            chosen_feature = "2257.0" 
            show_permutation_importance(rf_model, X_test, y_test, feature_name=chosen_feature, n_iter=10)

    else:
        st.info("Please train the Random Forest model first.")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()