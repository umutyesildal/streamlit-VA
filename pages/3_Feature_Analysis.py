import streamlit as st
from services.state_management import initialize_session_state, is_model_trained
from services.plotting import plot_feature_importance
from services.data_loader import load_data
from components.shared_sidebar import show_shared_sidebar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

def substitute_and_calculate_accuracy(feature, replacement_value, model, X, y):
    """Matches the implementation in feature.py"""
    X_modified = X.copy()
    X_modified[feature] = replacement_value
    y_pred_modified = model.predict(X_modified)
    return accuracy_score(y, y_pred_modified)

def interval_permutation_importance(feature, model, X, y, num_intervals=20):
    """Exact implementation from feature.py"""
    intervals = pd.qcut(X[feature], q=num_intervals, duplicates='drop')
    interval_labels = []
    importance_scores = []
    baseline_accuracy = accuracy_score(y, model.predict(X))

    for interval in intervals.unique():
        X_modified_p1 = X.copy()
        X_modified_p2 = X.copy()
        
        mid_left = interval.left
        mid_right = interval.right
        
        X_modified_p1[feature] = mid_left
        X_modified_p2[feature] = mid_right
        
        accuracy_p1 = accuracy_score(y, model.predict(X_modified_p1))
        accuracy_p2 = accuracy_score(y, model.predict(X_modified_p2))
        
        avg_importance = (baseline_accuracy - accuracy_p1 + baseline_accuracy - accuracy_p2) / 2
        importance_scores.append(avg_importance)
        interval_labels.append(f"{mid_left:.2e} - {mid_right:.2e}")
    
    return interval_labels, importance_scores

def format_scientific_to_simple(value):
    """Convert scientific notation to simplified number with K/M/B suffix"""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
            
    if abs(value) >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}K"
    elif abs(value) >= 1:
        return f"{value:.1f}"
    elif abs(value) >= 1e-3:
        return f"{value*1e3:.1f}m"  # milli
    elif abs(value) >= 1e-6:
        return f"{value*1e6:.1f}μ"  # micro
    elif abs(value) >= 1e-9:
        return f"{value*1e9:.1f}n"  # nano
    else:
        return f"{value:.2e}"

def plot_feature_importance_scatter(model, X):
    """Create scatter plot for all feature importances"""
    importances = pd.Series(model.feature_importances_, index=X.columns)
    
    # Convert column names to float for proper x-axis values
    wavelengths = X.columns.astype(float)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    scatter = ax.scatter(
        x=wavelengths,  # Use actual wavelength values
        y=importances,
        alpha=0.6,
        c=importances,
        cmap='viridis',
        s=50
    )
    
    plt.colorbar(scatter, label='Importance Score')
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Importance Score')
    ax.set_title('Feature Importance Distribution by Wavelength')
    
    # Set proper x-axis limits and ticks
    ax.set_xlim(min(wavelengths), max(wavelengths))
    ax.xaxis.set_major_locator(plt.MultipleLocator(200))  # Major ticks every 200nm
    
    ax.grid(True, alpha=0.3)
    return fig

def plot_2d_heatmap(feature1, feature2, model, X, y, bins=10):
    """Modified 2D heatmap with better number formatting"""
    min1, max1 = X[feature1].min(), X[feature1].max()
    min2, max2 = X[feature2].min(), X[feature2].max()

    x_edges = np.linspace(min1, max1, bins + 1)
    y_edges = np.linspace(min2, max2, bins + 1)
    heatmap = np.zeros((bins, bins))

    # Calculate heatmap values
    for i in range(bins):
        for j in range(bins):
            X_copy = X.copy()
            X_copy[feature1] = (x_edges[i] + x_edges[i + 1]) / 2
            X_copy[feature2] = (y_edges[j] + y_edges[j + 1]) / 2
            y_pred = model.predict(X_copy)
            heatmap[i, j] = accuracy_score(y, y_pred)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Format x and y tick labels
    x_ticks = [format_scientific_to_simple((x_edges[i] + x_edges[i+1])/2) for i in range(len(x_edges)-1)]
    y_ticks = [format_scientific_to_simple((y_edges[i] + y_edges[i+1])/2) for i in range(len(y_edges)-1)]
    
    sns.heatmap(
        heatmap,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        xticklabels=x_ticks,
        yticklabels=y_ticks,
        annot_kws={'size': 8}
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    ax.set_xlabel(f'Wavelength {feature1} (nm)')
    ax.set_ylabel(f'Wavelength {feature2} (nm)')
    ax.set_title('2D Feature Interaction Heatmap')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    return fig

def get_top_features(model, X, n=2):
    """Get the top n most important features"""
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importances.head(n).index.tolist()

def main():
    st.title("Feature Analysis")
    initialize_session_state()
    
    # Load data for sidebar
    _, y_data = load_data()
    selected_classes = show_shared_sidebar(y_data)
    
    if not is_model_trained():
        st.warning("Please train the model first in the main page.")
        return

    # Feature Importance Analysis
    st.header("Feature Importances")
    
    # Add tabs for different visualizations
    tab1, tab2 = st.tabs(["Top 20 Features", "All Features"])
    
    with tab1:
        # Existing top 20 features bar plot
        top_features = get_top_features(st.session_state.rf_model_encoded, 
                                      st.session_state.X_train_encoded, n=20)
    
        # Plot top 20 features
        feature_importances = pd.Series(
            st.session_state.rf_model_encoded.feature_importances_,
            index=st.session_state.X_train_encoded.columns
        ).sort_values(ascending=False)
    
        fig, ax = plt.subplots(figsize=(16, 6))
        feature_importances.head(20).plot(kind='bar', ax=ax)
        ax.set_title('Top 20 Features (Spectral Wavelengths)')
        ax.set_xlabel('Features (Spectral Wavelengths)')
        ax.set_ylabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # New scatter plot for all features
        fig = plot_feature_importance_scatter(
            st.session_state.rf_model_encoded,
            st.session_state.X_train_encoded
        )
        st.pyplot(fig)

    # Feature Selection for Analysis
    st.header("Feature Analysis")
    selected_feature = st.selectbox(
        "Select feature for analysis",
        options=top_features,
        index=0
    )

    # Substitution Analysis
    st.subheader("Substitution Analysis")
    min_value = st.session_state.X_test_encoded[selected_feature].min()
    mean_value = st.session_state.X_test_encoded[selected_feature].mean()
    max_value = st.session_state.X_test_encoded[selected_feature].max()
    
    st.write(f"Feature '{selected_feature}' values:")
    st.write(f"- Minimum: {min_value:.4f}")
    st.write(f"- Mean: {mean_value:.4f}")
    st.write(f"- Maximum: {max_value:.4f}")
    
    replacement_values = [min_value, mean_value, max_value]
    accuracies = [
        substitute_and_calculate_accuracy(
            selected_feature, v, 
            st.session_state.rf_model_encoded,
            st.session_state.X_test_encoded,
            st.session_state.y_test_encoded
        ) for v in replacement_values
    ]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(["Min", "Mean", "Max"], accuracies, marker='o', linestyle='-', color='b')
    ax.set_title(f"Substitution Accuracy Score for {selected_feature}")
    ax.set_xlabel('Fixed Values')
    ax.set_ylabel('Accuracy Score')
    ax.grid(True)
    st.pyplot(fig)
    
    # Interval-Based Feature Importance
    st.subheader("Interval-Based Feature Importance")
    interval_labels, importance_scores = interval_permutation_importance(
        selected_feature,
        st.session_state.rf_model_encoded,
        st.session_state.X_test_encoded,
        st.session_state.y_test_encoded
    )
    
    # Improve interval label formatting
    simplified_labels = []
    for label in interval_labels:
        try:
            start, end = label.split('-')
            start_val = float(start.strip())
            end_val = float(end.strip())
            # Format wavelength values in nanometers
            simplified_labels.append(
                f"{int(start_val)} - {int(end_val)} nm"
            )
        except (ValueError, IndexError):
            simplified_labels.append(label)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(simplified_labels, importance_scores, color='orange')
    ax.set_xlabel('Wavelength Intervals (nm)')
    ax.set_ylabel('Permutation-Based Feature Importance')
    ax.set_title(f'Interval-Based Feature Importance Score for {selected_feature} nm')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # 2D Feature Interaction
    st.header("2D Feature Interaction")
    st.write("Select two features for interaction analysis (default: top 2 important features)")
    
    # Get top 2 most important features as defaults
    default_features = get_top_features(st.session_state.rf_model_encoded, 
                                      st.session_state.X_train_encoded, n=2)
    
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Select first feature", 
                              options=top_features,
                              index=top_features.index(default_features[0]))
    with col2:
        feature2 = st.selectbox("Select second feature", 
                              options=top_features,
                              index=top_features.index(default_features[1]))
    
    if feature1 != feature2:
        fig = plot_2d_heatmap(
            feature1, feature2,
            st.session_state.rf_model_encoded,
            st.session_state.X_test_encoded,
            st.session_state.y_test_encoded
        )
        st.pyplot(fig)
    else:
        st.warning("Please select different features for interaction analysis")

if __name__ == "__main__":
    main()
