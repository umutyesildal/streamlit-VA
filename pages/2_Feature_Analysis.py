
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
import plotly.graph_objects as go
import plotly.express as px

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
    """Create interactive scatter plot for all feature importances using Plotly"""
    importances = pd.Series(model.feature_importances_, index=X.columns)
    wavelengths = X.columns.astype(float)
    
    fig = px.scatter(
        x=wavelengths,
        y=importances,
        color=importances,
        color_continuous_scale='Turbo',  # Changed to Turbo for better visibility
        labels={
            'x': 'Wavelength (nm)',
            'y': 'Importance Score',
            'color': 'Importance'
        },
        title='Feature Importance Distribution by Wavelength'
    )
    
    fig.update_traces(
        hovertemplate="<br>".join([
            "Wavelength: %{x:.1f} nm",
            "Importance: %{y:.6f}",
            "<extra></extra>"
        ]),
        marker=dict(size=8)
    )
    
    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title="Importance Score",
        showlegend=False,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  # White text
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',  # Subtle grid
            dtick=200  # Show grid every 200nm
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'  # Subtle grid
        )
    )
    
    return fig

def plot_2d_heatmap(feature1, feature2, model, X, y, bins=10):
    """Modified 2D heatmap with single-color (blue) gradient and better value differentiation"""
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
    
    # Calculate min and max values for better color scaling
    min_val = np.min(heatmap)
    max_val = np.max(heatmap)
    
    # Create custom blue colormap with more differentiation
    colors = sns.light_palette('blue', n_colors=256, as_cmap=True)
    
    sns.heatmap(
        heatmap,
        annot=True,
        fmt='.4f',  # Show 4 decimal places to better see small differences
        cmap=colors,
        xticklabels=x_ticks,
        yticklabels=y_ticks,
        annot_kws={'size': 8},
        vmin=min_val,
        vmax=max_val,
        center=(min_val + max_val) / 2  # Center the colormap
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

def parse_scientific_notation(value_str):
    """Parse scientific notation that might be incomplete"""
    try:
        # Handle complete scientific notation
        return float(value_str)
    except ValueError:
        # Handle incomplete scientific notation (e.g., '2.15e')
        if 'e' in value_str:
            base, exp = value_str.split('e')
            if exp == '':  # Incomplete exponent
                return float(base)
            return float(base) * (10 ** float(exp))
        raise

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
        # Get top 20 features and their importances
        feature_importances = pd.Series(
            st.session_state.rf_model_encoded.feature_importances_,
            index=st.session_state.X_train_encoded.columns
        ).sort_values(ascending=False)
        top_20 = feature_importances.head(20)
        
        # Create interactive bar plot for top 20 features
        fig = go.Figure(data=[
            go.Bar(
                x=[str(i+1) for i in range(len(top_20))],  # Numeric x-axis
                y=top_20.values,
                customdata=[[f"{float(feat):.1f}", val] for feat, val in zip(top_20.index, top_20.values)],
                hovertemplate="<br>".join([
                    "Rank: %{x}",
                    "Wavelength: %{customdata[0]} nm",
                    "Importance: %{customdata[1]:.6f}",
                    "<extra></extra>"
                ]),
                marker_color='rgb(102, 197, 204)'  # Light turquoise color
            )
        ])
        
        fig.update_layout(
            title='Top 20 Features (Spectral Wavelengths)',
            xaxis_title='Rank',
            yaxis_title='Importance Score',
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.2)'
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.2)'
            )
        )
        
        # Add gridlines
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store top features for later use
        top_features = top_20.index.tolist()
    
    with tab2:
        # New interactive scatter plot for all features
        fig = plot_feature_importance_scatter(
            st.session_state.rf_model_encoded,
            st.session_state.X_train_encoded
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Selection for Analysis
    st.header("Feature Analysis")
    selected_feature = st.selectbox(
        "Select feature for analysis",
        options=top_features,
        index=0
    )
    
    # Interval-Based Feature Importance
    st.subheader("Interval-Based Feature Importance")
    interval_labels, importance_scores = interval_permutation_importance(
        selected_feature,
        st.session_state.rf_model_encoded,
        st.session_state.X_test_encoded,
        st.session_state.y_test_encoded
    )
    
    # Create numeric labels and interval mapping
    numeric_labels = [str(i+1) for i in range(len(interval_labels))]
    
    # Safely parse and format interval ranges
    interval_ranges = []
    for label in interval_labels:
        try:
            start, end = label.split('-')
            start_val = parse_scientific_notation(start.strip())
            end_val = parse_scientific_notation(end.strip())
            interval_ranges.append(f"{format_scientific_to_simple(start_val)} - {format_scientific_to_simple(end_val)} nm")
        except (ValueError, IndexError):
            # If parsing fails, use the original label
            interval_ranges.append(label)
    
    # Create interactive Plotly figure
    fig = go.Figure(data=[
        go.Bar(
            x=numeric_labels,
            y=importance_scores,
            hovertemplate="<br>".join([
                "Interval Number: %{x}",
                "Importance Score: %{y:.4f}",
                "Range: %{customdata}",
                "<extra></extra>"  # This removes the trace name from hover
            ]),
            customdata=interval_ranges,
            marker_color='rgb(255, 183, 77)'  # Light orange color
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=f'Interval-Based Feature Importance Score for {selected_feature} nm',
        xaxis_title='Interval Number',
        yaxis_title='Permutation-Based Feature Importance',
        showlegend=False,
        hovermode='closest',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        )
    )
    
    # Add grid
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Show the interactive plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Interval ranges in an expander
    with st.expander("View Wavelength Ranges", expanded=False):
        interval_df = pd.DataFrame({
            'Interval Number': numeric_labels,
            'Range (nm)': interval_ranges
        })
        st.dataframe(
            interval_df,
            hide_index=True,  # Hide the index numbers
            use_container_width=True  # Make dataframe use full width
        )

    # 2D Feature Interaction
    st.header("2D Feature Interaction")
    st.write("Select two features for interaction analysis (default: top 2 important features)")
    
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