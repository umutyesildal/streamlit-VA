import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from services.state_management import initialize_session_state, get_filtered_predictions, is_model_trained
from services.plotting import plot_classwise_metrics, plot_actual_vs_predicted_distributions
from services.data_loader import load_data
from components.shared_sidebar import show_shared_sidebar

def show_metrics_table(y_test, y_pred, rf_model):
    """Modified metrics table function that handles DataFrame input"""
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values.ravel()
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    metrics_df = pd.DataFrame(report).transpose()
    
    # Remove 'support' column and accuracy/macro/weighted rows
    metrics_df = metrics_df.drop('support', axis=1)
    metrics_df = metrics_df.iloc[:-3]
    
    st.dataframe(
        metrics_df.style.format("{:.3f}"),
        use_container_width=True
    )

def create_interactive_confusion_matrix(y_true, y_pred, model, normalized=False, title="Confusion Matrix"):
    """Modified confusion matrix for dark theme"""
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.ravel()
    
    labels = model.classes_
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 3)
        text_format = '.1%'
        color_label = "Percentage"
    else:
        text_format = 'd'
        color_label = "Count"

    # Light blue color scheme
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale=[
            [0, '#f7fbff'],       # Lightest blue
            [0.2, '#deebf7'],     # Very light blue
            [0.4, '#c6dbef'],     # Light blue
            [0.6, '#9ecae1'],     # Medium blue
            [0.8, '#6baed6'],     # Blue
            [1.0, '#2171b5']      # Dark blue
        ],
        aspect='equal'
    )

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            total = np.sum(cm[i])
            percentage = value / total * 100 if total > 0 else 0
            row_sum = cm[i].sum()
            col_sum = cm[:, j].sum()
            
            if normalized:
                text = f'{value:.1%}'
                hover_text = (
                    f"True: {labels[i]}<br>"
                    f"Predicted: {labels[j]}<br>"
                    f"Value: {value:.1%}<br>"
                    f"Row Total: {row_sum:.1%}<br>"
                    f"Column Total: {col_sum/cm.sum():.1%}"
                )
            else:
                text = f'{int(value)}'
                hover_text = (
                    f"True: {labels[i]}<br>"
                    f"Predicted: {labels[j]}<br>"
                    f"Count: {int(value)}<br>"
                    f"Row %: {value/row_sum*100:.1f}%<br>"
                    f"Column %: {value/col_sum*100:.1f}%"
                )

            fig.add_annotation(
                x=j,
                y=i,
                text=text,
                hovertext=hover_text,
                showarrow=False,
                font=dict(
                    color='black' if value < (cm.max() / 2) else 'white',
                    size=16,
                    family='Arial'
                )
            )

    # Update layout for dark theme
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=24, color='white')
        ),
        xaxis_title=dict(text="<b>Predicted Label</b>", font=dict(color='white')),
        yaxis_title=dict(text="<b>True Label</b>", font=dict(color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=700
    )

    return fig

def create_interactive_classwise_metrics(y_test, y_pred, model, selected_classes):
    """Create interactive bar plot for classwise metrics using Plotly"""
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    metrics_df = pd.DataFrame(report).transpose()
    
    # Filter out the summary rows and keep only selected classes
    metrics_df = metrics_df.iloc[:-3]
    metrics_df = metrics_df.loc[metrics_df.index.isin(selected_classes)]
    
    metrics_df.index.name = 'Class'
    metrics_df = metrics_df.reset_index()
    
    # Melt the DataFrame for plotting
    plot_df = pd.melt(
        metrics_df,
        id_vars=['Class'],
        value_vars=['precision', 'recall', 'f1-score'],
        var_name='Metric',
        value_name='Score'
    )
    
    # Create interactive bar plot
    fig = px.bar(
        plot_df,
        x='Class',
        y='Score',
        color='Metric',
        barmode='group',
        color_discrete_sequence=['#8dd3c7', '#bebada', '#fb8072'],
        title='Classwise Performance Metrics'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend_title_text='',
        xaxis_title="Class",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        hovermode='x unified'
    )
    
    fig.update_traces(
        hovertemplate="Score: %{y:.3f}<extra></extra>"
    )
    
    return fig

def create_interactive_distribution_plot(y_test, y_pred, custom_order, selected_classes):
    """Create interactive distribution comparison plot using Plotly"""
    # Convert to pandas Series if needed
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    
    # Filter custom_order to only include selected classes
    filtered_order = [c for c in custom_order if c in selected_classes]
    
    # Create DataFrames for actual and predicted (only for selected classes)
    actual_counts = pd.Series(y_test).value_counts()
    actual_counts = actual_counts[actual_counts.index.isin(selected_classes)]
    actual_counts = actual_counts.reindex(filtered_order).fillna(0)
    
    pred_counts = pd.Series(y_pred).value_counts()
    pred_counts = pred_counts[pred_counts.index.isin(selected_classes)]
    pred_counts = pred_counts.reindex(filtered_order).fillna(0)
    
    # Combine data for plotting
    plot_df = pd.DataFrame({
        'Class': filtered_order * 2,
        'Count': pd.concat([actual_counts, pred_counts]),
        'Type': ['Actual'] * len(filtered_order) + ['Predicted'] * len(filtered_order)
    })
    
    # Create interactive bar plot
    fig = px.bar(
        plot_df,
        x='Class',
        y='Count',
        color='Type',
        barmode='group',
        color_discrete_sequence=['#a1dab4', '#41b6c4'],
        title='Actual vs Predicted Class Distribution'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend_title_text='',
        xaxis_title="Class",
        yaxis_title="Count",
        hovermode='x unified'
    )
    
    fig.update_traces(
        hovertemplate="%{y:d} instances<extra></extra>"
    )
    
    return fig

def main():
    st.title("Model Analysis")
    initialize_session_state()
    
    _, y_data = load_data()
    selected_classes = show_shared_sidebar(y_data)
    
    if not is_model_trained():
        st.warning("Please train the model first in the main page.")
        return

    # Matrix Analysis Section
    st.header("Matrix Analysis")
    matrix_tabs = st.tabs(["Regular Matrix", "Normalized Matrix", "Metrics Table"])
    
    # Use full dataset for matrices
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    
    with matrix_tabs[0]:
        st.plotly_chart(
            create_interactive_confusion_matrix(
                y_test, y_pred,
                st.session_state.rf_model,
                normalized=False,
                title="Confusion Matrix (Counts)"
            ),
            use_container_width=True
        )
    
    with matrix_tabs[1]:
        st.plotly_chart(
            create_interactive_confusion_matrix(
                y_test, y_pred,
                st.session_state.rf_model,
                normalized=True,
                title="Normalized Confusion Matrix"
            ),
            use_container_width=True
        )
    
    with matrix_tabs[2]:
        show_metrics_table(y_test, y_pred, st.session_state.rf_model)

    # Class Analysis Section
    st.header("Class Analysis")
    class_tabs = st.tabs(["Classwise Metrics", "Distribution Analysis"])
    
    if not selected_classes:
        for tab in class_tabs:
            with tab:
                st.warning("Please select classes in the sidebar.")
        return
    
    y_test_filtered, y_pred_filtered = get_filtered_predictions(selected_classes)
    
    if y_test_filtered is None or y_pred_filtered is None:
        for tab in class_tabs:
            with tab:
                st.error("No filtered data available.")
        return
    
    with class_tabs[0]:
        st.plotly_chart(
            create_interactive_classwise_metrics(
                y_test_filtered,
                y_pred_filtered,
                st.session_state.rf_model,
                selected_classes  # Pass selected classes
            ),
            use_container_width=True
        )
    
    with class_tabs[1]:
        custom_order = ["very_low", "low", "moderate", "high", "very_high"]
        st.plotly_chart(
            create_interactive_distribution_plot(
                y_test_filtered,
                y_pred_filtered,
                custom_order,
                selected_classes  # Pass selected classes
            ),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
