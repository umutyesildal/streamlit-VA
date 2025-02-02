import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix
from services.state_management import initialize_session_state, get_filtered_predictions, is_model_trained
from services.plotting import show_metrics_table
from services.data_loader import load_data
from components.shared_sidebar import show_shared_sidebar

def create_interactive_confusion_matrix(y_true, y_pred, model, normalized=False, title="Confusion Matrix"):
    # Convert y_true to numpy array if it's a DataFrame
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

    # Calculate color scale range based on data
    max_val = cm.max()
    min_val = cm.min()
    
    # Create the heatmap with adjusted color scale
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale=[
            [0, '#ffffff'],       # White for zero
            [0.2, '#edf8fb'],     # Very light blue
            [0.4, '#b2e2e2'],     # Light blue-green
            [0.6, '#66c2a4'],     # Medium green
            [0.8, '#2ca25f'],     # Dark green
            [1.0, '#006d2c']      # Very dark green
        ],
        zmin=min_val,            # Set minimum of color scale
        zmax=max_val,            # Set maximum of color scale
        aspect='equal'           # Make cells square
    )

    # Add text annotations with improved formatting
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

    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            y=0.95,
            font=dict(size=24, family='Arial Black')
        ),
        xaxis_title=dict(
            text="<b>Predicted Label</b>",
            font=dict(size=16, family='Arial')
        ),
        yaxis_title=dict(
            text="<b>True Label</b>",
            font=dict(size=16, family='Arial')
        ),
        xaxis={'side': 'bottom'},
        width=900,  # Larger size
        height=700,
        coloraxis_colorbar=dict(
            title=dict(
                text=f"<b>{color_label}</b>",
                font=dict(size=14, family='Arial')
            ),
            len=0.75,          # Adjust colorbar length
            thickness=20,      # Adjust colorbar thickness
            tickformat='.2f' if normalized else 'd',
            nticks=10         # Adjust number of ticks
        ),
        font=dict(size=14, family='Arial'),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot
    )

    # Update hover template with more information
    fig.update_traces(
        hovertemplate=None,  # Use the custom hovertext from annotations
    )

    return fig

def main():
    st.set_page_config(layout="wide", page_title="Confusion Matrix Analysis")
    initialize_session_state()
    
    # Load data for sidebar
    _, y_data = load_data()
    
    # Show shared sidebar
    selected_classes = show_shared_sidebar(y_data)
    
    st.title("Confusion Matrix and Metrics")
    
    if not is_model_trained():
        st.warning("Please train the model first in the main page.")
        return
    
    if not selected_classes:
        st.warning("Please select classes in the sidebar of the homepage.")
        return
        
    y_test_filtered, y_pred_filtered = get_filtered_predictions(selected_classes)
    
    if y_test_filtered is None or y_pred_filtered is None:
        st.error("No filtered data available.")
        return
    
    # Add description
    st.markdown("""
        <style>
            .header-description {
                font-size: 18px;
                color: #666;
                margin-bottom: 30px;
            }
        </style>
        <div class="header-description">
            Interactive confusion matrices showing model predictions. Hover over cells for detailed statistics.
            The normalized matrix shows percentages while the regular matrix shows counts.
        </div>
    """, unsafe_allow_html=True)
    
    # Interactive confusion matrices
    st.subheader("Regular Confusion Matrix")
    st.plotly_chart(
        create_interactive_confusion_matrix(
            y_test_filtered, 
            y_pred_filtered, 
            st.session_state.rf_model,
            normalized=False,
            title="Confusion Matrix (Counts)"
        )
    )
    
    st.subheader("Normalized Confusion Matrix")
    st.plotly_chart(
        create_interactive_confusion_matrix(
            y_test_filtered, 
            y_pred_filtered, 
            st.session_state.rf_model,
            normalized=True,
            title="Normalized Confusion Matrix (Percentages)"
        )
    )
    
    # Metrics table
    st.subheader("Metrics Table")
    show_metrics_table(y_test_filtered, y_pred_filtered, st.session_state.rf_model)

if __name__ == "__main__":
    main()
