import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

custom_order = ["very_low", "low", "moderate", "high", "very_high"]


# -------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------
def load_data(features_path, target_path):
    """
    Loads the features (X) and target (y) data from CSV files.
    Returns two pandas DataFrames: X_data, y_data.
    """
    X_data = pd.read_csv(features_path)
    y_data = pd.read_csv(target_path)
    return X_data, y_data


# -------------------------------------------------------------------
# 2. TRAIN MODEL
# -------------------------------------------------------------------
def train_random_forest(X_train, y_train):
    """
    Trains and returns a RandomForestClassifier with predefined hyperparameters.
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        min_samples_split=10,
        min_samples_leaf=1,
        random_state=42
    )
    rf_model.fit(X_train, y_train.values.ravel())
    return rf_model

# -------------------------------------------------------------------
# 3. HELPER: FILTER CLASSES GLOBALLY
# -------------------------------------------------------------------
def filter_classes(y_test, y_pred, selected_classes):
    """
    Returns filtered y_test and y_pred, keeping only rows where y_test 
    is in selected_classes. Also converts y_pred to a Series with the same index.
    NOTE: Assumes the target column is named 'x' if y_test is a DataFrame.
    """
    # Convert y_test to a Series if it's a DataFrame
    if isinstance(y_test, pd.DataFrame):
        # Assuming the single column in your target DataFrame is named 'x'
        y_test_series = y_test['x']
    else:
        y_test_series = y_test

    # Convert y_pred to a Series with same index
    y_pred_series = pd.Series(y_pred, index=y_test_series.index)

    # Keep only the rows for which y_test is in selected_classes
    mask = y_test_series.isin(selected_classes)
    y_test_filtered = y_test_series[mask]
    y_pred_filtered = y_pred_series[mask]

    return y_test_filtered, y_pred_filtered



# -------------------------------------------------------------------
# 4. EVALUATE MODEL
# -------------------------------------------------------------------
def evaluate_model(rf_model, X_test, y_test):
    """
    Predicts on X_test using the trained rf_model and prints out:
    - accuracy_score
    - classification_report
    Also returns y_pred (predictions) for further analysis.
    """
    y_pred = rf_model.predict(X_test)

    # Streamlit display
    st.subheader("Model Performance")
    st.write(f"**Accuracy Score**: {accuracy_score(y_test, y_pred):.4f}")
    
    # Generate text classification report
    report_txt = classification_report(y_test, y_pred)
    st.text("Classification Report:\n" + report_txt)

    return y_pred


# -------------------------------------------------------------------
# 5. PLOTTING UTILITIES
# -------------------------------------------------------------------
def show_confusion_matrix(y_test, y_pred, rf_model, title="Confusion Matrix", cmap='Reds', log_scale=False):
    """
    Plots a confusion matrix (and optionally a log-scale version) using matplotlib + seaborn.
    Only includes classes that appear in y_test (intersection with model classes).
    """
    # Determine which classes are present after filtering
    present_classes = np.intersect1d(rf_model.classes_, y_test.unique())
    
    # Compute confusion matrix only for present classes
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    fig, ax = plt.subplots(figsize=(10, 7))

    if log_scale:
        cm_log = np.log1p(cm)
        sns.heatmap(cm_log, annot=cm, fmt='d', cmap=cmap,
                    xticklabels=present_classes, yticklabels=present_classes, ax=ax)
        ax.set_title(title + " (Log Scale)")
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=present_classes, yticklabels=present_classes, ax=ax)
        ax.set_title(title)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

def show_confusion_matrix_normalized(y_test, y_pred, rf_model, title="Normalized Confusion Matrix"):
    """
    Plots a normalized confusion matrix where each row is normalized by its sum.
    Only includes classes that appear in y_test (intersection with model classes).
    """
    present_classes = np.intersect1d(rf_model.classes_, y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Reds',
        xticklabels=present_classes, yticklabels=present_classes, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    st.pyplot(fig)

def show_metrics_table(y_test, y_pred, rf_model):
    """
    Displays a DataFrame with True Positives, False Positives, False Negatives, True Negatives 
    for each class actually present in y_test (intersection with model classes).
    """
    present_classes = np.intersect1d(rf_model.classes_, y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    metrics = pd.DataFrame({
        'Class': present_classes,
        'True Positives': TP,
        'False Positives': FP,
        'False Negatives': FN,
        'True Negatives': TN
    })
    
    st.subheader("Confusion Matrix Detail")
    st.dataframe(metrics)

def plot_classwise_metrics(y_test, y_pred, rf_model):
    """
    Plots a bar chart showing precision, recall, and F1-score for each class 
    actually present in y_test (excludes overall accuracy, macro avg, and weighted avg rows).
    """
    present_classes = np.intersect1d(rf_model.classes_, y_test.unique())
    fig, ax = plt.subplots(figsize=(6, 5))
    report = classification_report(
        y_test, 
        y_pred, 
        labels=present_classes, 
        output_dict=True
    )
    metrics_df = pd.DataFrame(report).transpose()

    # Filter out the last three rows (accuracy, macro avg, weighted avg)
    metrics_df = metrics_df.iloc[:-3, :3]

    # Rename index to "Class" and move it as a column
    metrics_df.index.name = 'Class'
    metrics_df.reset_index(inplace=True)

    # Melt the DataFrame for a seaborn bar plot
    metrics_melted = metrics_df.melt(id_vars='Class', var_name='Metric', value_name='Score')

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Class', y='Score', hue='Metric', data=metrics_melted, ax=ax)
    ax.set_title('Class-wise Precision, Recall, and F1-Score')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    st.pyplot(fig)



def show_true_class_distribution(y_test):
    """
    Plots the distribution (countplot) of the true classes from y_test.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=y_test.values.flatten(), palette="Set2", ax=ax)
    ax.set_title("True Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def plot_classification_report_heatmap(y_test, y_pred):
    """
    Shows a heatmap of precision, recall, and F1-score for each class.
    """
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Filter only the classes that appear in y_test
    unique_classes = np.unique(y_test)
    metrics_df = report_df.loc[unique_classes, ['precision', 'recall', 'f1-score']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(metrics_df, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=0.5, ax=ax)
    ax.set_title("Precision, Recall, and F1-Score Heatmap")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Classes")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

def plot_actual_vs_predicted_distributions(y_test, y_pred):
    """
    Plots a side-by-side bar chart comparing the distribution of actual vs. predicted classes.
    Assumes y_test and y_pred have already been filtered to only include desired classes.
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # If y_test is a DataFrame, extract its target column (e.g., 'x')
    if isinstance(y_test, pd.DataFrame):
        y_test_labels = y_test['x']
    else:
        y_test_labels = y_test

    # Convert y_pred to a pandas Series with the same (filtered) index
    y_pred_series = pd.Series(y_pred, index=y_test_labels.index)

    # Identify classes actually present (both actual & predicted) after filtering
    # so that any classes not in the filter are excluded.
    present_classes = sorted(set(y_test_labels.unique()) & set(y_pred_series.unique()))

    # Compute distribution counts
    true_class_counts = y_test_labels.value_counts().sort_index()
    pred_class_counts = y_pred_series.value_counts().sort_index()
    

    
    # Combine into a DataFrame
    class_counts_df = pd.DataFrame({
        'Actual': true_class_counts,
        'Predicted': pred_class_counts
    }).fillna(0)
    
    # Keep only those that match your custom order
    final_order = [c for c in custom_order if c in present_classes]
    
    # Reindex with final_order
    class_counts_df = class_counts_df.reindex(final_order).fillna(0)

    # Reindex only those classes that appear in the union (or intersection, if desired)
    class_counts_df = class_counts_df.reindex(present_classes).fillna(0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Comparison of Actual vs. Predicted Class Distributions')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Instances')
    ax.legend(title='Legend')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

def plot_feature_importance(rf_model, X_train, threshold=0.005):
    """
    Shows a bar chart of feature importances for features whose importance is above the threshold.
    """
    feature_importances = rf_model.feature_importances_
    important_indices = feature_importances > threshold
    features = X_train.columns[important_indices]
    importances = feature_importances[important_indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(features, importances)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Filtered Feature Importance (threshold > {:.3f})'.format(threshold))
    plt.xticks(rotation=90)
    st.pyplot(fig)
    


# -------------------------------------------------------------------
# 5. STREAMLIT MAIN
# -------------------------------------------------------------------
def main():
    st.title("Random Forest Classification with Streamlit")

    # --- Setup session state variables ---
    if "rf_model" not in st.session_state:
        st.session_state["rf_model"] = None
    if "y_pred" not in st.session_state:
        st.session_state["y_pred"] = None

    # --- Load Data ---
    features_path = 'data/lucas_organic_carbon_training_and_test_data.csv'
    target_path = 'data/lucas_organic_carbon_target.csv'
    X_data, y_data = load_data(features_path, target_path)

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

    # --- Train/Test Split ---
    test_size = st.sidebar.slider("Test Size (fraction)", 0.05, 0.95, 0.3, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, 
        y_data, 
        test_size=test_size, 
        random_state=42
    )

    st.write(f"**Train size**: {len(X_train)}, **Test size**: {len(X_test)}")

    # ----------------------------------------------------------------
    # Button: Train the model and auto-run all plots once
    # ----------------------------------------------------------------
    if st.button("Train Random Forest"):
        rf_model = train_random_forest(X_train, y_train)
        st.session_state["rf_model"] = rf_model
        st.session_state["y_pred"] = rf_model.predict(X_test)
        st.success("Model training complete. Automatically generating all plots...")

        # Evaluate Model (on full test set)
        evaluate_model(rf_model, X_test, y_test)

        # Filter for selected classes
        y_test_filtered, y_pred_filtered = filter_classes(y_test, st.session_state["y_pred"], selected_classes)

        # Show Confusion Matrix
        show_confusion_matrix(y_test_filtered, y_pred_filtered, rf_model, 
                             title="Confusion Matrix (Regular)", cmap='Reds')
        
        # Show Normalized Confusion Matrix
        show_confusion_matrix_normalized(y_test_filtered, y_pred_filtered, rf_model, 
                                         title="Normalized Confusion Matrix")
        
        # Confusion Matrix Detail
        show_metrics_table(y_test_filtered, y_pred_filtered, rf_model)

        # Side-by-side columns for Classwise Metrics and Actual vs Predicted
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classwise Metrics")
            plot_classwise_metrics(y_test_filtered, y_pred_filtered, rf_model)
        with col2:
            st.subheader("Actual vs. Predicted Distribution")
            plot_actual_vs_predicted_distributions(y_test_filtered, y_pred_filtered)

        # Feature Importance
        plot_feature_importance(rf_model, X_train, threshold=0.005)

    # ----------------------------------------------------------------
    # If the model is trained, present individual buttons for each plot
    # ----------------------------------------------------------------
    if st.session_state["rf_model"] is not None:
        rf_model = st.session_state["rf_model"]
        y_pred = st.session_state["y_pred"]  # predictions on the full test set

        # Filter y_test, y_pred based on selected classes
        y_test_filtered, y_pred_filtered = filter_classes(y_test, y_pred, selected_classes)

        st.write("---")
        st.write("## Individual Actions")

        # Evaluate Model Button
        if st.button("Evaluate Model"):
            evaluate_model(rf_model, X_test, y_test)

        # Show Confusion Matrix Button
        if st.button("Show Confusion Matrix"):
            show_confusion_matrix(y_test_filtered, y_pred_filtered, rf_model, 
                                 title="Confusion Matrix (Regular)", cmap='Reds')

        # Normalized Confusion Matrix
        if st.button("Show Normalized Confusion Matrix"):
            show_confusion_matrix_normalized(y_test_filtered, y_pred_filtered, rf_model, 
                                             title="Normalized Confusion Matrix")

        # Confusion Matrix Detail
        if st.button("Show Confusion Matrix Detail"):
            show_metrics_table(y_test_filtered, y_pred_filtered, rf_model)

        # Side-by-side: Classwise Metrics & Actual vs. Predicted Distribution
        show_both = st.button("Show Classwise + Actual vs Predicted")
        if show_both:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Classwise Metrics")
                plot_classwise_metrics(y_test_filtered, y_pred_filtered, rf_model)
            with col2:
                st.subheader("Actual vs. Predicted Distribution")
                plot_actual_vs_predicted_distributions(y_test_filtered, y_pred_filtered)

        # Feature Importance
        if st.button("Show Feature Importance"):
            plot_feature_importance(rf_model, X_train, threshold=0.005)

    else:
        st.info("Please train the Random Forest model first.")



if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()