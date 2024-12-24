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
# 3. EVALUATE MODEL
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
# 4. PLOTTING UTILITIES
# -------------------------------------------------------------------
def show_confusion_matrix(y_test, y_pred, rf_model, title="Confusion Matrix", cmap='Reds', log_scale=False):
    """
    Plots a confusion matrix (and optionally a log-scale version) using matplotlib + seaborn.
    """
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if log_scale:
        # Apply log scale
        cm_log = np.log1p(cm)
        sns.heatmap(cm_log, annot=cm, fmt='d', cmap=cmap,
                    xticklabels=rf_model.classes_, yticklabels=rf_model.classes_, ax=ax)
        ax.set_title(title + " (Log Scale)")
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=rf_model.classes_, yticklabels=rf_model.classes_, ax=ax)
        ax.set_title(title)
        
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)


def show_confusion_matrix_normalized(y_test, y_pred, rf_model, title="Normalized Confusion Matrix"):
    """
    Plots a normalized confusion matrix where each row is normalized by its sum.
    """
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Reds',
        xticklabels=rf_model.classes_, yticklabels=rf_model.classes_, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    st.pyplot(fig)


def show_metrics_table(y_test, y_pred, rf_model):
    """
    Displays a DataFrame with True Positives, False Positives, False Negatives, True Negatives for each class.
    """
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    
    metrics = pd.DataFrame({
        'Class': rf_model.classes_,
        'True Positives': TP,
        'False Positives': FP,
        'False Negatives': FN,
        'True Negatives': TN
    })
    
    st.subheader("Confusion Matrix Detail")
    st.dataframe(metrics)


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


def plot_pca_scatter(X_test, y_test, y_pred):
    """
    Performs PCA on X_test (2 components), then plots a scatter showing true vs. predicted labels.
    Misclassified points are marked with 'X'.
    """
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    pca_df = pd.DataFrame({
        'Component 1': X_test_pca[:, 0],
        'Component 2': X_test_pca[:, 1],
        'True Label': y_test.values.ravel(),
        'Predicted Label': y_pred,
    })
    pca_df['Correct'] = pca_df['True Label'] == pca_df['Predicted Label']

    unique_labels = pca_df['True Label'].unique()
    palette = sns.color_palette("hsv", len(unique_labels))
    label_color_dict = dict(zip(unique_labels, palette))

    fig, ax = plt.subplots(figsize=(12, 8))
    for label in unique_labels:
        subset = pca_df[pca_df['True Label'] == label]
        ax.scatter(
            subset['Component 1'],
            subset['Component 2'],
            c=[label_color_dict[label]],
            label=f"Class {label}",
            marker='o',
            edgecolors='k',
            s=100,
            alpha=0.7
        )

    # Overlay misclassified points
    misclassified = pca_df[~pca_df['Correct']]
    ax.scatter(
        misclassified['Component 1'],
        misclassified['Component 2'],
        c=[label_color_dict[label] for label in misclassified['True Label']],
        marker='x',
        s=100,
        label='Misclassified',
        alpha=0.9
    )

    ax.legend()
    ax.set_title('PCA Scatter Plot of Model Predictions')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    st.pyplot(fig)


def plot_actual_vs_predicted_distributions(y_test, y_pred):
    """
    Plots a side-by-side bar chart comparing the distribution of actual vs. predicted classes.
    """
    # Extract the labels from y_test DataFrame/Series
    if isinstance(y_test, pd.DataFrame):
        # Assuming the column is 'x' based on your code snippet
        y_test_labels = y_test['x']
    else:
        # If y_test is already a series
        y_test_labels = y_test

    # Convert y_pred to a pandas Series
    y_pred_series = pd.Series(y_pred, name='Predicted')

    true_class_counts = y_test_labels.value_counts().sort_index()
    pred_class_counts = y_pred_series.value_counts().sort_index()

    class_counts_df = pd.DataFrame({
        'Actual': true_class_counts,
        'Predicted': pred_class_counts
    }).fillna(0)

    # (Optional) Reorder the classes if you have a known class order
    class_order = ['very_low', 'low', 'moderate', 'high', 'very_high']
    # This reindex won't break if any are missing, but you'll see rows with 0 if so
    class_counts_df = class_counts_df.reindex(class_order).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Comparison of Actual vs Predicted Class Distributions')
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

    # --- Load Data ---
    features_path = 'lucas_organic_carbon_training_and_test_data.csv'
    target_path = 'lucas_organic_carbon_target.csv'
    X_data, y_data = load_data(features_path, target_path)

    st.write("### Data Shapes")
    st.write("Features shape:", X_data.shape)
    st.write("Target shape:", y_data.shape)

    # --- Train/Test Split ---
    test_size = st.slider("Test Size (fraction)", 0.05, 0.95, 0.3, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, 
        y_data, 
        test_size=test_size, 
        random_state=42
    )

    st.write(f"**Train size**: {len(X_train)}, **Test size**: {len(X_test)}")

    # --- Train Model ---
    if st.button("Train Random Forest"):
        rf_model = train_random_forest(X_train, y_train)
        st.success("Model training complete!")
        
        # --- Evaluate Model ---
        y_pred = evaluate_model(rf_model, X_test, y_test)

        # --- Show Confusion Matrix (Regular) ---
        show_confusion_matrix(y_test, y_pred, rf_model, title="Confusion Matrix (Regular)", cmap='Reds')
        
        # --- Show Confusion Matrix (Log Scale) ---
        show_confusion_matrix(y_test, y_pred, rf_model, title="Confusion Matrix", cmap='Blues', log_scale=True)
        
        # --- Confusion Matrix Detail (TP, FP, FN, TN) ---
        show_metrics_table(y_test, y_pred, rf_model)
        
        # --- Show True Class Distribution ---
        show_true_class_distribution(y_test)
        
        # --- Show Normalized Confusion Matrix ---
        show_confusion_matrix_normalized(y_test, y_pred, rf_model, 
                                         title="Normalized Confusion Matrix Highlighting Misclassifications")
        
        # --- PCA Scatter Plot ---
        plot_pca_scatter(X_test, y_test, y_pred)
        
        # --- Classification Report Heatmap ---
        plot_classification_report_heatmap(y_test, y_pred)
        
        # --- Compare Actual vs. Predicted Distribution ---
        plot_actual_vs_predicted_distributions(y_test, y_pred)
        
        # --- Feature Importances ---
        plot_feature_importance(rf_model, X_train, threshold=0.005)
    else:
        st.info("Click 'Train Random Forest' to run the model and see all outputs.")


if __name__ == "__main__":
    main()
