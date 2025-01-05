import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from utils import calculate_interval_impact, substitute_and_calculate_accuracy

def show_confusion_matrix(y_test, y_pred, rf_model, title="Confusion Matrix", cmap='Reds', log_scale=False):
    """
    Plots a confusion matrix (and optionally a log-scale version) using matplotlib + seaborn.
    Only includes classes that appear in y_test (intersection with model classes).
    """
    present_classes = np.intersect1d(rf_model.classes_, y_test.unique())
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

def plot_actual_vs_predicted_distributions(y_test, y_pred, custom_order):
    """
    Plots a side-by-side bar chart comparing the distribution of actual vs. predicted classes.
    Assumes y_test and y_pred have already been filtered to only include desired classes.
    """
    if isinstance(y_test, pd.DataFrame):
        y_test_labels = y_test['x']
    else:
        y_test_labels = y_test

    y_pred_series = pd.Series(y_pred, index=y_test_labels.index)
    present_classes = sorted(set(y_test_labels.unique()) & set(y_pred_series.unique()))

    true_class_counts = y_test_labels.value_counts().sort_index()
    pred_class_counts = y_pred_series.value_counts().sort_index()
    
    class_counts_df = pd.DataFrame({
        'Actual': true_class_counts,
        'Predicted': pred_class_counts
    }).fillna(0)
    
    final_order = [c for c in custom_order if c in present_classes]
    class_counts_df = class_counts_df.reindex(final_order).fillna(0)

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

def show_interval_impact(rf_model, X_test, y_test, feature_importances, bins=10):
    """
    Identifies the most important feature, creates intervals with qcut,
    calls calculate_interval_impact, and then plots the impact of each interval on accuracy.
    """
    top_feature_idx = np.argmax(feature_importances)
    feature = X_test.columns[top_feature_idx]

    intervals_series = pd.qcut(X_test[feature], q=bins, duplicates='drop')
    intervals = [(interval.left, interval.right) for interval in intervals_series.unique()]

    impacts = calculate_interval_impact(feature, intervals, rf_model, X_test, y_test)

    fig, ax = plt.subplots(figsize=(10, 7))
    x_labels = [f"{start:.5f}-{end:.5f}" for start, end in intervals]
    ax.plot(x_labels, impacts, marker='o')
    ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
    ax.set_xlabel("Intervals")
    ax.set_ylabel("Impact on Accuracy")
    ax.set_title(f"Impact of Intervals on '{feature}'")

    st.pyplot(fig)


def show_substitution_analysis(rf_model, X_test, y_test, feature_name):
    """
    Performs substitution analysis on the given feature and plots accuracy for min, mean, and max values.
    """
    if feature_name not in X_test.columns:
        st.error(f"Feature '{feature_name}' not found in the dataset.")
        return

    min_value = X_test[feature_name].min()
    mean_value = X_test[feature_name].mean()
    max_value = X_test[feature_name].max()

    st.write(f"**Substitution Analysis for '{feature_name}'**")
    st.write(f"Min: {min_value:.4f}, Mean: {mean_value:.4f}, Max: {max_value:.4f}")

    replacement_values = [min_value, mean_value, max_value]
    accuracies = [
        substitute_and_calculate_accuracy(feature_name, value, rf_model, X_test, y_test)
        for value in replacement_values
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(["Min", "Mean", "Max"], accuracies, marker="o", color="b", label="Accuracy")
    ax.set_title(f"Substitution Analysis for '{feature_name}'")
    ax.set_xlabel("Replacement Values")
    ax.set_ylabel("Accuracy")
    ax.grid()
    st.pyplot(fig)

def show_permutation_importance(rf_model, X_test, y_test, feature_name, n_iter=10):
    """
    Calculates and displays permutation importance for the specified feature.
    """
    if feature_name not in X_test.columns:
        st.error(f"Feature '{feature_name}' not found in the dataset.")
        return

    baseline_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    accuracy_drop = []

    for _ in range(n_iter):
        X_permuted = X_test.copy()
        X_permuted[feature_name] = shuffle(X_permuted[feature_name].values)
        permuted_accuracy = accuracy_score(y_test, rf_model.predict(X_permuted))
        accuracy_drop.append(baseline_accuracy - permuted_accuracy)

    perm_importance_score = np.mean(accuracy_drop)

    st.write(f"**Permutation Importance for '{feature_name}': {perm_importance_score:.5f}**")

def show_interval_permutation_importance(feature, model, X, y, num_intervals=20):
    """
    Calculate interval-based feature importance using permutation within intervals.
    Arguments:
    - feature: The selected feature (column) for analysis.
    - model: The trained machine learning model (e.g., RandomForestClassifier).
    - X: Test dataset (features).
    - y: True labels (target).
    - num_intervals: Number of intervals to partition the feature into.

    Output: Bar plot showing the feature importance scores for each interval.
    """
    # Partition feature into intervals
    intervals = pd.qcut(X[feature], q=num_intervals, duplicates='drop')
    interval_labels = []
    importance_scores = []

    baseline_accuracy = accuracy_score(y, model.predict(X))

    for interval in intervals.unique():
        # Create modified datasets P' and P'' within the interval
        X_modified_p1 = X.copy()
        X_modified_p2 = X.copy()

        # Substitute P to P' and P''
        mid_left = interval.left
        mid_right = interval.right

        X_modified_p1[feature] = mid_left  # Substitute with left interval boundary (P')
        X_modified_p2[feature] = mid_right  # Substitute with right interval boundary (P'')

        # Calculate accuracy after substitution
        accuracy_p1 = accuracy_score(y, model.predict(X_modified_p1))
        accuracy_p2 = accuracy_score(y, model.predict(X_modified_p2))

        # Calculate feature importance as the average accuracy drop
        avg_importance = (baseline_accuracy - accuracy_p1 + baseline_accuracy - accuracy_p2) / 2
        importance_scores.append(avg_importance)
        interval_labels.append(f"{mid_left:.2e} - {mid_right:.2e}")

    # Plot interval feature importance scores
    plt.figure(figsize=(12, 6))
    plt.bar(interval_labels, importance_scores, color='orange')
    plt.xlabel('Interval Range')
    plt.ylabel('Permutation-Based Feature Importance')
    plt.title(f'Interval-Based Feature Importance Score for {feature} (with Permutation)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_2d_heatmap( model, X, y, feature1, feature2, bins=10):
    """
    Creates a 2D heatmap of model accuracy when feature1 and feature2 are replaced 
    by the midpoint of each bin in a grid.

    Parameters:
    - feature1: str, first feature name.
    - feature2: str, second feature name.
    - model: trained RandomForestClassifier.
    - X: DataFrame, feature set.
    - y: Series, true labels.
    - bins: int, number of bins to divide each feature range.

    Returns:
    - None (displays the heatmap).
    """
    if feature1 not in X.columns or feature2 not in X.columns:
        st.error(f"One or both features '{feature1}' and '{feature2}' not found in the dataset.")
        return

    min_feature1 = X[feature1].min()
    max_feature1 = X[feature1].max()
    min_feature2 = X[feature2].min()
    max_feature2 = X[feature2].max()

    x_edges = np.linspace(min_feature1, max_feature1, bins + 1)
    y_edges = np.linspace(min_feature2, max_feature2, bins + 1)

    heatmap = np.zeros((bins, bins))

    for i in range(bins):
        for j in range(bins):
            X_modified = X.copy()

            midpoint_x = (x_edges[i] + x_edges[i + 1]) / 2
            X_modified[feature1] = X_modified[feature1].apply(
                lambda v: midpoint_x if x_edges[i] <= v <= x_edges[i+1] else v
            )

            midpoint_y = (y_edges[j] + y_edges[j + 1]) / 2
            X_modified[feature2] = X_modified[feature2].apply(
                lambda v: midpoint_y if y_edges[j] <= v <= y_edges[j+1] else v
            )

            y_pred_modified = model.predict(X_modified)
            heatmap[i, j] = accuracy_score(y, y_pred_modified)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        heatmap, cmap='coolwarm',
        xticklabels=np.round(x_edges, 2),
        yticklabels=np.round(y_edges, 2),
        cbar_kws={'label': 'Accuracy'},
        ax=ax
    )
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title('2D Heatmap of Feature Interactions')
    st.pyplot(fig)