import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.utils import shuffle

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
    
def calculate_interval_impact(feature, intervals, model, X, y):
    """
    Calculates how replacing feature values within each interval by the 
    midpoint of that interval affects model accuracy.

    :param feature: The name of the feature (string) to analyze.
    :param intervals: A list of (start, end) tuples that define intervals for the feature.
    :param model: A trained model (e.g., RandomForestClassifier).
    :param X: The feature DataFrame (test set).
    :param y: The true labels (test set).
    :return: A list of accuracy scores, one for each interval replacement.
    """

    impacts = []
    for start, end in intervals:
        X_modified = X.copy()
        if feature in X_modified.columns:
            midpoint = (start + end) / 2
            X_modified[feature] = X_modified[feature].apply(
                lambda x: midpoint if (start <= x <= end) else x
            )
            y_pred_modified = model.predict(X_modified)
            impact = accuracy_score(y, y_pred_modified)
            impacts.append(impact)
        else:
            print(f"Feature '{feature}' not found in the dataset.")
            return []
    return impacts

def show_interval_impact(rf_model, X_test, y_test, feature_importances, bins=10):
    """
    Identifies the most important feature, creates intervals with qcut,
    calls calculate_interval_impact, and then plots the impact of each interval on accuracy.

    :param rf_model: Trained RandomForestClassifier or similar model
    :param X_test: Pandas DataFrame (test features)
    :param y_test: Pandas Series or array-like (test labels)
    :param feature_importances: Numpy array of feature importances
    :param bins: Number of bins for qcut (default: 10)
    """

    # 1. Pick the top feature
    top_feature_idx = np.argmax(feature_importances)
    feature = X_test.columns[top_feature_idx]

    # 2. Create intervals with qcut
    intervals_series = pd.qcut(X_test[feature], q=bins, duplicates='drop')
    intervals = [(interval.left, interval.right) for interval in intervals_series.unique()]

    # 3. Calculate interval impact
    impacts = calculate_interval_impact(feature, intervals, rf_model, X_test, y_test)

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    x_labels = [f"{start:.5f}-{end:.5f}" for start, end in intervals]
    ax.plot(x_labels, impacts, marker='o')
    ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
    ax.set_xlabel("Intervals")
    ax.set_ylabel("Impact on Accuracy")
    ax.set_title(f"Impact of Intervals on '{feature}'")

    st.pyplot(fig)


# === Substitution Function ===
def substitute_and_calculate_accuracy(feature, replacement_value, model, X, y):
    X_modified = X.copy()
    X_modified[feature] = replacement_value
    y_pred_modified = model.predict(X_modified)
    return accuracy_score(y, y_pred_modified)


def plot_2d_heatmap(feature1, feature2, model, X, y, bins=10):
    """
    Creates a 2D heatmap of model accuracy when feature1 and feature2 are replaced 
    by the midpoint of each bin in a grid.

    :param feature1: The name of the first feature.
    :param feature2: The name of the second feature.
    :param model: A trained model (RandomForestClassifier or similar).
    :param X: The feature DataFrame (test set).
    :param y: The true labels (test set).
    :param bins: Number of bins along each feature axis.
    :return: A matplotlib figure containing the heatmap.
    """

    if feature1 not in X.columns or feature2 not in X.columns:
        print(f"One or both features '{feature1}' and '{feature2}' not found in the dataset.")
        return None

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

            # Replace feature1 with midpoint in bin i
            midpoint_x = (x_edges[i] + x_edges[i + 1]) / 2
            X_modified[feature1] = X_modified[feature1].apply(
                lambda v: midpoint_x if x_edges[i] <= v <= x_edges[i+1] else v
            )

            # Replace feature2 with midpoint in bin j
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
    return fig

def show_2d_heatmap(rf_model, X_test, y_test, feature_importances, bins=10):
    """
    Identifies the top two most important features, calls plot_2d_heatmap,
    and displays the resulting figure in Streamlit.

    :param rf_model: Trained RandomForestClassifier or similar model
    :param X_test: Pandas DataFrame (test features)
    :param y_test: Pandas Series or array-like (test labels)
    :param feature_importances: Numpy array of feature importances
    :param bins: Number of bins along each feature axis (default: 10)
    """

    # Find the top two features
    top_feature1_idx = np.argmax(feature_importances)
    feature1 = X_test.columns[top_feature1_idx]

    # Second best: the next highest importance
    sorted_indices = np.argsort(feature_importances)
    feature2 = X_test.columns[sorted_indices[-2]]  # second-highest importance

    fig = plot_2d_heatmap(feature1, feature2, rf_model, X_test, y_test, bins=bins)
    if fig is not None:
        st.pyplot(fig)

def get_single_feature_importance(rf_model, X_data, feature_name):
    """
    Returns the feature importance score for a single feature from a trained RandomForest model.
    
    Args:
        rf_model: A trained RandomForestClassifier (or similar model with feature_importances_).
        X_data (pd.DataFrame): The DataFrame containing the features used to train rf_model.
        feature_name (str): The name of the feature whose importance to retrieve.
    
    Returns:
        float: The importance score for the specified feature.
    
    Raises:
        ValueError: If the feature_name is not a column in X_data.
    """
    if feature_name not in X_data.columns:
        raise ValueError(f"Feature '{feature_name}' not found in the dataset.")
    
    feature_index = list(X_data.columns).index(feature_name)
    return rf_model.feature_importances_[feature_index]



def show_substitution_analysis(rf_model, X_test, y_test, feature_name):
    """
    Performs substitution analysis on the given feature and plots accuracy for min, mean, and max values.

    Args:
        rf_model: Trained RandomForestClassifier model.
        X_test: DataFrame of test features.
        y_test: Series or array of test labels.
        feature_name: Name of the feature to analyze.
    """
    if feature_name not in X_test.columns:
        st.error(f"Feature '{feature_name}' not found in the dataset.")
        return

    # Calculate min, mean, max values
    min_value = X_test[feature_name].min()
    mean_value = X_test[feature_name].mean()
    max_value = X_test[feature_name].max()

    st.write(f"**Substitution Analysis for '{feature_name}'**")
    st.write(f"Min: {min_value:.4f}, Mean: {mean_value:.4f}, Max: {max_value:.4f}")

    # Replacement values and corresponding accuracies
    replacement_values = [min_value, mean_value, max_value]
    accuracies = [
        substitute_and_calculate_accuracy(feature_name, value, rf_model, X_test, y_test)
        for value in replacement_values
    ]

    # Plot the results
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

    Args:
        rf_model: Trained RandomForestClassifier model.
        X_test: DataFrame of test features.
        y_test: Series or array of test labels.
        feature_name: Name of the feature to analyze.
        n_iter: Number of permutation iterations (default: 10).
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


def show_interval_permutation_importance(rf_model, X_test, y_test, feature_name, num_intervals=20):
    """
    Displays interval-based feature importance using permutation within intervals.

    Args:
        rf_model: Trained RandomForestClassifier model.
        X_test: DataFrame of test features.
        y_test: Series or array of test labels.
        feature_name: Name of the feature to analyze.
        num_intervals: Number of intervals to partition the feature (default: 20).
    """
    if feature_name not in X_test.columns:
        st.error(f"Feature '{feature_name}' not found in the dataset.")
        return

    intervals = pd.qcut(X_test[feature_name], q=num_intervals, duplicates="drop")
    interval_labels = []
    importance_scores = []

    baseline_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

    for interval in intervals.unique():
        X_modified_p1 = X_test.copy()
        X_modified_p2 = X_test.copy()
        mid_left = interval.left
        mid_right = interval.right

        X_modified_p1[feature_name] = mid_left
        X_modified_p2[feature_name] = mid_right

        accuracy_p1 = accuracy_score(y_test, rf_model.predict(X_modified_p1))
        accuracy_p2 = accuracy_score(y_test, rf_model.predict(X_modified_p2))

        avg_importance = (baseline_accuracy - accuracy_p1 + baseline_accuracy - accuracy_p2) / 2
        importance_scores.append(avg_importance)
        interval_labels.append(f"{mid_left:.2e} - {mid_right:.2e}")

    # Plot interval feature importance scores
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(interval_labels, importance_scores, color="orange")
    ax.set_xlabel("Interval Range")
    ax.set_ylabel("Feature Importance (Permutation)")
    ax.set_title(f"Interval-Based Feature Importance for '{feature_name}'")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)



def show_2d_heatmap_interaction(rf_model, X_test, y_test, feature1, feature2, bins=10):
    """
    Displays a 2D heatmap of feature interactions using model accuracy.

    Args:
        rf_model: Trained RandomForestClassifier model.
        X_test: DataFrame of test features.
        y_test: Series or array of test labels.
        feature1: First feature name.
        feature2: Second feature name.
        bins: Number of bins for heatmap calculation (default: 10).
    """
    if feature1 not in X_test.columns or feature2 not in X_test.columns:
        st.error(f"One or both features '{feature1}' and '{feature2}' not found.")
        return

    min1, max1 = X_test[feature1].min(), X_test[feature1].max()
    min2, max2 = X_test[feature2].min(), X_test[feature2].max()

    x_edges = np.linspace(min1, max1, bins + 1)
    y_edges = np.linspace(min2, max2, bins + 1)

    heatmap = np.zeros((bins, bins))

    for i in range(bins):
        for j in range(bins):
            X_copy = X_test.copy()
            midpoint_x = (x_edges[i] + x_edges[i + 1]) / 2
            midpoint_y = (y_edges[j] + y_edges[j + 1]) / 2

            X_copy[feature1] = X_copy[feature1].apply(
                lambda v: midpoint_x if x_edges[i] <= v <= x_edges[i + 1] else v
            )
            X_copy[feature2] = X_copy[feature2].apply(
                lambda v: midpoint_y if y_edges[j] <= v <= y_edges[j + 1] else v
            )

            y_pred = rf_model.predict(X_copy)
            heatmap[i, j] = accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        heatmap,
        cmap="coolwarm",
        xticklabels=np.round(x_edges, 2),
        yticklabels=np.round(y_edges, 2),
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title("2D Feature Interaction Heatmap")
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
    features_path = 'data/lucas_organic_carbon_training_and_test_data_corrupted.csv'
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
        feature_importances = rf_model.feature_importances_
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
        
        # Interval Impact
        show_interval_impact(rf_model, X_test, y_test, feature_importances, bins=10)
        
        # 2D Heatmap
        show_2d_heatmap(rf_model, X_test, y_test, feature_importances, bins=10)
        chosen_feature = "2257" 
        show_substitution_analysis(rf_model, X_test, y_test, feature_name="2257")
        show_permutation_importance(rf_model, X_test, y_test, feature_name="2257")
        show_interval_permutation_importance(rf_model, X_test, y_test, feature_name="2257")
        show_2d_heatmap_interaction(rf_model, X_test, y_test, feature1="2257", feature2="2252")

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
            
        feature_importances = rf_model.feature_importances_
        # Button for Interval Impact
        if st.button("Show Interval Impact"):
            show_interval_impact(rf_model, X_test, y_test, feature_importances, bins=10)


        # Button for 2D Heatmap
        if st.button("Show 2D Heatmap"):
            show_2d_heatmap(rf_model, X_test, y_test, feature_importances, bins=10)
            
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
            show_substitution_analysis(chosen_feature, rf_model, X_test, y_test)

        if st.button("2D Feature Interaction Heatmap"):
            show_2d_heatmap_interaction("2257.0", "2252.0", rf_model, X_test, y_test, bins=10)
            
        if st.button("Interval Permutation Importance"):
            chosen_feature = "2257.0"
            show_interval_permutation_importance(chosen_feature, rf_model, X_test, y_test, num_intervals=20)

        if st.button("Permutation Importance"):
            chosen_feature = "2257.0" 
            show_permutation_importance(chosen_feature, rf_model, X_test, y_test, n_iter=10)

    else:
        st.info("Please train the Random Forest model first.")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()