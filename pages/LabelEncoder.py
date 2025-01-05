import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from services.data_loader import load_data

def substitute_and_calculate_accuracy(feature, replacement_value, model, X, y):
    X_modified = X.copy()
    X_modified[feature] = replacement_value
    y_pred_modified = model.predict(X_modified)
    return accuracy_score(y, y_pred_modified)

def permutation_importance(feature, model, X, y, n_iter=10):
    baseline_accuracy = accuracy_score(y, model.predict(X))
    accuracy_drop = []

    for _ in range(n_iter):
        X_permuted = X.copy()
        X_permuted[feature] = shuffle(X_permuted[feature].values)
        permuted_accuracy = accuracy_score(y, model.predict(X_permuted))
        accuracy_drop.append(baseline_accuracy - permuted_accuracy)

    return np.mean(accuracy_drop)

def interval_permutation_importance(feature, model, X, y, num_intervals=20):
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

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(interval_labels, importance_scores, color='orange')
    ax.set_xlabel('Interval Range')
    ax.set_ylabel('Feature Importance Score')
    ax.set_title(f'Interval-Based Feature Importance for {feature}')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_2d_heatmap(feature1, feature2, model, X, y, bins=10):
    min1, max1 = X[feature1].min(), X[feature1].max()
    min2, max2 = X[feature2].min(), X[feature2].max()

    x_edges = np.linspace(min1, max1, bins + 1)
    y_edges = np.linspace(min2, max2, bins + 1)
    heatmap = np.zeros((bins, bins))

    for i in range(bins):
        for j in range(bins):
            X_copy = X.copy()
            X_copy[feature1] = (x_edges[i] + x_edges[i + 1]) / 2
            X_copy[feature2] = (y_edges[j] + y_edges[j + 1]) / 2
            y_pred = model.predict(X_copy)
            heatmap[i, j] = accuracy_score(y, y_pred)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(heatmap, annot=False, cmap='coolwarm',
                xticklabels=[f"{x_edges[i]:.2f}" for i in range(len(x_edges) - 1)],
                yticklabels=[f"{y_edges[j]:.2f}" for j in range(len(y_edges) - 1)], ax=ax)
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title('2D Feature Interaction Heatmap')
    st.pyplot(fig)

def main():
    st.title("Streamlit App for LUCAS Organic Carbon Analysis")

    # Load Data
    X_data, y_data = load_data()

    st.header("Dataset Overview")
    st.write("Features Dataset", X_data.head())
    st.write("Target Dataset", y_data.head())

    # Encode Target Variable
    le = LabelEncoder()
    y_data['target'] = le.fit_transform(y_data.values.ravel())

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data['target'], test_size=0.3, random_state=42)

    # Train Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    st.success("Random Forest Model Trained Successfully!")

    # Feature Importances
    st.header("Feature Importances")
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(16, 6))
    feature_importances.head(20).plot(kind='bar', ax=ax)
    ax.set_title('Top 20 Features')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Score')
    st.pyplot(fig)

    # Substitution Analysis
    st.header("Substitution Analysis")
    selected_feature = "2257.0"
    if selected_feature in X_test.columns:
        min_value, mean_value, max_value = X_test[selected_feature].min(), X_test[selected_feature].mean(), X_test[selected_feature].max()
        replacement_values = [min_value, mean_value, max_value]
        accuracies = [substitute_and_calculate_accuracy(selected_feature, v, rf_model, X_test, y_test) for v in replacement_values]

        st.write(f"Selected Feature: {selected_feature}")
        st.write(f"Min: {min_value}, Mean: {mean_value}, Max: {max_value}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(["Min", "Mean", "Max"], accuracies, marker='o')
        ax.set_title(f"Substitution Analysis for {selected_feature}")
        ax.set_ylabel("Accuracy Score")
        st.pyplot(fig)

    # Permutation Importance
    st.header("Permutation Importance")
    perm_importance_score = permutation_importance(selected_feature, rf_model, X_test, y_test)
    st.write(f"Permutation importance score for {selected_feature}: {perm_importance_score:.4f}")

    # Interval Analysis
    st.header("Interval-Based Feature Importance")
    interval_permutation_importance(selected_feature, rf_model, X_test, y_test, num_intervals=20)

    # 2D Heatmap
    st.header("2D Heatmap")
    feature1, feature2 = "2257.0", "2252.0"
    if feature1 in X_test.columns and feature2 in X_test.columns:
        plot_2d_heatmap(feature1, feature2, rf_model, X_test, y_test)

    # Confusion Matrix
    st.header("Confusion Matrix")
    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

    # Classification Report
    st.header("Classification Report")
    st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
