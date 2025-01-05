from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

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
