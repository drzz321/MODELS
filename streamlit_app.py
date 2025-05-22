import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import io
import base64
import os

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Prediction App",
    page_icon="🔮",  # Example: Crystal Ball emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load deployment artifacts ---
# Remove all pre-trained model loading logic, require user upload and training

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}

# Header
st.markdown('<h1 class="main-header">💰 Loan Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Model Training", "Single Prediction", "Batch Prediction", "Model Performance"])

# --- Flexible CSV structure: user selects target and categorical columns ---
def preprocess_data(df, target_col, categorical_cols):
    """Preprocess the data for training"""
    df_processed = df.copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    return df_processed, label_encoders

def train_models(df, target_col, categorical_cols):
    df_processed, label_encoders = preprocess_data(df, target_col, categorical_cols)
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }
    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred
        }
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['f1'])
    best_model = model_results[best_model_name]['model']
    return best_model, model_results, label_encoders, X.columns.tolist(), best_model_name

# Page 1: Model Training
if page == "Model Training":
    st.header("🔧 Model Training")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Training Data")
        st.info("Upload any CSV file for classification. Select your target and categorical columns.")
        uploaded_file = st.file_uploader("Upload your CSV file for training", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded data shape:", df.shape)
            st.write(df.head())
            # User selects target column
            target_col = st.selectbox("Select the target column (label)", df.columns)
            # User selects categorical columns
            default_cats = list(df.select_dtypes(include=['object']).columns)
            if target_col in default_cats:
                default_cats.remove(target_col)
            categorical_cols = st.multiselect("Select categorical columns", df.columns, default=default_cats)
        else:
            st.warning("Please upload a CSV file to proceed with training.")
            df = None
            target_col = None
            categorical_cols = []
    with col2:
        st.subheader("Training Controls")
        if st.button("🚀 Train Models", type="primary"):
            if df is not None and target_col is not None:
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        best_model, model_results, label_encoders, feature_names, best_model_name = train_models(df, target_col, categorical_cols)
                        st.session_state.best_model = best_model
                        st.session_state.model_performance = model_results
                        st.session_state.label_encoders = label_encoders
                        st.session_state.feature_names = feature_names
                        st.session_state.model_trained = True
                        st.session_state.target_col = target_col
                        st.session_state.categorical_cols = categorical_cols
                        st.success("✅ Models trained successfully!")
                        st.info(f"🏆 Best Model: {best_model_name} (F1 Score: {model_results[best_model_name]['f1']:.4f})")
                        st.markdown("**Accuracy:**")
                        st.write(f"{model_results[best_model_name]['accuracy']:.4f}")
                        st.markdown("**Confusion Matrix:**")
                        cm = model_results[best_model_name]['confusion_matrix']
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        plt.title(f'Confusion Matrix - {best_model_name}')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        st.pyplot(fig)
                        if best_model_name == "Random Forest":
                            st.markdown("**Feature Importances (Random Forest):**")
                            importances = best_model.feature_importances_
                            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                            feat_df = feat_df.sort_values(by='Importance', ascending=False)
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax)
                            ax.set_title('Random Forest - Feature Importances')
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
            else:
                st.error("Please provide training data and select target column.")
    if st.session_state.model_trained:
        st.success("✅ Models are ready for prediction!")

# Page 2: Single Prediction
elif page == "Single Prediction":
    st.header("🎯 Single Prediction")
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first in the 'Model Training' page.")
    else:
        st.subheader("Enter Feature Values for Prediction")
        input_data = {}
        for col in st.session_state.feature_names:
            if col in st.session_state.categorical_cols:
                le = st.session_state.label_encoders[col]
                options = list(le.classes_)
                input_data[col] = st.selectbox(f"{col}", options)
            else:
                input_data[col] = st.number_input(f"{col}")
        if st.button("🔮 Predict", type="primary"):
            input_df = pd.DataFrame([input_data])
            for col in st.session_state.categorical_cols:
                le = st.session_state.label_encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col])
                except Exception:
                    input_df[col] = 0
            input_df = input_df.reindex(columns=st.session_state.feature_names, fill_value=0)
            prediction = st.session_state.best_model.predict(input_df)[0]
            prediction_proba = st.session_state.best_model.predict_proba(input_df)[0] if hasattr(st.session_state.best_model, 'predict_proba') else None
            st.markdown(f"**Predicted Class:** `{prediction}`")
            if prediction_proba is not None:
                st.write("Class Probabilities:")
                st.write(dict(zip(st.session_state.best_model.classes_, prediction_proba)))

# Page 3: Batch Prediction
elif page == "Batch Prediction":
    st.header("📊 Batch Prediction")
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first in the 'Model Training' page.")
    else:
        st.subheader("Upload CSV File for Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded data shape:", batch_df.shape)
            st.write("Preview of uploaded data:")
            st.write(batch_df.head())
            if st.button("🔮 Make Batch Predictions", type="primary"):
                try:
                    batch_processed = batch_df.copy()
                    for col in st.session_state.categorical_cols:
                        if col in batch_processed.columns and col in st.session_state.label_encoders:
                            le = st.session_state.label_encoders[col]
                            try:
                                batch_processed[col] = le.transform(batch_processed[col].astype(str))
                            except Exception:
                                batch_processed[col] = 0
                    if st.session_state.target_col in batch_processed.columns:
                        batch_processed = batch_processed.drop(columns=[st.session_state.target_col])
                    batch_processed = batch_processed.reindex(columns=st.session_state.feature_names, fill_value=0)
                    predictions = st.session_state.best_model.predict(batch_processed)
                    result_df = batch_df.copy()
                    result_df['Predicted_Class'] = predictions
                    st.subheader("Prediction Results")
                    st.write(result_df)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Samples", len(result_df))
                    with col2:
                        st.metric("Unique Predicted Classes", len(set(predictions)))
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    fig, ax = plt.subplots(figsize=(8, 6))
                    result_df['Predicted_Class'].value_counts().plot(kind='bar', ax=ax)
                    plt.title('Batch Prediction Results')
                    plt.ylabel('Count')
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error during batch prediction: {str(e)}")
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Made with ❤️ using Streamlit | Loan Prediction System
    </div>
    """, 
    unsafe_allow_html=True
)
