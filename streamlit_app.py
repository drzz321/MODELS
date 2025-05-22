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
    page_icon="💰",
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
page = st.sidebar.selectbox("Choose a page", ["Model Training"])

# --- Flexible CSV structure: user selects target and categorical columns ---
def preprocess_data(df, target_col, categorical_cols):
    """Preprocess the data for training, including missing value imputation"""
    df_processed = df.copy()
    # Impute missing values
    for col in df_processed.columns:
        if col == target_col:
            continue
        if col in categorical_cols:
            mode = df_processed[col].mode()
            if not mode.empty:
                df_processed[col] = df_processed[col].fillna(mode[0])
            else:
                df_processed[col] = df_processed[col].fillna('Unknown')
        else:
            median = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median)
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
    # --- Always show CSV upload at the top ---
    st.subheader("Step 1: Upload Your CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file for training (required)", type="csv", key="main_csv_upload")
    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed. The rest of the options will appear after upload.")
        st.stop()
    # --- After upload, show the rest of the UI ---
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded data shape:", df.shape)
    st.write(df.head())
    # User selects target column
    st.subheader("Step 2: Select Target and Categorical Columns")
    target_col = st.selectbox("Select the target column (label)", df.columns)
    # Warn if any class has <2 samples
    class_counts = df[target_col].value_counts()
    if (class_counts < 2).any():
        st.error(f"The selected target column contains at least one class with fewer than 2 samples. Please check your data or choose a different target column.\nClass counts:\n{class_counts.to_string()}")
        st.stop()
    default_cats = list(df.select_dtypes(include=['object']).columns)
    if target_col in default_cats:
        default_cats.remove(target_col)
    categorical_cols = st.multiselect("Select categorical columns", df.columns, default=default_cats)
    # Manual input preview
    st.markdown("---")
    st.subheader("Step 3: Manual Input Example (Optional)")
    st.info("You can manually input feature values below to see how a sample row would look. This does not make a prediction until after training.")
    manual_input = {}
    for col in df.columns:
        if col == target_col:
            continue
        if col in categorical_cols:
            options = df[col].unique().tolist()
            manual_input[col] = st.selectbox(f"{col}", options, key=f"manual_{col}")
        else:
            manual_input[col] = st.number_input(f"{col}", key=f"manual_{col}")
    st.write("Your manual input as a DataFrame:")
    st.write(pd.DataFrame([manual_input]))
    # Training controls
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
                    # Model performance for all models
                    st.subheader("Model Performance (All Models)")
                    performance_data = []
                    for model_name, results in model_results.items():
                        performance_data.append({
                            'Model': model_name,
                            'Accuracy': results['accuracy'],
                            'Precision': results['precision'],
                            'Recall': results['recall'],
                            'F1 Score': results['f1']
                        })
                    performance_df = pd.DataFrame(performance_data)
                    performance_df = performance_df.round(4)
                    st.dataframe(performance_df, use_container_width=True)
                    # Best model info
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
                    # --- User Input for Prediction ---
                    st.subheader("Try a Prediction with Your Model")
                    st.info("Enter feature values below to see a prediction from the best model.")
                    input_data = {}
                    for col in feature_names:
                        if col in categorical_cols:
                            le = label_encoders[col]
                            options = list(le.classes_)
                            input_data[col] = st.selectbox(f"{col}", options, key=f"input_{col}")
                        else:
                            input_data[col] = st.number_input(f"{col}", key=f"input_{col}")
                    if st.button("Predict with Best Model"):
                        input_df = pd.DataFrame([input_data])
                        for col in categorical_cols:
                            le = label_encoders[col]
                            try:
                                input_df[col] = le.transform(input_df[col])
                            except Exception:
                                input_df[col] = 0
                        input_df = input_df.reindex(columns=feature_names, fill_value=0)
                        prediction = best_model.predict(input_df)[0]
                        prediction_proba = best_model.predict_proba(input_df)[0] if hasattr(best_model, 'predict_proba') else None
                        st.markdown(f"**Predicted Class:** `{prediction}`")
                        if prediction_proba is not None:
                            st.write("Class Probabilities:")
                            st.write(dict(zip(best_model.classes_, prediction_proba)))
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.error("Please provide training data and select target column.")
    if st.session_state.model_trained:
        st.success("✅ Models are ready for prediction!")

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
