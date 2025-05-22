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
st.markdown('<h1 class="main-header"> Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Model Training"])
# page = st.sidebar.selectbox("Choose a page", ["Model Training", "Prediction"])

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
    # Store X_train in session_state for scaling during prediction
    st.session_state['X_train'] = X_train
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=False, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }
    model_results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
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
    st.markdown("""
    **Method 1: Use the dropdowns below to select your columns.**
    """)
    # Dropdown for target column
    target_col_dropdown = st.selectbox("Select the target column (label)", df.columns)
    # Multiselect for categorical columns
    default_cats = list(df.select_dtypes(include=['object']).columns)
    if target_col_dropdown in default_cats:
        default_cats.remove(target_col_dropdown)
    categorical_cols_multiselect = st.multiselect("Select categorical columns", df.columns, default=default_cats)
    st.markdown("---")
    st.markdown("""
    **Method 2: Or type the column names directly (comma-separated for categorical columns).**
    If you use the text input, it will override the dropdown/multiselect selection above.
    """)
    # Text input for target column (optional, overrides dropdown if filled)
    target_col_text = st.text_input("Enter the target column name (optional, overrides dropdown)")
    target_col = target_col_text.strip() if target_col_text.strip() else target_col_dropdown
    # Text input for categorical columns (optional, overrides multiselect if filled)
    cat_col_text = st.text_input("Enter categorical column names (comma-separated, optional, overrides selection)")
    if cat_col_text.strip():
        categorical_cols = [col.strip() for col in cat_col_text.split(',') if col.strip() in df.columns]
    else:
        categorical_cols = categorical_cols_multiselect
    # Training controls
    st.subheader("Training Controls")
    # Only enable training if all classes have at least 2 samples
    class_counts = df[target_col].value_counts()
    can_train = (class_counts >= 2).all()
    if not can_train:
        st.warning("Training is disabled because at least one class in your target column has fewer than 2 samples.")
    if st.button("🚀 Train Models", type="primary", disabled=not can_train):
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
                    st.session_state.X_train = df.drop(target_col, axis=1)
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
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.error("Please provide training data and select target column.")
    if st.session_state.model_trained:
        st.success("✅ Models are ready for prediction!")

# # --- Prediction Page ---
# if page == "Prediction":
#     st.header("🔮 Prediction")
#     if not st.session_state.get("model_trained", False):
#         st.warning("Please train a model first on the 'Model Training' page.")
#         st.stop()
#     best_model = st.session_state.best_model
#     label_encoders = st.session_state.label_encoders
#     feature_names = st.session_state.feature_names
#     target_col = st.session_state.target_col
#     categorical_cols = st.session_state.categorical_cols
#     best_model_name = None
#     for name, res in st.session_state.model_performance.items():
#         if res['model'] == best_model:
#             best_model_name = name
#             break
#     st.info(f"Using best model: {best_model_name}")
#     st.markdown(f"**F1 Score:** {st.session_state.model_performance[best_model_name]['f1']:.4f}")
#     st.markdown("---")
#     pred_mode = st.radio("Choose prediction mode", ["Manual Input", "Batch CSV Upload"])
#     if pred_mode == "Manual Input":
#         st.subheader("Manual Input Prediction")
#         manual_input = {}
#         for col in feature_names:
#             if col in categorical_cols:
#                 options = label_encoders[col].classes_.tolist()
#                 manual_input[col] = st.selectbox(f"{col}", options, key=f"pred_manual_{col}")
#             else:
#                 manual_input[col] = st.number_input(f"{col}", key=f"pred_manual_{col}")
#         if st.button("Predict", key="manual_predict_btn"):
#             input_df = pd.DataFrame([manual_input])
#             # Encode categorical columns
#             for col in categorical_cols:
#                 le = label_encoders[col]
#                 input_df[col] = le.transform([input_df[col][0]])
#             # Scale if best model is Logistic Regression
#             if best_model_name == "Logistic Regression":
#                 scaler = StandardScaler()
#                 # Fit scaler on training data (stored in session_state)
#                 X_train = st.session_state.get('X_train')
#                 if X_train is not None:
#                     scaler.fit(X_train)
#                     input_df = scaler.transform(input_df)
#             pred = best_model.predict(input_df)[0]
#             proba = None
#             if hasattr(best_model, "predict_proba"):
#                 proba = best_model.predict_proba(input_df)[0]
#             st.success(f"Prediction: {pred}")
#             if proba is not None:
#                 st.write("Prediction Probabilities:")
#                 class_labels = best_model.classes_
#                 st.write({str(label): float(p) for label, p in zip(class_labels, proba)})
#     else:
#         st.subheader("Batch Prediction (CSV Upload)")
#         batch_file = st.file_uploader("Upload CSV for batch prediction", type="csv", key="batch_pred_upload")
#         if batch_file is not None:
#             batch_df = pd.read_csv(batch_file)
#             st.write("Uploaded batch data shape:", batch_df.shape)
#             st.write(batch_df.head())
#             # Check columns
#             missing_cols = [col for col in feature_names if col not in batch_df.columns]
#             if missing_cols:
#                 st.error(f"Missing columns in uploaded file: {missing_cols}")
#             else:
#                 # Encode categorical columns
#                 for col in categorical_cols:
#                     le = label_encoders[col]
#                     batch_df[col] = le.transform(batch_df[col].astype(str))
#                 # Scale if best model is Logistic Regression
#                 if best_model_name == "Logistic Regression":
#                     scaler = StandardScaler()
#                     X_train = st.session_state.get('X_train')
#                     if X_train is not None:
#                         scaler.fit(X_train)
#                         batch_df[feature_names] = scaler.transform(batch_df[feature_names])
#                 preds = best_model.predict(batch_df[feature_names])
#                 proba = None
#                 if hasattr(best_model, "predict_proba"):
#                     proba = best_model.predict_proba(batch_df[feature_names])
#                 result_df = batch_df.copy()
#                 result_df['Prediction'] = preds
#                 if proba is not None:
#                     for idx, class_label in enumerate(best_model.classes_):
#                         result_df[f'Prob_{class_label}'] = proba[:, idx]
#                 st.write("Predictions:")
#                 st.dataframe(result_df, use_container_width=True)
#                 # Download link
#                 csv = result_df.to_csv(index=False)
#                 b64 = base64.b64encode(csv.encode()).decode()
#                 href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
#                 st.markdown(href, unsafe_allow_html=True)
#                 if proba is not None:
#                     st.info("Each 'Prob_Class' column shows the model's confidence (from 0 to 1) that the row belongs to that class. For example, a value of 0.54 for 'Prob_1' means a 54% chance the row is class 1. The 'Prediction' column shows the most likely class for each row.")
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
