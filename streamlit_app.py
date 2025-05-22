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

# Page configuration
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

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

# Sample data for demonstration (you should replace this with your actual training data)
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration purposes"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'loan_amnt': np.random.randint(1000, 50000, n_samples),
        'term': np.random.choice([36, 60], n_samples),
        'annual_inc': np.random.randint(20000, 150000, n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'emp_length': np.random.randint(0, 11, n_samples),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples),
        'verification_status': np.random.choice(['Verified', 'Source Verified', 'Not Verified'], n_samples),
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'other'], n_samples),
        'delinq_2yrs': np.random.randint(0, 5, n_samples),
        'inq_last_6mths': np.random.randint(0, 10, n_samples),
        'open_acc': np.random.randint(1, 25, n_samples),
        'pub_rec': np.random.randint(0, 5, n_samples),
        'revol_bal': np.random.randint(0, 100000, n_samples),
        'revol_util': np.random.uniform(0, 100, n_samples),
        'total_acc': np.random.randint(1, 50, n_samples),
    }
    
    # Create loan_status based on some logic
    loan_status = []
    for i in range(n_samples):
        # Simple logic for demonstration
        risk_score = (data['dti'][i] / 40) + (data['delinq_2yrs'][i] / 5) - (data['annual_inc'][i] / 150000)
        loan_status.append(1 if risk_score < 0.3 else 0)  # 1 = Approved, 0 = Rejected
    
    data['loan_status'] = loan_status
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the data for training"""
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        if col != 'loan_status':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    return df_processed, label_encoders

def train_models(df):
    """Train multiple models and return the best one"""
    df_processed, label_encoders = preprocess_data(df)
    
    X = df_processed.drop('loan_status', axis=1)
    y = df_processed['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(random_state=42),
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
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    # Find best model based on F1 score
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['f1'])
    best_model = model_results[best_model_name]['model']
    
    return best_model, model_results, label_encoders, X.columns.tolist()

# Page 1: Model Training
if page == "Model Training":
    st.header("🔧 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Data")
        
        option = st.radio("Choose data source:", ["Use Sample Data", "Upload CSV File"])
        
        if option == "Upload CSV File":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded data shape:", df.shape)
                st.write(df.head())
            else:
                st.info("Please upload a CSV file to proceed with training.")
                df = None
        else:
            df = load_sample_data()
            st.write("Sample data shape:", df.shape)
            st.write(df.head())
    
    with col2:
        st.subheader("Training Controls")
        
        if st.button("🚀 Train Models", type="primary"):
            if df is not None:
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        best_model, model_results, label_encoders, feature_names = train_models(df)
                        
                        st.session_state.best_model = best_model
                        st.session_state.model_performance = model_results
                        st.session_state.label_encoders = label_encoders
                        st.session_state.feature_names = feature_names
                        st.session_state.model_trained = True
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Display best model info
                        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['f1'])
                        st.info(f"🏆 Best Model: {best_model_name} (F1 Score: {model_results[best_model_name]['f1']:.4f})")
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
            else:
                st.error("Please provide training data first.")
    
    if st.session_state.model_trained:
        st.success("✅ Models are ready for prediction!")

# Page 2: Single Prediction
elif page == "Single Prediction":
    st.header("🎯 Single Loan Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first in the 'Model Training' page.")
    else:
        st.subheader("Enter Loan Application Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=25000)
            term = st.selectbox("Term (months)", [36, 60])
            annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000)
            dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, 15.0)
            emp_length = st.slider("Employment Length (years)", 0, 15, 5)
        
        with col2:
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
            verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])
            purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "other"])
            delinq_2yrs = st.number_input("Delinquencies (2 years)", min_value=0, max_value=10, value=0)
            inq_last_6mths = st.number_input("Inquiries (6 months)", min_value=0, max_value=20, value=1)
        
        with col3:
            open_acc = st.number_input("Open Accounts", min_value=1, max_value=50, value=10)
            pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)
            revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=15000)
            revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0)
            total_acc = st.number_input("Total Accounts", min_value=1, max_value=100, value=20)
        
        if st.button("🔮 Predict Loan Status", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'loan_amnt': [loan_amnt],
                'term': [term],
                'annual_inc': [annual_inc],
                'dti': [dti],
                'emp_length': [emp_length],
                'home_ownership': [home_ownership],
                'verification_status': [verification_status],
                'purpose': [purpose],
                'delinq_2yrs': [delinq_2yrs],
                'inq_last_6mths': [inq_last_6mths],
                'open_acc': [open_acc],
                'pub_rec': [pub_rec],
                'revol_bal': [revol_bal],
                'revol_util': [revol_util],
                'total_acc': [total_acc]
            })
            
            # Encode categorical variables
            for col in ['home_ownership', 'verification_status', 'purpose']:
                if col in st.session_state.label_encoders:
                    try:
                        input_data[col] = st.session_state.label_encoders[col].transform(input_data[col])
                    except ValueError:
                        # Handle unseen categories
                        input_data[col] = 0
            
            # Make prediction
            prediction = st.session_state.best_model.predict(input_data)[0]
            prediction_proba = st.session_state.best_model.predict_proba(input_data)[0] if hasattr(st.session_state.best_model, 'predict_proba') else None
            
            # Display result
            if prediction == 1:
                st.markdown('<div class="prediction-result approved">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-result rejected">❌ LOAN REJECTED</div>', unsafe_allow_html=True)
            
            if prediction_proba is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Approval Probability", f"{prediction_proba[1]:.2%}")
                with col2:
                    st.metric("Rejection Probability", f"{prediction_proba[0]:.2%}")

# Page 3: Batch Prediction
elif page == "Batch Prediction":
    st.header("📊 Batch Loan Prediction")
    
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
                    # Preprocess the batch data
                    batch_processed = batch_df.copy()
                    
                    # Encode categorical variables
                    categorical_cols = ['home_ownership', 'verification_status', 'purpose']
                    for col in categorical_cols:
                        if col in batch_processed.columns and col in st.session_state.label_encoders:
                            try:
                                batch_processed[col] = st.session_state.label_encoders[col].transform(batch_processed[col].astype(str))
                            except ValueError:
                                # Handle unseen categories
                                batch_processed[col] = 0
                    
                    # Make predictions
                    predictions = st.session_state.best_model.predict(batch_processed)
                    
                    # Add predictions to the original dataframe
                    result_df = batch_df.copy()
                    result_df['Predicted_Loan_Status'] = predictions
                    result_df['Prediction_Label'] = result_df['Predicted_Loan_Status'].map({1: 'Approved', 0: 'Rejected'})
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.write(result_df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Applications", len(result_df))
                    with col2:
                        approved_count = sum(predictions)
                        st.metric("Approved", approved_count)
                    with col3:
                        rejected_count = len(predictions) - approved_count
                        st.metric("Rejected", rejected_count)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="loan_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 6))
                    result_df['Prediction_Label'].value_counts().plot(kind='bar', ax=ax)
                    plt.title('Batch Prediction Results')
                    plt.ylabel('Count')
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during batch prediction: {str(e)}")

# Page 4: Model Performance
elif page == "Model Performance":
    st.header("📈 Model Performance Summary")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first in the 'Model Training' page.")
    else:
        st.subheader("Model Comparison")
        
        # Create performance comparison dataframe
        performance_data = []
        for model_name, results in st.session_state.model_performance.items():
            performance_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1']
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.round(4)
        
        # Display performance table
        st.dataframe(performance_df, use_container_width=True)
        
        # Find best model
        best_model_name = performance_df.loc[performance_df['F1 Score'].idxmax(), 'Model']
        st.success(f"🏆 Best Model: {best_model_name}")
        
        # Performance metrics visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            performance_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='skyblue', legend=False)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion Matrix for Best Model
        st.subheader(f"Confusion Matrix - {best_model_name}")
        
        best_results = st.session_state.model_performance[best_model_name]
        cm = best_results['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader(f"Classification Report - {best_model_name}")
        
        y_test = best_results['y_test']
        y_pred = best_results['y_pred']
        
        report = classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4))

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
