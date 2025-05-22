import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your data
DATA_PATH = 'Loans.csv'
df = pd.read_csv(DATA_PATH)

# Drop columns if needed (as in your training script)
drop_cols = ['int_rate', 'credit_score']
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Prepare features and target
y = df['loan_status']
X = df.drop('loan_status', axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Save artifacts for each model
deploy_dir = 'streamlit_deploy_artifacts'
os.makedirs(deploy_dir, exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    cmatrix = confusion_matrix(y_test, y_pred)
    # Save model and metrics
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, os.path.join(deploy_dir, f'{safe_name}_model.joblib'))
    joblib.dump(metrics, os.path.join(deploy_dir, f'{safe_name}_metrics.joblib'))
    joblib.dump(cmatrix, os.path.join(deploy_dir, f'{safe_name}_cmatrix.joblib'))

# Save encoders and columns (shared)
joblib.dump(label_encoders, os.path.join(deploy_dir, 'label_encoders.joblib'))
joblib.dump(list(X_train.columns), os.path.join(deploy_dir, 'X_train_columns.joblib'))
print(f"Artifacts for all models saved to {deploy_dir}")
