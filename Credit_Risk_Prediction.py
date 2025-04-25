import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load dataset
data = pd.read_csv("data/german_credit_data.csv", index_col=0)

# Handle missing values
data['Saving accounts'].fillna('unknown', inplace=True)
data['Checking account'].fillna('unknown', inplace=True)

# Cap outliers
numeric_cols = ['Age', 'Credit amount', 'Duration']
for col in numeric_cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data[col] = np.where(data[col] < lower, lower, data[col])
    data[col] = np.where(data[col] > upper, upper, data[col])

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target
data['Risk'] = data['Risk'].map({'good': 1, 'bad': 0})

# Save full input features before RFE
all_features = data.drop('Risk', axis=1).columns.tolist()
joblib.dump(all_features, "model/features.pkl")

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.close()

# Features and target
X = data.drop('Risk', axis=1)
y = data['Risk']

# Mutual Information
mi = mutual_info_classif(X, y)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
mi_series.plot(kind='bar')
plt.title("Mutual Information with Target")
plt.ylabel("Mutual Information")
plt.tight_layout()
plt.savefig("outputs/mutual_information.png")
plt.close()

# Recursive Feature Elimination (RFE)
logreg = LogisticRegression(max_iter=1000)
rfe = RFE(logreg, n_features_to_select=5)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print("Selected features for modeling:", selected_features.tolist())

# Save selected features for Streamlit app
joblib.dump(selected_features.tolist(), "model/selected_features.pkl")

# Use only selected features for training
X = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and hyperparameters
models = {
    "DecisionTree": (DecisionTreeClassifier(), {
        'max_depth': [3, 5, 10],
        'criterion': ['gini', 'entropy']
    }),
    "KNN": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    })
}

# Model training and selection
best_score = 0
best_model = None
best_model_name = ""

plt.figure(figsize=(10, 6))

for name, (model, params) in models.items():
    clf = GridSearchCV(model, params, cv=5, scoring='f1')
    clf.fit(X_train_scaled, y_train)

    print(f"\n{name} Best Params: {clf.best_params_}")
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # ROC Curve
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = clf.decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

    if clf.best_score_ > best_score:
        best_score = clf.best_score_
        best_model = clf.best_estimator_
        best_model_name = name

# Final ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_auc_comparison.png")
plt.close()

# Save model and assets
joblib.dump(best_model, f"model/best_model_{best_model_name}.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

print(f"\nBest model: {best_model_name} (saved as best_model_{best_model_name}.pkl)")
