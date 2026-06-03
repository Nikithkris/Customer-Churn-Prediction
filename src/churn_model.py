import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv(
    "data/customer_churn.csv"
)

# Remove Customer ID
df.drop(
    columns=["customerID"],
    inplace=True
)

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

# Encode Categorical Variables
encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(
            df[column]
        )

# Split Data
X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

# Predictions
predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("=" * 50)
print("CUSTOMER CHURN PREDICTION")
print("=" * 50)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(report)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
accuracy = accuracy_score(
    y_test,
    predictions
)
cm = confusion_matrix(
    y_test,
    predictions
)

report = classification_report(
    y_test,
    predictions
)
