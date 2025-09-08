import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# 1️⃣ Load CSV
df = pd.read_csv("train.csv")

# Only use rows where Category is not empty
train_df = df[df["Category"].notnull()].copy()
if train_df.empty:
    raise ValueError("No valid rows with Category found for training.")

X = train_df[["Description", "Amount"]]
y = train_df["Category"]

# 2️⃣ Preprocessing
def reshape_amount(X):
    return X.values.reshape(-1, 1)

preprocessor = ColumnTransformer(
    transformers=[
        ("desc", TfidfVectorizer(), "Description"),
        ("amt", FunctionTransformer(reshape_amount), ["Amount"])
    ]
)

# 3️⃣ Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# 4️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train
pipeline.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = pipeline.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# 7️⃣ Save model
with open("expense_model2.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\n✅ Model trained and saved as expense_model2.pkl")
