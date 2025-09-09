import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import pickle

# ---------------- 1️⃣ Load CSV ----------------
df = pd.read_csv("train.csv")
train_df = df[df["Category"].notnull()].copy()
if train_df.empty:
    raise ValueError("No valid rows with Category found for training.")

X = train_df[["Description", "Amount"]]
y = train_df["Category"]

# ---------------- 2️⃣ Preprocessing ----------------
def reshape_amount(X):
    """
    Reshape numeric column to 2D array for sklearn.
    Works with both DataFrame/Series and numpy array.
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.values.reshape(-1, 1)
    elif isinstance(X, np.ndarray):
        return X.reshape(-1, 1)
    else:
        raise ValueError(f"Unexpected input type: {type(X)}")

# Numeric preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("reshape", FunctionTransformer(reshape_amount)),
    ("scaler", StandardScaler())
])

# Text preprocessing
text_transformer = TfidfVectorizer(
    ngram_range=(1,2),  # uni-grams + bi-grams
    stop_words="english",
    max_features=5000
)

# Combine columns
preprocessor = ColumnTransformer(
    transformers=[
        ("desc", text_transformer, "Description"),
        ("amt", numeric_transformer, ["Amount"])
    ]
)

# ---------------- 3️⃣ Pipeline ----------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# ---------------- 4️⃣ Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- 5️⃣ Hyperparameter Tuning (Optional) ----------------
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10]
}

search = RandomizedSearchCV(
    pipeline, param_distributions=param_grid,
    n_iter=5, cv=3, scoring="accuracy", n_jobs=-1, random_state=42, verbose=1
)

search.fit(X_train, y_train)

print("\n✅ Best Parameters:", search.best_params_)

# ---------------- 6️⃣ Evaluate ----------------
y_pred = search.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- 7️⃣ Save Model ----------------
with open("expense_model2.pkl", "wb") as f:
    pickle.dump(search.best_estimator_, f)

print("\n✅ Model trained and saved as expense_model2.pkl")
