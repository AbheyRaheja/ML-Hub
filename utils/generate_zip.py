import os
import pickle
import zipfile

def get_model_module(model_name):
    # Known classifiers and their modules
    model_modules = {
        "LogisticRegression": "sklearn.linear_model",
        "RidgeClassifier": "sklearn.linear_model",
        "SVC": "sklearn.svm",
        "LinearSVC": "sklearn.svm",
        "DecisionTreeClassifier": "sklearn.tree",
        "RandomForestClassifier": "sklearn.ensemble",
        "ExtraTreesClassifier": "sklearn.ensemble",
        "GradientBoostingClassifier": "sklearn.ensemble",
        "AdaBoostClassifier": "sklearn.ensemble",
        "BaggingClassifier": "sklearn.ensemble",
        "KNeighborsClassifier": "sklearn.neighbors",
        "GaussianNB": "sklearn.naive_bayes",
        "BernoulliNB": "sklearn.naive_bayes",
        "MultinomialNB": "sklearn.naive_bayes",
        "MLPClassifier": "sklearn.neural_network",
        "QuadraticDiscriminantAnalysis": "sklearn.discriminant_analysis",
        "LinearDiscriminantAnalysis": "sklearn.discriminant_analysis",
        # Add more if needed
    }

    return model_modules.get(model_name, "sklearn")  # fallback if unknown


def generate_zip(model, X, y, model_name, hyperparams, save_path="trained_model.zip"):
    temp_dir = "temp_export"
    os.makedirs(temp_dir, exist_ok=True)

    # Save model.pkl
    model_path = os.path.join(temp_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save dataset as CSV
    df = X.copy()
    df["target"] = y
    data_path = os.path.join(temp_dir, "data.csv")
    df.to_csv(data_path, index=False)

    # Save training code
    code_path = os.path.join(temp_dir, "train_pipeline.py")
    with open(code_path, "w") as f:
        f.write(f"""\
# Auto-generated training script
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from {get_model_module(model_name)} import {model_name}

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = {model_name}({dict_to_args(hyperparams)})
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {{acc:.4f}}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
""")

    # Create .zip
    with zipfile.ZipFile(save_path, "w") as zipf:
        zipf.write(model_path, arcname="model.pkl")
        zipf.write(data_path, arcname="data.csv")
        zipf.write(code_path, arcname="train_pipeline.py")

    return save_path

def dict_to_args(d):
    # Convert Python dict to sklearn-style argument string
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
