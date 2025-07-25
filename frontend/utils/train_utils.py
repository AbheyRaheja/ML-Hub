import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def train_model(model_class, hyperparams, df, target_col):
    try:
        X = df.drop(columns = [target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

        model = model_class(**hyperparams)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        return {
            "success": True,
            "model":model,
            "accuracy":acc,
            "X_columns": list(X.columns),
            "X_test": X_test,
            "y_test": y_test
        }
    
    except Exception as e:
        return{
            "success": False,
            "error":str(e)
        }
    
def plot_confusion_matrix(model, X_test, y_test, figsize=(4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    return fig

def plot_scatter(X, y, figsize=(5, 4)):
    import matplotlib.pyplot as plt

    # If more than 2 features, take first 2
    if X.shape[1] > 2:
        X = X.iloc[:, :2]

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap="viridis", edgecolor="k", alpha=0.7)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_title("Scatter Plot (first 2 features)")
    return fig
