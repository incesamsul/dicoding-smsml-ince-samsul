from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":
    data = load_breast_cancer(as_frame=True).frame
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"), {"n_estimators":[80,160], "max_depth":[None,6,10], "min_samples_leaf":[1,2]}, scoring="f1", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(classification_report(y_test, grid.best_estimator_.predict(X_test)))
