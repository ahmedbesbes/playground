import streamlit as st
from sklearn.ensemble import RandomForestClassifier


def rf_param_selector():

    criterion = st.selectbox("criterion", ["gini", "entropy"])
    n_estimators = st.number_input("n_estimators", 50, 300, 100, 10)
    max_depth = st.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

    params = {
        "criterion": criterion,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }

    model = RandomForestClassifier(**params)
    return model


def rf_code_snippet():
    return """
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    RandomForestClassifier(...)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """