import streamlit as st
from sklearn.linear_model import LogisticRegression


def lr_param_selector():

    solver = st.selectbox(
        "solver", options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
    )

    if solver in ["newton-cg", "lbfgs", "sag"]:
        penalties = ["l2", "none"]

    elif solver == "saga":
        penalties = ["l1", "l2", "none", "elasticnet"]

    elif solver == "liblinear":
        penalties = ["l1"]

    penalty = st.selectbox("penalty", options=penalties)
    C = st.number_input("C", 0.1, 2.0, 1.0, 0.1)
    max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100)

    params = {"solver": solver, "penalty": penalty, "C": C, "max_iter": max_iter}

    model = LogisticRegression(**params)
    return model


def lr_code_snippet():

    return """
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
        [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...    
    """
