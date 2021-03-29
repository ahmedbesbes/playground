import numpy as np
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
    C = np.round(C, 3)
    max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100)

    params = {"solver": solver, "penalty": penalty, "C": C, "max_iter": max_iter}

    model = LogisticRegression(**params)
    return model
