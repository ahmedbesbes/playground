import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier


def gb_param_selector():
    learning_rate = st.slider("learning_rate", 0.001, 0.5, 0.1, 0.005)
    n_estimators = st.number_input("n_estimators", 10, 500, 100, 10)
    max_depth = st.number_input("max_depth", 3, 30, 3, 1)

    params = {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    model = GradientBoostingClassifier(**params)
    return model
