import streamlit as st
from sklearn.linear_model import Perceptron


def lp_param_selector():
    eta0 = st.slider("learning_rate", 0.001, 10.0, step=0.001, value=1.0)
    max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100)

    penalty = st.selectbox("penalty", options=["None", "l2", "l1", "elasticnet"])

    if penalty in ["l2", "l1", "elasticnet"]:
        alpha = st.slider("alpha", 0.00001, 0.001, step=0.00001, value=0.0001)
    else:
        alpha = 0.0001

    early_stopping = st.checkbox('early_stopping', value=False)

    if early_stopping:
        validation_fraction = st.number_input("validation_fraction", 0.0, 1.0, step=0.05, value=0.1)
        n_iter_no_change = st.number_input("n_iter_no_change", 2, 100, step=1, value=5)
    else:
        validation_fraction = 0.1
        n_iter_no_change = 5

    params = {"eta0": eta0, 
              "max_iter": max_iter, 
              "penalty": penalty, 
              "alpha": alpha, 
              "early_stopping": early_stopping,
              "validation_fraction": validation_fraction, 
              "n_iter_no_change": n_iter_no_change}

    model = Perceptron(**params)
    return model
