from sklearn import svm
import streamlit as st
from sklearn.svm import SVC


def svc_param_selector():
    C = st.number_input("C", 0.01, 2.0, 1.0, 0.01)
    kernel = st.selectbox("kernel", ("rbf", "linear", "poly", "sigmoid"))
    params = {"C": C, "kernel": kernel}
    model = SVC(**params)
    return model