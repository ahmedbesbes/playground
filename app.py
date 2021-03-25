from utils.ui import dataset_selector, model_selector
from models.LogisticRegression import lr_code_snippet, lr_param_selector
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import f1_score, accuracy_score
from utils.functions import plot_data, plot_decision_boundary, set_model

from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt

st.set_page_config(layout="wide")


st.title("Welcome to Playground")
st.info(
    "Discover how hyperparameters impact the performance of your model on unseen and corrupt data"
)

col1, col2 = st.beta_columns((1, 1))

x, y, dataset = dataset_selector()

with col1:
    plot_placeholder = st.empty()

with col2:
    doc_placeholder = st.empty()
    code_header = st.empty()
    code_placeholder = st.empty()


fig_data = plot_data(x, y, dataset)

plot_placeholder.plotly_chart(fig_data, use_container_width=True)


train_metrics_placeholder = st.empty()

train_button, model_type, model, snippet = model_selector()


if train_button:
    code_header.header("**Code snippet**")
    code_placeholder.code(snippet)
    doc_placeholder.markdown("Link to scikit learn official doc here ")

    model.fit(x, y)
    y_pred = model.predict(x)
    f1 = np.round(f1_score(y, y_pred), 3)
    accuracy = np.round(accuracy_score(y, y_pred), 3)

    metrics = pd.DataFrame()
    metrics["f1"] = [f1]
    metrics["accuracy"] = [accuracy]

    (c0, c1, c2) = train_metrics_placeholder.beta_columns((1, 1, 1))

    with c0:
        st.warning("metrics on train")

    with c1:
        st.info(f"F1 score : {f1}")

    with c2:
        st.info(f"Accuracy : {accuracy}")

    # decision_boundary_placeholder.pyplot(fig)

    fig = plot_decision_boundary(model, model_type, x, y)

    plot_placeholder.plotly_chart(fig, use_container_width=True)
