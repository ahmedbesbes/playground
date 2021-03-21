import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import f1_score, accuracy_score
from utils.functions import plot_data, set_model

from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt


st.title("Welcome to Playground")
st.info(
    "Discover how hyperparameters impact the performance of your model on unseen and corrupt data"
)


dataset_container = st.sidebar.beta_expander("Configure a dataset")
with dataset_container:
    dataset = st.selectbox("Choose a dataset", ("moons", "circles", "blobs"))
    n_samples = st.number_input(
        "Number of samples",
        min_value=50,
        max_value=1000,
        step=10,
        value=300,
    )

    noise = st.slider("Set the noise", min_value=0.01, max_value=0.2, step=0.005)

    if dataset == "moons":
        x, y = make_moons(n_samples=n_samples, noise=noise)
    elif dataset == "circles":
        x, y = make_circles(n_samples=n_samples, noise=noise)
    elif dataset == "blobs":
        x, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise * 47 + 0.57)


plot_data(x, y, dataset)

placeholder = st.empty()

decision_boundary_placeholder = st.empty()


model_training_container = st.sidebar.beta_expander("Train a model")


with model_training_container:
    model_type = st.selectbox(
        "Choose a model", ("Logistic Regression", "Random Forest", "SVM")
    )
    model = set_model(model_type)

    train_button = st.button("Train model")
    if train_button:
        model.fit(x, y)
        y_pred = model.predict(x)
        f1 = np.round(f1_score(y, y_pred), 3)
        accuracy = np.round(accuracy_score(y, y_pred), 3)

        metrics = pd.DataFrame()
        metrics["f1"] = [f1]
        metrics["accuracy"] = [accuracy]

        (c0, c1, c2) = placeholder.beta_columns((1, 1, 1))

        with c0:
            st.warning("metrics on train")

        with c1:
            st.info(f"F1 score : {f1}")

        with c2:
            st.info(f"Accuracy : {accuracy}")

        fig = plt.figure()
        # Plotting decision regions
        plot_decision_regions(x, y, clf=model, legend=2)

        # Adding axes annotations
        plt.xlabel("sepal length [cm]")
        plt.ylabel("petal length [cm]")
        plt.title("SVM on Iris")

        decision_boundary_placeholder.pyplot(fig)
