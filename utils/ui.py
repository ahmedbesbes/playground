from models.NeuralNetwork import nn_code_snippet, nn_param_selector
from models.RandomForet import rf_code_snippet, rf_param_selector
from models.DecisionTree import dt_code_snippet, dt_param_selector
from models.LogisticRegression import lr_code_snippet, lr_param_selector
from sklearn import datasets
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs


def dataset_selector():
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

    return dataset, n_samples, noise


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def generate_dataset(dataset, n_samples, noise):
    if dataset == "moons":
        x, y = make_moons(n_samples=n_samples, noise=noise)
    elif dataset == "circles":
        x, y = make_circles(n_samples=n_samples, noise=noise)
    elif dataset == "blobs":
        x, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise * 47 + 0.57)
    return x, y, dataset


def model_selector():
    model_training_container = st.sidebar.beta_expander("Train a model")
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            ("Logistic Regression", "Decision Tree", "Random Forest", "Neural Network"),
        )

        if model_type == "Logistic Regression":
            model = lr_param_selector()
            snippet = lr_code_snippet()
        elif model_type == "Decision Tree":
            model = dt_param_selector()
            snippet = dt_code_snippet()
        elif model_type == "Random Forest":
            model = rf_param_selector()
            snippet = rf_code_snippet()
        elif model_type == "Neural Network":
            model = nn_param_selector()
            snippet = nn_code_snippet()

        train_button = st.button("Train model")

    return train_button, model_type, model, snippet
