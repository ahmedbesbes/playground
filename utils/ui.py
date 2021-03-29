import numpy as np
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

from models.NeuralNetwork import nn_param_selector
from models.RandomForet import rf_param_selector
from models.DecisionTree import dt_param_selector
from models.LogisticRegression import lr_param_selector
from models.KNearesNeighbors import knn_param_selector

from models.utils import model_imports, model_urls, model_infos


def introduction():
    st.title("Welcome to playground 🧪")
    st.subheader(
        """
        This is a place where you can get familiar with machine learning models directly from your browser
        """
    )
    st.markdown(
        """
    - 🗂️ Choose a dataset
    - ⚙️ Pick a model and set its hyper-parameters
    - 📉 Train it and check its performance metrics and decision boundary on train and test data
    - 🩺 Diagnose possible overitting and experiment with other settings
    --- 
    """
    )


def dataset_selector():
    dataset_container = st.sidebar.beta_expander("Configure a dataset", True)
    with dataset_container:
        dataset = st.selectbox("Choose a dataset", ("moons", "circles", "blobs"))
        n_samples = st.number_input(
            "Number of samples",
            min_value=50,
            max_value=1000,
            step=10,
            value=300,
        )

        train_noise = st.slider(
            "Set the noise (train data)",
            min_value=0.01,
            max_value=0.2,
            step=0.005,
            value=0.06,
        )
        test_noise = st.slider(
            "Set the noise (test data)",
            min_value=0.01,
            max_value=1.0,
            step=0.005,
            value=train_noise,
        )

        if dataset == "blobs":
            n_classes = st.number_input("centers", 2, 5, 2, 1)
        else:
            n_classes = None

    return dataset, n_samples, train_noise, test_noise, n_classes


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def generate_data(dataset, n_samples, train_noise, test_noise, n_classes):
    if dataset == "moons":
        x_train, y_train = make_moons(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_moons(n_samples=n_samples, noise=test_noise)
    elif dataset == "circles":
        x_train, y_train = make_circles(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_circles(n_samples=n_samples, noise=test_noise)
    elif dataset == "blobs":
        x_train, y_train = make_blobs(
            n_features=2,
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=train_noise * 47 + 0.57,
            random_state=42,
        )
        x_test, y_test = make_blobs(
            n_features=2,
            n_samples=n_samples // 2,
            centers=2,
            cluster_std=test_noise * 47 + 0.57,
            random_state=42,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        # scaler = StandardScaler()
        x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def model_selector():
    model_training_container = st.sidebar.beta_expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Neural Network",
                "K Nearest Neighbors",
            ),
        )

        if model_type == "Logistic Regression":
            model = lr_param_selector()

        elif model_type == "Decision Tree":
            model = dt_param_selector()

        elif model_type == "Random Forest":
            model = rf_param_selector()

        elif model_type == "Neural Network":
            model = nn_param_selector()

        elif model_type == "K Nearest Neighbors":
            model = knn_param_selector()

        # train_button = st.button("Train model")

    return model_type, model


def show_metrics(accuracy, f1, train=True):

    (c0, c1, c2, _, _, _) = st.beta_columns((2, 3, 3, 3, 3, 3))
    (c0, c1, c2, c3) = st.beta_columns((1, 2, 2, 5))

    tag = "train" if train else "test"

    info = f"metrics on {tag}"

    with c0:
        st.warning(info)

    with c1:
        st.info(f"F1 score : {f1}")

    with c2:
        st.info(f"Accuracy : {accuracy}")

    with c3:
        st.markdown(
            """
        This is a place where you can get familiar with machine learning models directly from your browser
        
        
        """
        )


def generate_snippet(model, model_type, n_samples, train_noise, test_noise, dataset):
    train_noise = np.round(train_noise, 3)
    test_noise = np.round(test_noise, 3)

    model_text_rep = repr(model)
    model_import = model_imports[model_type]

    if dataset == "moons":
        dataset_import = "from sklearn.datasets import make_moons"
        train_data_def = (
            f"x_train, y_train = make_moons(n_samples={n_samples}, noise={train_noise})"
        )
        test_data_def = f"x_test, y_test = make_moons(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "circles":
        dataset_import = "from sklearn.datasets import make_circles"
        train_data_def = f"x_train, y_train = make_circles(n_samples={n_samples}, noise={train_noise})"
        test_data_def = f"x_test, y_test = make_circles(n_samples={n_samples // 2}, noise={test_noise})"
    elif dataset == "blobs":
        dataset_import = "from sklearn.datasets import make_blobs"
        train_data_def = f"x_train, y_train = make_blobs(n_samples={n_samples}, clusters=2, noise={train_noise* 47 + 0.57})"
        test_data_def = f"x_test, y_test = make_blobs(n_samples={n_samples // 2}, clusters=2, noise={test_noise* 47 + 0.57})"

    snippet = f"""
    >>> {dataset_import}
    >>> {model_import}
    >>> from sklearn.metrics import accuracy_score, f1_score
    >>> 
    >>> {train_data_def}
    >>> {test_data_def}    
    >>> model = {model_text_rep}
    >>> model.fit(x_train, y_train)
    
    >>> y_train_pred = model.predict(x_train)
    >>> y_test_pred = model.predict(x_test)

    >>> train_accuracy = accuracy_score(y_train, y_train_pred)
    >>> test_accuracy = accuracy_score(y_test, y_test_pred)
    """

    return snippet


def documentation(model_type):
    model_url = model_urls[model_type]
    text = f"** 🔗 Link to scikit-learn official documentation [here]({model_url})**"
    return text


def info(model_type):
    model_info = model_infos[model_type]
    return model_info


def get_max_polynomial_degree():
    return st.sidebar.number_input("Highest polynomial degree", 1, 10, 1, 1)
