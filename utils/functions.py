import pandas as pd
import streamlit as st
import altair as alt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def plot_data(x, y, dataset):
    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["class"] = y

    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x="x1",
            y="x2",
            color="class:N",
            tooltip=["x1", "x2", "class"],
        )
        .properties(title=f"Visualization of the {dataset} dataset")
    )

    st.altair_chart(chart, use_container_width=True)


hyperparams = {
    "Logistic Regression": {
        "options": {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "C": [1, 0.8, 0.5, 0.1, 0.01],
        },
        "object": LogisticRegression,
    },
    "Random Forest": {
        "options": {
            "n_estimators": [100, 150, 200, 250, 300],
            "max_depth": [None, 1, 3, 5, 10],
            "max_features": ["auto", "sqrt", "log2"],
        },
        "object": RandomForestClassifier,
    },
    "SVM": {
        "options": {
            "C": [1, 5, 10, 15],
        },
        "object": SVC,
    },
}


def set_model(model_type):
    options = hyperparams[model_type]["options"]
    params = {}
    for param, values in options.items():
        selected_parameters = st.selectbox(param, values)
        params[param] = selected_parameters

    model_class = hyperparams[model_type]["object"]
    model = model_class(**params)
    return model
