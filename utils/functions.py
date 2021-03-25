from matplotlib.pyplot import title
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

import plotly_express as px
import plotly.graph_objs as go
from plotly import tools


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def plot_data(x, y, dataset):

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["class"] = y
    df["class"] = df["class"].map(str)

    # fig = px.scatter(df, x="x1", y="x2", color="class")
    # fig.update_xaxes(range=[x_min, x_max])
    # fig.update_xaxes(range=[y_min, y_max])

    # fig.update_layout(height=450)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x[:, 0],
                y=x[:, 1],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color=y,
                    colorscale="oxy",
                    line=dict(color="white", width=2),
                ),
            )
        ]
    )
    fig.update_layout(title={"text": f"{dataset} dataset"})
    fig.update_xaxes(range=[x_min, x_max], title="x1")
    fig.update_yaxes(range=[y_min, y_max], title="x2")

    return fig


hyperparams = {
    "Logistic Regression": {
        "options": {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "C": [1, 0.8, 0.5, 0.1, 0.01],
            "solver": ["lbfgs"],
        },
        "object": LogisticRegression,
    },
    "Random Forest": {
        "options": {
            "n_estimators": [10, 25, 50, 100, 150, 200, 250, 300],
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


def plot_decision_boundary(model, model_type, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure(
        data=[go.Heatmap(x=xx[0], y=y_, z=Z, colorscale="oxy", showscale=False)],
    )

    trace2 = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode="markers",
        showlegend=True,
        marker=dict(
            size=10, color=y, colorscale="oxy", line=dict(color="white", width=2)
        ),
    )

    fig.add_trace(trace2)
    fig.update_layout(height=450, title={"text": f"Decision boundary of {model_type}"})
    fig.update_xaxes(range=[x_min, x_max], title="x1")
    fig.update_yaxes(range=[y_min, y_max], title="x2")

    return fig
