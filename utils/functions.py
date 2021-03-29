import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import plotly.graph_objs as go
import plotly_express as px


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def plot_data(x, y, dataset):

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["class"] = y
    df["class"] = df["class"].map(str)

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


def plot_decision_boundary(model, model_type, x_train, y_train, x_test, y_test):
    d = x_train.shape[1]

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    model_input = [(xx.ravel() ** p, yy.ravel() ** p) for p in range(1, d // 2 + 1)]
    aux = []
    for c in model_input:
        aux.append(c[0])
        aux.append(c[1])

    print(len(aux))

    # Z = model.predict(np.c_[aux])

    Z = model.predict(np.concatenate([v.reshape(-1, 1) for v in aux], axis=1))

    Z = Z.reshape(xx.shape)

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=xx[0],
                y=y_,
                z=Z,
                # colorscale="oxy",
                colorscale=["tomato", "rgb(27,158,119)"],
                showscale=False,
            )
        ],
    )

    trace1 = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        name="train data",
        mode="markers",
        showlegend=True,
        marker=dict(
            size=10,
            color=y_train,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(trace1)

    trace2 = go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        name="test data",
        mode="markers",
        showlegend=True,
        marker_symbol="cross",
        visible="legendonly",
        marker=dict(
            size=10,
            color=y_test,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )
    fig.add_trace(trace2)

    fig.update_layout(
        height=500,
        title={"text": f"Decision boundary of {model_type}"},
        # margin=dict(b=20, t=100, r=100),
    )
    fig.update_xaxes(range=[x_min, x_max], title="x1")
    fig.update_yaxes(range=[y_min, y_max], title="x2")

    return fig


def train_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)

    df_metrics = pd.DataFrame()
    df_metrics["dataset"] = ["train", "test"]
    df_metrics["Accuracy"] = [train_accuracy, test_accuracy]
    df_metrics["F1 Score"] = [train_f1, test_f1]
    df_metrics.set_index("dataset", inplace=True)

    return model, train_accuracy, train_f1, test_accuracy, test_f1, df_metrics


def display_kpis(train_metric, test_metric, label):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=test_metric,
            title={"text": f"{label} (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": train_metric},
        )
    )
    fig.update_layout(
        # width=400,
        height=160,
        margin=dict(l=0, r=0, b=5, t=50, pad=0),
    )

    return fig