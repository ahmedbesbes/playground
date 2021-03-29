import streamlit as st
from sklearn.neural_network import MLPClassifier


def nn_param_selector():
    number_hidden_layers = st.number_input("number of hidden layers", 1, 5, 1)

    hidden_layer_sizes = []

    for i in range(number_hidden_layers):
        n_neurons = st.number_input(
            f"Number of neurons at layer {i+1}", 2, 200, 100, 25
        )
        hidden_layer_sizes.append(n_neurons)

    hidden_layer_sizes = tuple(hidden_layer_sizes)
    params = {"hidden_layer_sizes": hidden_layer_sizes}

    model = MLPClassifier(**params)
    return model