import streamlit as st
from sklearn.neural_network import MLPClassifier


def nn_param_selector():
    number_hidden_layers = st.number_input("number of hidden layers", 1, 5, 1)

    hidden_layer_sizes = []

    for i in range(number_hidden_layers):
        n_neurons = st.number_input(f"Number of neurons at layer {i+1}", 2, 200, 100, 1)
        hidden_layer_sizes.append(n_neurons)

    hidden_layer_sizes = tuple(hidden_layer_sizes)
    params = {"hidden_layer_sizes": hidden_layer_sizes}

    model = MLPClassifier(**params)
    return model


def nn_code_snippet():
    return """
    >>> from sklearn.neural_network import MLPClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    ...                                                     random_state=1)
    >>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    >>> clf.predict_proba(X_test[:1])
    array([[0.038..., 0.961...]])
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
