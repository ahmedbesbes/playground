import numpy as np
import streamlit as st
from utils.functions import (
    display_kpis,
    local_css,
    plot_data,
    plot_decision_boundary_and_metrics,
    train_model,
)

from utils.ui import (
    dataset_selector,
    documentation,
    generate_snippet,
    generate_data,
    get_max_polynomial_degree,
    info,
    introduction,
    model_selector,
)


st.set_page_config(layout="wide")
local_css("./css/style.css")


introduction()

col1, col2 = st.beta_columns((1, 1))

dataset, n_samples, train_noise, test_noise, n_classes = dataset_selector()

x_train, y_train, x_test, y_test = generate_data(
    dataset, n_samples, train_noise, test_noise, n_classes
)


with col1:
    plot_placeholder = st.empty()

with col2:
    execution_time = st.empty()
    doc_placeholder = st.empty()
    code_header = st.empty()
    code_placeholder = st.empty()
    info_placeholder = st.empty()
    tips_header = st.empty()
    tips = st.empty()


fig_data = plot_data(x_train, y_train, dataset)

plot_placeholder.plotly_chart(fig_data, use_container_width=True)


model_type, model = model_selector()


st.sidebar.header("Feature engineering")
degree = get_max_polynomial_degree()

for d in range(2, degree + 1):
    x_train = np.concatenate(
        (x_train, x_train[:, 0].reshape(-1, 1) ** d, x_train[:, 1].reshape(-1, 1) ** d),
        axis=1,
    )
    x_test = np.concatenate(
        (x_test, x_test[:, 0].reshape(-1, 1) ** d, x_test[:, 1].reshape(-1, 1) ** d),
        axis=1,
    )


doc_text = documentation(model_type)
doc_placeholder.markdown(doc_text)


(
    model,
    train_accuracy,
    train_f1,
    test_accuracy,
    test_f1,
    duration,
) = train_model(model, x_train, y_train, x_test, y_test)


metrics = {
    "train_accuracy": train_accuracy,
    "train_f1": train_f1,
    "test_accuracy": test_accuracy,
    "test_f1": test_f1,
}


snippet = generate_snippet(
    model, model_type, n_samples, train_noise, test_noise, dataset, degree
)
code_header.header("**Retrain the same model in Python**")
execution_time.warning(f"Training took {duration:.3f} seconds")
code_placeholder.code(snippet)

model_info = info(model_type)

col3, col4, col5 = st.beta_columns((1, 1, 2))


# with col3:
#     kpi_plot = display_kpis(train_accuracy, test_accuracy, "Accuracy")
#     st.plotly_chart(kpi_plot, True)

# with col4:
#     kpi_plot = display_kpis(train_f1, test_f1, "F1 Score")
#     st.plotly_chart(kpi_plot, True)


tips_header.header(f"**Tips on the {model_type} 💡 **")
tips.info(model_info)


fig = plot_decision_boundary_and_metrics(
    model, model_type, x_train, y_train, x_test, y_test, metrics
)

plot_placeholder.plotly_chart(fig, True)