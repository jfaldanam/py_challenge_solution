import streamlit as st

from py_challenge_frontend.db import insert_into_db, render_db
from py_challenge_frontend.ml import list_models, predict_model, train_model
from py_challenge_frontend.models import (
    AnimalCharacteristics,
)

st.title("Demo solution for py_challenge")

with st.sidebar:
    st.markdown("### Models")
    models = list_models()
    st.markdown("List of trained models:")
    st.json(models)
    st.markdown("### Train a model")
    n = st.slider(
        "Number of data points", min_value=500, max_value=1000, step=10, value=500
    )
    seed = st.slider("Seed", min_value=0, max_value=100, step=1, value=42)

    if st.button("Train the model"):
        trained_model = train_model(n, seed)
        st.markdown(f"""
Model with id `{trained_model.model_id}` trained successfully! Here are the results:
- Accuracy: {trained_model.metrics.accuracy}
- Precision: {trained_model.metrics.precision}
- Recall: {trained_model.metrics.recall}
- F1 Score: {trained_model.metrics.f1}
""")

model_id = st.text_input("Model ID", placeholder="seed-X-datapoints-Y")

st.markdown("Data to predict")
col1, col2 = st.columns(2)
with col1:
    walks_on_n_legs = st.slider(
        "Walks on N legs", min_value=0, max_value=5, step=1, value=2
    )
    has_tail = st.checkbox("Has tail")
    has_wings = st.checkbox("Has wings")

with col2:
    height = st.slider("Height", min_value=0.0, max_value=10.0, step=0.1, value=2.0)
    weight = st.slider("Weight", min_value=0.0, max_value=1000.0, step=0.5, value=50.0)

if st.button("Predict", disabled=not model_id):
    data = AnimalCharacteristics(
        walks_on_n_legs=walks_on_n_legs,
        height=height,
        weight=weight,
        has_wings=has_wings,
        has_tail=has_tail,
    )
    predictions = predict_model(model_id, [data])
    for prediction in predictions:
        st.markdown(f"Predicted species: `{prediction.species}`")
        st.markdown("Model confidence per class:")
        st.json(prediction.probabilities)

    insert_into_db(data, prediction)

render_db()
