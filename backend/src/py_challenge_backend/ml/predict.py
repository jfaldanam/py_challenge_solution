import os

import numpy as np
import onnxruntime as rt
import pandas as pd
from minio import Minio, error

from py_challenge_backend import logger
from py_challenge_backend.models import (
    AnimalCharacteristics,
    AnimalSpecies,
    ModelPrediction,
)


def prediction_pipeline(
    model_id: str,
    animal_characteristics: list[AnimalCharacteristics],
    minio_client: Minio,
) -> list[ModelPrediction]:
    df = pd.DataFrame([ac.model_dump() for ac in animal_characteristics])

    return predict(model_id, df.to_numpy(), minio_client)


def predict(
    model_id: str, x_test: np.ndarray, minio_client: Minio
) -> list[ModelPrediction]:
    bucket = os.environ.get("PY_CHALLENGE_MINIO_BUCKET")
    try:
        onnx_model = minio_client.get_object(bucket, f"{model_id}/model.onnx").data
    except error.S3Error:
        logger.error("Model not found", model_id=model_id)
        raise ValueError(f"Model not trained for model_id: {model_id}")
    sess = rt.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

    # Ensure the input is float32
    x_test = x_test.astype(np.float32)
    # Get inputs name
    input_name = sess.get_inputs()[0].name
    # Run inference
    y_pred_label, y_pred_prob = sess.run(None, {input_name: x_test})

    object_predictions = [AnimalSpecies(p) for p in y_pred_label]

    model_response = []
    for label, prob_dict in zip(object_predictions, y_pred_prob):
        model_response.append(
            ModelPrediction(
                species=AnimalSpecies(label),
                probabilities={AnimalSpecies(k): v for k, v in prob_dict.items()},
            )
        )

    return model_response
