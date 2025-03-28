import os

import requests

from py_challenge_frontend.models import (
    Animal,
    AnimalCharacteristics,
    ModelPrediction,
    TrainedModelResponse,
)

DATA_SERVICE = os.environ.get("PY_CHALLENGE_DATA_SERVICE")
DATA_ENDPOINT = f"{DATA_SERVICE}/api/v1/animals/data"
BACKEND_SERVICE = os.environ.get("PY_CHALLENGE_BACKEND_SERVICE")
TRAIN_ENDPOINT = f"{BACKEND_SERVICE}/models/train"
PREDICT_ENDPOINT = f"{BACKEND_SERVICE}/models/predict"
LIST_MODELS_ENDPOINT = f"{BACKEND_SERVICE}/models"


def list_models() -> list[str]:
    response = requests.get(LIST_MODELS_ENDPOINT)
    response.raise_for_status()
    if response.ok:
        return response.json()


def retrieve_challenge_data(n: int, seed: int) -> list[AnimalCharacteristics]:
    """Retrieve a batch of challenge data from the data service

    :param n: The number of data points to retrieve
    :param seed: The seed to use for the random number generator

    :return: A list of AnimalCharacteristics instances"""
    response = requests.post(
        DATA_ENDPOINT, json={"number_of_datapoints": n, "seed": seed}
    )
    response.raise_for_status()
    if response.ok:
        return [AnimalCharacteristics(**data) for data in response.json()]


def train_model(n: int, seed: int) -> TrainedModelResponse:
    response = requests.post(
        TRAIN_ENDPOINT, json={"number_of_datapoints": n, "seed": seed}
    )
    response.raise_for_status()
    if response.ok:
        return TrainedModelResponse(**response.json())


def predict_model(model_id: str, data: list[AnimalCharacteristics]) -> list[Animal]:
    response = requests.post(
        PREDICT_ENDPOINT,
        params={
            "model_id": model_id,
        },
        json=[d.model_dump() for d in data],
    )
    response.raise_for_status()
    if response.ok:
        return [ModelPrediction(**data) for data in response.json()]
