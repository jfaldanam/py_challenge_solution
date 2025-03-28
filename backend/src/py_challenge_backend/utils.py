import os

import requests

from py_challenge_backend.models import AnimalCharacteristics

HOST = os.environ.get("PY_CHALLENGE_DATA_SERVICE", "http://localhost:8777")
DATA_ENDPOINT = f"{HOST}/api/v1/animals/data"


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
