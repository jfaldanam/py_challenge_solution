import os

from fastapi import Depends, FastAPI, HTTPException
from minio import Minio

from py_challenge_backend import __version__, logger
from py_challenge_backend.ml.predict import prediction_pipeline
from py_challenge_backend.ml.train import training_pipeline
from py_challenge_backend.models import (
    AnimalCharacteristics,
    ModelPrediction,
    TrainedModelResponse,
    TrainInput,
)
from py_challenge_backend.utils import retrieve_challenge_data

MINIO_CONSOLE = os.environ.get("PY_CHALLENGE_MINIO_CONSOLE")

app = FastAPI(
    title="py_challenge backend",
    description=(
        "A REST API application to run as the backend required to complete the py_challenge proposed at [https://github.com/jfaldanam/py_challenge](https://github.com/jfaldanam/py_challenge)\n\n"
        f"The models will be stored in MinIO at [{MINIO_CONSOLE}]({MINIO_CONSOLE})"
    ),
    version=__version__,
)


def minio_client():
    """Dependency to create a MinIO client

    :return: A MinIO client"""
    minio_service = os.environ.get("PY_CHALLENGE_MINIO_SERVICE")
    minio_access_key = os.environ.get("PY_CHALLENGE_MINIO_ACCESS_KEY")
    minio_secret_key = os.environ.get("PY_CHALLENGE_MINIO_SECRET_KEY")

    if minio_service:
        logger.info("MinIO service configured", minio_service=minio_service)

    yield Minio(
        minio_service,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

    return


@app.get(
    "/models",
    summary="Get the list of models",
    description="This endpoint returns the list of trained models.",
    response_model=list[str],
    tags=["ml"],
)
def models(minio_client: Minio = Depends(minio_client)) -> list[str]:
    bucket = os.environ.get("PY_CHALLENGE_MINIO_BUCKET")
    if not minio_client.bucket_exists(bucket):
        return []
    return [o.object_name.removesuffix("/") for o in minio_client.list_objects(bucket)]


@app.post(
    "/models/train",
    summary="Train a new model",
    description="This endpoint trains a new model.",
    response_model=TrainedModelResponse,
    tags=["ml"],
)
def train_endpoint(
    request: TrainInput, minio_client: Minio = Depends(minio_client)
) -> TrainedModelResponse:
    model_id = f"seed-{request.seed}-datapoints-{request.number_of_datapoints}"

    data = retrieve_challenge_data(n=request.number_of_datapoints, seed=request.seed)
    metrics = training_pipeline(
        model_id=model_id, animal_characteristics=data, minio_client=minio_client
    )

    return TrainedModelResponse(model_id=model_id, metrics=metrics)


@app.post(
    "/models/predict",
    summary="Predict the species of a set of animals",
    description="This endpoint predicts the species of a set of animals using one of the previously trained models.",
    response_model=list[ModelPrediction],
    responses={404: {"description": "Model not found"}},
    tags=["ml"],
)
def predict(
    model_id: str,
    characteristics: list[AnimalCharacteristics],
    minio_client: Minio = Depends(minio_client),
) -> list[ModelPrediction]:
    try:
        return prediction_pipeline(
            model_id=model_id,
            animal_characteristics=characteristics,
            minio_client=minio_client,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
