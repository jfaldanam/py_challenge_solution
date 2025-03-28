import io
import os

import pandas as pd
from minio import Minio, error
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from py_challenge_backend import logger
from py_challenge_backend.models import (
    Animal,
    AnimalCharacteristics,
    AnimalSpecies,
    ModelMetrics,
)


def training_pipeline(
    model_id: str,
    animal_characteristics: list[AnimalCharacteristics],
    minio_client: Minio,
) -> ModelMetrics:
    """Train a model using the provided animal characteristics

    :param model_id: The ID of the model
    :param animal_characteristics: A list of animal characteristics
    :param minio_client: A MinIO client to store the trained model

    :return: The metrics of the trained model"""

    bucket = os.environ.get("PY_CHALLENGE_MINIO_BUCKET")
    if minio_client.bucket_exists(bucket):
        try:
            # Check if the model already exists by trying to retrieve the metrics
            metrics = minio_client.get_object(bucket, f"{model_id}/metrics.json")
            logger.info("Model already exists, skipping training", model_id=model_id)
            return ModelMetrics(**metrics.json())
        except error.S3Error:
            logger.info("Model not found, training a new model", model_id=model_id)

    assert len(animal_characteristics) >= 500, (
        "At least one hundred animal characteristics must be provided to train a good model"
    )

    # As data is unlabelled, use clustering to group similar animals
    training_animals = label_animals(animal_characteristics)

    # Train a model using the labelled data
    metrics = train_model(model_id, training_animals, minio_client)

    return metrics


def label_animals(animal_characteristics: list[AnimalCharacteristics]) -> list[Animal]:
    """Label a set of unknown animals by using clustering and domain knowledge.

    The DBSCAN algorithm is well-known, density-based clustering algorithm: given a set
    of points in some space, it groups together points that are closely packed together
    (points with many nearby neighbors), allowing to identify outliers.

    :param animal_characteristics: A list containing the characteristics of the animals to be labelled

    :return: A list of the labelled animals"""
    df = pd.DataFrame([ac.model_dump() for ac in animal_characteristics])
    dbscan = DBSCAN(eps=0.5, min_samples=20)
    df["cluster_id"] = dbscan.fit_predict(df.loc[:, ["walks_on_n_legs", "height"]])

    # Detect which cluster is which animal
    assert df["cluster_id"].nunique() == len(AnimalSpecies), (
        f"Expected {len(AnimalSpecies)} clusters, got {df['cluster_id'].nunique()}"
    )

    # Assign the cluster labels to the animals

    # Chickens: filter the dataframe for elements with 2 legs and have wings
    chicken_df = df[(df["walks_on_n_legs"] == 2) & (df["has_wings"])]

    # Kangaroo: filter the dataframe for elements with 2 legs and do not have wings
    kangaroo_df = df[(df["walks_on_n_legs"] == 2) & (~df["has_wings"])]

    # Elephants: filter the dataframe for elements with 4 legs and the heaviest weight
    # Filter the top 10% of the heaviest animals
    # FIXME: This will fail if there are a very uneven distribution of animals with 4 legs
    elephant_df = df[(df["walks_on_n_legs"] == 4)].sort_values(
        by="weight", ascending=False
    )[: int(0.1 * len(df))]

    # Dogs: The remaining cluster

    # Find the cluster with the highest count of each animal
    chicken_cluster_id = chicken_df["cluster_id"].mode()[0]
    kangaroo_cluster_id = kangaroo_df["cluster_id"].mode()[0]
    elephant_cluster_id = elephant_df["cluster_id"].mode()[0]

    # Get the last cluster that must be dogs
    clusters = set(df["cluster_id"])

    # Remove known clusters
    clusters.remove(-1)
    clusters.remove(chicken_cluster_id)
    clusters.remove(kangaroo_cluster_id)
    clusters.remove(elephant_cluster_id)

    # The last cluster must be dogs
    dog_cluster_id = clusters.pop()

    df["cluster_names"] = df["cluster_id"].replace(
        {
            chicken_cluster_id: AnimalSpecies.CHICKEN,
            kangaroo_cluster_id: AnimalSpecies.KANGAROO,
            elephant_cluster_id: AnimalSpecies.ELEPHANT,
            dog_cluster_id: AnimalSpecies.DOG,
            -1: AnimalSpecies.UNKNOWN,
        }
    )

    # Create return objects
    result = []
    for _, row in df.iterrows():
        result.append(
            Animal(
                species=AnimalSpecies(row["cluster_names"]),
                characteristics=AnimalCharacteristics(
                    **row.drop(["cluster_id", "cluster_names"]).to_dict()
                ),
            )
        )

    return result


def train_model(
    model_id: str,
    training_animals: list[Animal],
    minio_client: Minio,
    train_split: float = 0.8,
) -> ModelMetrics:
    """Train a model using the provided labelled data

    :param model_id: The ID of the model
    :param training_animals: A list of labelled animals
    :param minio_client: A MinIO client to store the trained model
    :param train_split: The proportion of the data to use for training

    :return: The metrics of the trained model"""
    x_list = []
    y_list = []
    for animal in training_animals:
        x_list.append(animal.characteristics.model_dump())
        y_list.append(animal.species.value)

    X = pd.DataFrame(x_list)
    Y = pd.Series(y_list)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_split)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)

    logger.info("Model trained", **metrics.model_dump())

    store_model(
        model_id, model, metrics, x_train, x_test, y_train, y_test, minio_client
    )

    return metrics


def evaluate_model(
    model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> ModelMetrics:
    """Evaluate a model using the provided test data"

    :param model: The model to evaluate
    :param x_test: The test data
    :param y_test: The test labels

    :return: The metrics of the model
    """
    metrics = {}
    # Predict and evaluate the model
    y_pred = model.predict(x_test)
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["recall"] = recall_score(y_test, y_pred, average="weighted")
    metrics["precision"] = precision_score(y_test, y_pred, average="weighted")
    metrics["f1"] = f1_score(y_test, y_pred, average="weighted")
    metrics["confusion"] = confusion_matrix(y_test, y_pred)

    return ModelMetrics(**metrics)


def store_model(
    model_id: str,
    model: RandomForestClassifier,
    metrics: ModelMetrics,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    minio_client: Minio,
) -> None:
    """Store the trained model, the data used for training and its metrics in MinIO"

    :param model_id: The ID of the model
    :param model: The trained model
    :param metrics: The metrics of the model
    :param x_train: The training data
    :param x_test: The testing data
    :param y_train: The training labels
    :param y_test: The testing labels
    :param minio_client: A MinIO client to store the trained
    """
    bucket = os.environ.get("PY_CHALLENGE_MINIO_BUCKET")
    if not minio_client.bucket_exists(bucket):
        minio_client.make_bucket(bucket)

    # Training input is used to infer input types
    onnx = to_onnx(model, [("input", FloatTensorType([None, x_train.shape[1]]))])

    # Store the model
    serialized_onnx = onnx.SerializeToString()
    minio_client.put_object(
        bucket,
        f"{model_id}/model.onnx",
        io.BytesIO(serialized_onnx),
        len(serialized_onnx),
        content_type="application/octet-stream",
    )
    # Store the metrics
    serialized_metrics = metrics.model_dump_json().encode()
    minio_client.put_object(
        bucket,
        f"{model_id}/metrics.json",
        io.BytesIO(serialized_metrics),
        len(serialized_metrics),
        content_type="application/json",
    )
    # Store the training and testing data
    serialized_x_train = x_train.to_csv(index=False).encode()
    minio_client.put_object(
        bucket,
        f"{model_id}/x_train.csv",
        io.BytesIO(serialized_x_train),
        len(serialized_x_train),
        content_type="text/csv",
    )
    serialized_x_test = x_test.to_csv(index=False).encode()
    minio_client.put_object(
        bucket,
        f"{model_id}/x_test.csv",
        io.BytesIO(serialized_x_test),
        len(serialized_x_test),
        content_type="text/csv",
    )
    serialized_y_train = y_train.to_csv(index=False).encode()
    minio_client.put_object(
        bucket,
        f"{model_id}/y_train.csv",
        io.BytesIO(serialized_y_train),
        len(serialized_y_train),
        content_type="text/csv",
    )
    serialized_y_test = y_test.to_csv(index=False).encode()
    minio_client.put_object(
        bucket,
        f"{model_id}/y_test.csv",
        io.BytesIO(serialized_y_test),
        len(serialized_y_test),
        content_type="text/csv",
    )
